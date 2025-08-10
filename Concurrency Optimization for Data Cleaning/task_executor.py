import json
import asyncio
import time
from pathlib import Path
from openai import AsyncOpenAI, APIConnectionError
import tqdm.asyncio as ta
from data_models import DataPoint_T


class ProgressMonitor:
    """Monitors the progress of task completion."""

    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.error_count = 0
        self.timestep = 0
        self.record_queue = asyncio.Queue()
        self._lock = asyncio.Lock()

    async def increment_progress(self, errors: int = 0):
        async with self._lock:
            self.completed_tasks += 1
            if errors > 0:
                self.error_count += errors

    async def get_completion_rate(self) -> float:
        async with self._lock:
            return self.completed_tasks / self.total_tasks if self.total_tasks > 0 else 0


class TaskExecutor:
    """Manages the creation, execution, and result collection of data cleaning tasks."""

    def __init__(self, sema, async_client: AsyncOpenAI, config: dict, model_name: str):
        self.semaphore = sema
        self.async_client = async_client
        self.config = config
        self.model_name = model_name

        self.tasks = []
        self.output_path = Path(self.config['output_chunk_dir'])
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.input_path = Path(self.config['input_chunk_dir'])
        self.input_files = [str(f) for f in self.input_path.glob("*.json")][:self.config['num_chunks_to_process']]

        self.start_time = 0.0
        self.monitor: ProgressMonitor = None

        print("Loading original reference data...")
        with open(self.config['original_data_path'], "r", encoding="utf-8") as f:
            self.journal_data = json.load(f)
        print("Reference data loaded.")

    def get_total_tasks(self) -> int:
        return len(self.tasks) if self.tasks else 0

    def prepare_tasks(self):
        print("Preparing data cleaning tasks...")
        paper_data = [paper for journal in self.journal_data.values() for paper_list in journal.values() for paper in
                      paper_list]

        task_pairs = []
        paper_index = 0
        for filename in self.input_files:
            with open(filename, "r", encoding="utf-8") as f:
                chunk = json.load(f)
                for item in chunk:
                    if paper_index >= len(paper_data):
                        print("Warning: Ran out of reference articles. Some questions may not be processed.")
                        break
                    article = paper_data[paper_index]
                    paper_index += 1
                    for question in item.get("Questions", []):
                        clean_question = question.replace("<Question:", "").replace(">", "")
                        task_pairs.append({"question": clean_question, "article": article})
            if paper_index >= len(paper_data): break

        for pair in task_pairs:
            task = asyncio.create_task(self._process_item(pair["question"], pair["article"]))
            self.tasks.append(task)

        self.monitor = ProgressMonitor(total_tasks=len(self.tasks))
        print(f"[EXECUTOR] Prepared {self.get_total_tasks()} tasks.")

    async def _monitor_progress(self, interval: float = 5.0):
        print("[EXECUTOR] Progress monitor started.")
        while True:
            completion_rate = await self.monitor.get_completion_rate()
            elapsed = time.time() - self.start_time

            print(
                f"\n[PROGRESS] Completed: {self.monitor.completed_tasks}/{self.monitor.total_tasks} "
                f"({completion_rate:.2%}) | Errors: {self.monitor.error_count} | Elapsed: {elapsed:.1f}s"
            )

            record = DataPoint_T(
                timestamp_t=time.time(),
                timestep=self.monitor.timestep,
                completion_rate=completion_rate
            )
            await self.monitor.record_queue.put(record)

            if self.monitor.completed_tasks >= self.monitor.total_tasks:
                break

            await asyncio.sleep(interval)
            self.monitor.timestep += interval

    async def _process_item(self, question: str, article: str) -> tuple[str, int]:
        prompt = (
            f"You are the author of this article, and your chain of thought represents the research approach of this article.\n"
            f"Please answer the given question based on the content of the article.\n"
            pre-training"Please reason step by step in English, add show all the possible details, to conduct a PhD-level Chain of Thought.\n"
            f"Question: {question}\n"
        f"Article: {article}"
        )

        try:
            async with self.semaphore:
                completion = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                )

            await self.monitor.increment_progress()
            return completion.choices[0].message.content, completion.usage.completion_tokens

        except APIConnectionError:
            print("Error: vLLM connection timeout.")
            await self.monitor.increment_progress(errors=1)
            return "<VLLM_TIMEOUT_ERROR>", 0
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            await self.monitor.increment_progress(errors=1)
            return f"<ERROR: {str(e)}>", 0

    def write_results(self, completed_tasks):
        print("Writing results to output directory...")
        output_file = self.output_path / "processed_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            results_to_save = [res[0] for res in completed_tasks if isinstance(res, tuple)]
            json.dump(results_to_save, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file}")

    async def run_all_tasks(self):
        print("[EXECUTOR] Starting task execution...")
        self.start_time = time.time()
        monitor_task = asyncio.create_task(self._monitor_progress())

        completed_tasks = []
        try:
            completed_tasks = await ta.tqdm_asyncio.gather(*self.tasks)
            print("[EXECUTOR] All tasks finished.")
        finally:
            await asyncio.sleep(5)
            monitor_task.cancel()
            self.write_results(completed_tasks)