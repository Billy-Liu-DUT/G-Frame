"""v1 task runner that now preserves structured provenance at its output boundary."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from openai import APIConnectionError, AsyncOpenAI
import tqdm.asyncio as ta

from data_models import DataPoint_T


class ProgressMonitor:
    """Monitors task completion and emits the original ``DataPoint_T`` format."""

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
            self.error_count += max(errors, 0)

    async def get_completion_rate(self) -> float:
        async with self._lock:
            return self.completed_tasks / self.total_tasks if self.total_tasks else 0.0


class TaskExecutor:
    """Create legacy requests while writing non-trainable structured records.

    A one-call v1 response has not completed the Team Game review cycle, so it
    is deliberately marked ``sft_ready: false``. Feed it through the v2 Team
    Game before adding it to an SFT dataset.
    """

    def __init__(self, sema, async_client: AsyncOpenAI, config: dict, model_name: str):
        self.semaphore = sema
        self.async_client = async_client
        self.config = config
        self.model_name = model_name
        self.tasks: List[asyncio.Task] = []
        self.output_path = Path(self.config["output_chunk_dir"])
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.input_path = Path(self.config["input_chunk_dir"])
        self.input_files = [str(path) for path in self.input_path.glob("*.json")][: self.config["num_chunks_to_process"]]
        self.start_time = 0.0
        self.monitor: ProgressMonitor | None = None
        with open(self.config["original_data_path"], "r", encoding="utf-8") as handle:
            self.journal_data = json.load(handle)

    def get_total_tasks(self) -> int:
        return len(self.tasks)

    @staticmethod
    def _iter_articles(value: Any) -> Iterable[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from TaskExecutor._iter_articles(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from TaskExecutor._iter_articles(item)

    def prepare_tasks(self):
        """Pair source passages and questions with stable task/source identifiers."""
        articles = list(self._iter_articles(self.journal_data))
        task_pairs: List[Dict[str, str]] = []
        paper_index = 0
        for filename in self.input_files:
            with open(filename, "r", encoding="utf-8") as handle:
                chunk = json.load(handle)
            if not isinstance(chunk, list):
                raise ValueError(f"input chunk {filename} must contain a JSON list")
            for item_index, item in enumerate(chunk):
                if paper_index >= len(articles):
                    print("warning: ran out of reference articles")
                    break
                article = articles[paper_index]
                source_id = f"legacy-paper-{paper_index:06d}"
                paper_index += 1
                questions = item.get("Questions", []) if isinstance(item, dict) else []
                for question_index, question in enumerate(questions):
                    if not isinstance(question, str) or not question.strip():
                        continue
                    clean_question = question.replace("<Question:", "").replace(">", "").strip()
                    task_pairs.append(
                        {
                            "task_id": f"{Path(filename).stem}-{item_index}-{question_index}",
                            "source_id": source_id,
                            "question": clean_question,
                            "source": article,
                        }
                    )
            if paper_index >= len(articles):
                break
        self.tasks = [asyncio.create_task(self._process_item(pair)) for pair in task_pairs]
        self.monitor = ProgressMonitor(total_tasks=len(self.tasks))
        print(f"[EXECUTOR] Prepared {self.get_total_tasks()} tasks.")

    async def _monitor_progress(self, interval: float = 5.0):
        if self.monitor is None:
            raise RuntimeError("prepare_tasks must run before monitoring")
        while True:
            completion_rate = await self.monitor.get_completion_rate()
            record = DataPoint_T(timestamp_t=time.time(), timestep=self.monitor.timestep, completion_rate=completion_rate)
            await self.monitor.record_queue.put(record)
            if self.monitor.completed_tasks >= self.monitor.total_tasks:
                return
            await asyncio.sleep(interval)
            self.monitor.timestep += int(interval)

    async def _process_item(self, pair: Dict[str, str]) -> Dict[str, Any]:
        if self.monitor is None:
            raise RuntimeError("prepare_tasks must run before task execution")
        prompt = (
            "Answer the question only from the supplied article. Return a grounded explanation.\n"
            f"Question: {pair['question']}\nArticle: {pair['source']}"
        )
        record: Dict[str, Any] = {
            "task_id": pair["task_id"],
            "source_id": pair["source_id"],
            "question": pair["question"],
            "source": pair["source"],
            "model": self.model_name,
            "sft_ready": False,
            "status": "unreviewed",
        }
        try:
            async with self.semaphore:
                completion = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
            response = completion.choices[0].message.content or ""
            usage = getattr(completion, "usage", None)
            record["raw_response"] = response
            record["completion_tokens"] = int(getattr(usage, "completion_tokens", 0) or 0)
            await self.monitor.increment_progress()
        except APIConnectionError:
            record.update({"status": "error", "error": "vLLM connection timeout", "raw_response": ""})
            await self.monitor.increment_progress(errors=1)
        except Exception as exc:
            record.update({"status": "error", "error": str(exc), "raw_response": ""})
            await self.monitor.increment_progress(errors=1)
        return record

    def write_results(self, completed_tasks: Iterable[Dict[str, Any]]):
        """Write structured provenance; never write a bare ``list[str]`` as SFT data."""
        records = [record for record in completed_tasks if isinstance(record, dict)]
        json_path = self.output_path / "processed_results.json"
        jsonl_path = self.output_path / "processed_results.jsonl"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[EXECUTOR] Wrote {len(records)} structured legacy records to {json_path}")

    async def run_all_tasks(self) -> List[Dict[str, Any]]:
        if self.monitor is None:
            raise RuntimeError("prepare_tasks must run before run_all_tasks")
        self.start_time = time.time()
        monitor_task = asyncio.create_task(self._monitor_progress())
        completed_tasks: List[Dict[str, Any]] = []
        try:
            completed_tasks = await ta.tqdm_asyncio.gather(*self.tasks)
            return completed_tasks
        finally:
            monitor_task.cancel()
            await asyncio.gather(monitor_task, return_exceptions=True)
            self.write_results(completed_tasks)
