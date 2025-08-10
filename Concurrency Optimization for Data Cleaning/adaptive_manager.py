import asyncio
import time
import json
import yaml
import traceback
from pathlib import Path
from collections import defaultdict

import aiofiles
from openai import AsyncOpenAI

from task_executor import TaskExecutor
from vllm_server import VLLMServer
from data_models import DataPoint_V, DataPoint_T


class DynamicSemaphore:
    """A semaphore that allows for dynamic adjustment of its limit."""

    def __init__(self, initial_limit):
        if initial_limit < 0:
            raise ValueError("Semaphore initial value must be >= 0")
        self._limit = initial_limit
        self._running_tasks = 0
        self._condition = asyncio.Condition()

    async def acquire(self):
        async with self._condition:
            await self._condition.wait_for(lambda: self._running_tasks < self._limit)
            self._running_tasks += 1

    async def release(self):
        async with self._condition:
            self._running_tasks -= 1
            self._condition.notify()

    async def set_limit(self, new_limit):
        """Safely updates the concurrency limit and notifies waiting tasks."""
        async with self._condition:
            print(f"--- Concurrency limit changed from {self._limit} to {new_limit} ---")
            self._limit = new_limit
            self._condition.notify_all()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()


class AdaptiveManager:
    """Orchestrates the VLLM server, task execution, and adaptive concurrency control."""

    def __init__(self, config: dict):
        self.config = config
        self.vllm_config = config['vllm']
        self.manager_config = config['manager']

        self.vllm_api_url = f"http://{self.vllm_config['host']}:{self.vllm_config['port']}/v1"
        self.async_client = AsyncOpenAI(base_url=self.vllm_api_url, api_key="placeholder")
        self.semaphore = DynamicSemaphore(self.manager_config['initial_concurrency'])

        self.record_list: list[dict] = []
        self.start_time: float = 0.0
        self.current_concurrency = self.manager_config['initial_concurrency']

        self.agent_config = config.get('agent', {})
        self.agent_client = None
        if self.agent_config.get('endpoint_url'):
            self.agent_client = AsyncOpenAI(base_url=self.agent_config['endpoint_url'],
                                            api_key=self.agent_config.get('api_key'))

        self.adjustment_timestamp = -100.0
        self.last_processed_record = {}
        self.vllm_worker = None
        self.task_executor = None

    async def _adjustment_policy(self, action: str, now_concurrency: int, avg_kv_cache: float,
                                 last_pending_ratio: float) -> int:
        """Determines the new concurrency limit based on current system state."""
        if action == "turn_up":
            new_concurrency = int((now_concurrency / avg_kv_cache) * 0.85) if avg_kv_cache > 0 else now_concurrency * 2
            return min(new_concurrency, 400)

        elif action == "turn_down":
            new_concurrency = int(now_concurrency / 2)
            return max(new_concurrency, 25)
        else:
            raise NotImplementedError(f"Illegal Action: {action}")

    async def _concurrency_monitor(self):
        """Periodically evaluates system metrics and adjusts concurrency."""
        print("[MANAGER] Concurrency monitor started.")
        while True:
            await asyncio.sleep(5)

            if len(self.record_list) < self.manager_config['monitor_window_size']:
                continue

            latest_record = self.record_list[-1]
            if self.last_processed_record == latest_record:
                continue
            self.last_processed_record = latest_record

            window_size = self.manager_config['monitor_window_size']
            recent_records = self.record_list[-window_size:]

            pending_ratios = [rec["pending"] / self.current_concurrency for rec in recent_records if
                              self.current_concurrency > 0]
            kv_usages = [rec["kv_cache_usage"] for rec in recent_records]

            if any(ratio > 0.3 for ratio in pending_ratios):
                if time.time() - self.adjustment_timestamp > self.manager_config['monitor_decision_interval']:
                    print("[ADJUST] High pending ratio detected. Turning down concurrency.")
                    new_concurrency = await self._adjustment_policy("turn_down", self.current_concurrency,
                                                                    kv_usages[-1], pending_ratios[-1])
                    await self._adjust_concurrency(new_concurrency)
                continue

            avg_kv = sum(kv_usages) / len(kv_usages)
            if avg_kv < 0.8 and kv_usages[-1] < 0.8:
                if time.time() - self.adjustment_timestamp > self.manager_config['monitor_decision_interval']:
                    print(f"[ADJUST] Average KV cache usage ({avg_kv:.2%}) is low. Turning up concurrency.")
                    new_concurrency = await self._adjustment_policy("turn_up", self.current_concurrency, avg_kv,
                                                                    pending_ratios[-1])
                    await self._adjust_concurrency(new_concurrency)
                continue

    async def _adjust_concurrency(self, new_concurrency: int):
        if new_concurrency == self.current_concurrency:
            return
        await self.semaphore.set_limit(new_concurrency)
        self.current_concurrency = new_concurrency
        self.adjustment_timestamp = time.time()
        print(f"<SEMA>: Concurrency adjusted to {new_concurrency}")

    async def _autosave_results(self):
        output_dir = Path("./autosave")
        output_dir.mkdir(exist_ok=True)
        while True:
            await asyncio.sleep(15)
            log_data = {
                "meta": {
                    "model": self.vllm_config['model_path'],
                    "gpus": self.vllm_config['num_gpus'],
                    "workload": self.task_executor.get_total_tasks(),
                    "time_cost": time.time() - self.start_time
                },
                "records": self.record_list
            }
            async with aiofiles.open(output_dir / "adaptive_vllm_log.json", "w", encoding="utf-8") as f:
                await f.write(json.dumps(log_data, ensure_ascii=False, indent=4))
            print("[AUTOSAVE] Progress log saved.")

    async def _merge_metrics(self):
        cache = defaultdict(dict)
        data_queue = self.task_executor.monitor.record_queue
        vllm_queue = self.vllm_worker.monitor.record_queue
        queues = {"data": data_queue, "vllm": vllm_queue}

        async def consume(queue_name):
            while True:
                datapoint = await queues[queue_name].get()
                timestep = datapoint.timestep

                cache[timestep][queue_name] = datapoint

                if "data" in cache[timestep] and "vllm" in cache[timestep]:
                    data_point_t: DataPoint_T = cache[timestep]["data"]
                    data_point_v: DataPoint_V = cache[timestep]["vllm"]

                    merged_data = {
                        "timestep": timestep,
                        "time_shift": data_point_t.timestamp_t - data_point_v.timestamp_v,
                        "concurrency": self.current_concurrency,
                        "completion_rate": data_point_t.completion_rate,
                        "kv_cache_usage": data_point_v.kv_cache_usage,
                        "pending": data_point_v.pending_requests,
                        "avg_input_tokens": data_point_v.avg_input_tokens,
                        "avg_output_tokens": data_point_v.avg_output_tokens,
                    }
                    self.record_list.append(merged_data)
                    print(f"[METRICS MERGED] Timestep: {timestep}", merged_data)
                    del cache[timestep]

                queues[queue_name].task_done()

        await asyncio.gather(consume("data"), consume("vllm"))

    async def run(self):
        self.start_time = time.time()
        try:
            self.vllm_worker = VLLMServer(self.config['vllm'])
            self.task_executor = TaskExecutor(
                sema=self.semaphore,
                async_client=self.async_client,
                config=self.config['task_executor'],
                model_name=self.vllm_config['model_name']
            )

            self.task_executor.prepare_tasks()
            await self.vllm_worker.wait_for_server_ready()

            tasks = [
                asyncio.create_task(self.task_executor.run_all_tasks()),
                asyncio.create_task(self._merge_metrics()),
                asyncio.create_task(self._concurrency_monitor()),
                asyncio.create_task(self._autosave_results())
            ]

            await tasks[0]

            print("[MANAGER] Data processing finished. Allowing 15s for final metrics...")
            await asyncio.sleep(15)

            for task in tasks[1:]:
                task.cancel()
            await asyncio.gather(*tasks[1:], return_exceptions=True)

        except Exception as e:
            print(f"[MANAGER] An error occurred: {e}")
            traceback.print_exc()
        finally:
            log_path = Path("./final_log")
            log_path.mkdir(exist_ok=True)
            with open(log_path / "adaptive_vllm_final_log.json", "w", encoding="utf-8") as f:
                log = {
                    "meta": {"model": self.vllm_config['model_path'], "gpus": self.vllm_config['num_gpus'],
                             "total_tasks": self.task_executor.get_total_tasks(),
                             "time_cost": time.time() - self.start_time},
                    "records": self.record_list
                }
                json.dump(log, f, ensure_ascii=False, indent=4)
            print("[MANAGER] Final log file written.")

            if self.vllm_worker:
                await self.vllm_worker.shutdown()
            print("[MANAGER] Program finished.")

    def start(self):
        try:
            asyncio.run(asyncio.wait_for(self.run(), self.manager_config['max_waiting_time']))
        except asyncio.TimeoutError:
            print("[MANAGER] Max waiting time exceeded. Forcing shutdown.")
            if self.vllm_worker:
                self.vllm_worker.force_shutdown()
            # sys.exit(1) # Consider exiting with an error code


if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    manager = AdaptiveManager(config)
    manager.start()