"""v1-compatible adaptive manager backed by the typed v2 decision loop."""

import asyncio
import json
from pathlib import Path
import sys
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional

import yaml
from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from g_frame.actions import ActionExecutor, RuntimeState
from g_frame.clients import OpenAICompatibleClient
from g_frame.decision import DecisionAgent
from g_frame.prompts import PromptCatalog
from g_frame.schemas import EngineTelemetry, SystemState, TaskTelemetry
from g_frame.telemetry import DynamicSemaphore

from data_models import DataPoint_T, DataPoint_V
from task_executor import TaskExecutor
from vllm_server import VLLMServer


class AdaptiveManager:
    """Coordinate vLLM, v1 queues, and the paper-style decision agent.

    The old hard-coded threshold policy is intentionally gone. When the
    decision agent is disabled, the manager keeps its configured fixed
    concurrency and says so explicitly rather than claiming adaptivity.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vllm_config = config["vllm"]
        self.manager_config = config["manager"]
        self.vllm_api_url = f"http://{self.vllm_config['host']}:{self.vllm_config['port']}/v1"
        self.async_client = AsyncOpenAI(base_url=self.vllm_api_url, api_key="placeholder")
        self.semaphore = DynamicSemaphore(int(self.manager_config["initial_concurrency"]))
        self.current_concurrency = self.semaphore.limit
        self.record_list: List[Dict[str, Any]] = []
        self.start_time = 0.0
        self.adjustment_timestamp = -float("inf")
        self.last_processed_record: Optional[Dict[str, Any]] = None
        self.vllm_worker: Optional[VLLMServer] = None
        self.task_executor: Optional[TaskExecutor] = None
        self.decision_history = []
        self._fixed_concurrency_notice_printed = False

        self.decision_executor = ActionExecutor(RuntimeState(concurrency=self.current_concurrency))
        self.decision_executor.bind_concurrency_handler(self.semaphore.set_limit)
        self.agent_config = config.get("agent", {})
        self.agent_client = None
        self.decision_agent: Optional[DecisionAgent] = None
        self._configure_decision_agent()

    def _configure_decision_agent(self) -> None:
        enabled = bool(self.agent_config.get("enabled", False))
        endpoint = str(self.agent_config.get("endpoint_url", "")).strip()
        api_key = str(self.agent_config.get("api_key", "")).strip()
        model_name = str(self.agent_config.get("model_name", "")).strip()
        placeholders = ("your.", "YOUR_", "placeholder")
        valid = endpoint and api_key and model_name and not any(token in endpoint or token in api_key or token in model_name for token in placeholders)
        if enabled and not valid:
            raise ValueError("agent.enabled requires non-placeholder endpoint_url, api_key, and model_name")
        if not enabled:
            return
        self.agent_client = OpenAICompatibleClient(endpoint, api_key)
        self.decision_agent = DecisionAgent(
            client=self.agent_client,
            prompts=PromptCatalog(),
            executor=self.decision_executor,
            model=model_name,
        )

    @staticmethod
    def _write_json(path: Path, value: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)

    async def _concurrency_monitor(self) -> None:
        """Feed merged telemetry to the decision LLM at configured intervals."""
        print("[MANAGER] Decision monitor started.")
        while True:
            await asyncio.sleep(5)
            if len(self.record_list) < int(self.manager_config["monitor_window_size"]):
                continue
            latest_record = self.record_list[-1]
            if self.last_processed_record is latest_record:
                continue
            self.last_processed_record = latest_record
            if self.decision_agent is None:
                if not self._fixed_concurrency_notice_printed:
                    print("[MANAGER] Decision agent disabled; retaining configured fixed concurrency.")
                    self._fixed_concurrency_notice_printed = True
                continue
            if time.time() - self.adjustment_timestamp < float(self.manager_config["monitor_decision_interval"]):
                continue
            try:
                await self._decide_and_apply(latest_record)
            except Exception as exc:
                print(f"[MANAGER] Decision agent failed; concurrency unchanged: {exc}")

    async def _decide_and_apply(self, latest_record: Dict[str, Any]) -> None:
        if self.task_executor is None or self.task_executor.monitor is None or self.decision_agent is None:
            raise RuntimeError("manager is not ready for a decision")
        window_size = int(self.manager_config["monitor_window_size"])
        recent = self.record_list[-window_size:]
        timestep = int(latest_record["timestep"])
        engine = EngineTelemetry(
            timestep=timestep,
            kv_cache_usage=sum(float(record["kv_cache_usage"]) for record in recent) / len(recent),
            pending_requests=max(int(record["pending"]) for record in recent),
            active_requests=int(latest_record.get("active_requests", 0)),
            average_latency_s=float(latest_record.get("average_latency_s", 0.0)),
            error_count=int(latest_record.get("failed_tasks", 0)),
        )
        monitor = self.task_executor.monitor
        completed = monitor.completed_tasks
        total = monitor.total_tasks
        task = TaskTelemetry(
            timestep=timestep,
            completion_rate=await monitor.get_completion_rate(),
            queued_tasks=max(total - completed, 0),
            completed_tasks=completed,
            failed_tasks=monitor.error_count,
        )
        state = SystemState(timestep=timestep, engine=engine, tasks=task)
        decision = await self.decision_agent.decide(state, self.decision_history)
        executions = await self.decision_executor.execute_plan_async(decision.plan)
        self.decision_history.append(decision)
        self.current_concurrency = self.semaphore.limit
        self.adjustment_timestamp = time.time()
        latest_record["decision"] = decision.plan.to_dict()
        latest_record["executions"] = [execution.to_dict() for execution in executions]
        print(f"[MANAGER] Applied decision at timestep {timestep}; concurrency={self.current_concurrency}")

    async def _autosave_results(self) -> None:
        output_path = Path("autosave") / "adaptive_vllm_log.json"
        while True:
            await asyncio.sleep(15)
            workload = self.task_executor.get_total_tasks() if self.task_executor else 0
            await asyncio.to_thread(
                self._write_json,
                output_path,
                {
                    "meta": {
                        "model": self.vllm_config["model_path"],
                        "gpus": self.vllm_config["num_gpus"],
                        "workload": workload,
                        "time_cost": time.time() - self.start_time,
                    },
                    "records": self.record_list,
                },
            )
            print("[AUTOSAVE] Progress log saved.")

    async def _merge_metrics(self) -> None:
        if self.task_executor is None or self.vllm_worker is None:
            raise RuntimeError("workers must be initialized before telemetry merge")
        cache: Dict[int, Dict[str, Any]] = defaultdict(dict)
        queues = {
            "data": self.task_executor.monitor.record_queue,
            "vllm": self.vllm_worker.monitor.record_queue,
        }

        async def consume(queue_name: str) -> None:
            while True:
                datapoint = await queues[queue_name].get()
                timestep = int(datapoint.timestep)
                cache[timestep][queue_name] = datapoint
                if "data" in cache[timestep] and "vllm" in cache[timestep]:
                    data_point_t: DataPoint_T = cache[timestep]["data"]
                    data_point_v: DataPoint_V = cache[timestep]["vllm"]
                    monitor = self.task_executor.monitor
                    self.record_list.append(
                        {
                            "timestep": timestep,
                            "time_shift": data_point_t.timestamp_t - data_point_v.timestamp_v,
                            "concurrency": self.current_concurrency,
                            "completion_rate": data_point_t.completion_rate,
                            "kv_cache_usage": data_point_v.kv_cache_usage,
                            "pending": data_point_v.pending_requests,
                            "active_requests": data_point_v.running_requests,
                            "failed_tasks": monitor.error_count,
                            "avg_input_tokens": data_point_v.avg_input_tokens,
                            "avg_output_tokens": data_point_v.avg_output_tokens,
                        }
                    )
                    del cache[timestep]
                queues[queue_name].task_done()

        await asyncio.gather(consume("data"), consume("vllm"))

    async def run(self) -> None:
        self.start_time = time.time()
        try:
            self.vllm_worker = VLLMServer(self.vllm_config)
            await self.vllm_worker.wait_for_server_ready()
            self.task_executor = TaskExecutor(
                sema=self.semaphore,
                async_client=self.async_client,
                config=self.config["task_executor"],
                model_name=self.vllm_config["model_name"],
            )
            self.task_executor.prepare_tasks()
            tasks = [
                asyncio.create_task(self.task_executor.run_all_tasks()),
                asyncio.create_task(self._merge_metrics()),
                asyncio.create_task(self._concurrency_monitor()),
                asyncio.create_task(self._autosave_results()),
            ]
            await tasks[0]
            await asyncio.sleep(float(self.manager_config.get("final_metrics_grace_s", 5)))
            for task in tasks[1:]:
                task.cancel()
            await asyncio.gather(*tasks[1:], return_exceptions=True)
        except Exception as exc:
            print(f"[MANAGER] An error occurred: {exc}")
            traceback.print_exc()
            raise
        finally:
            workload = self.task_executor.get_total_tasks() if self.task_executor else 0
            await asyncio.to_thread(
                self._write_json,
                Path("final_log") / "adaptive_vllm_final_log.json",
                {
                    "meta": {
                        "model": self.vllm_config["model_path"],
                        "gpus": self.vllm_config["num_gpus"],
                        "total_tasks": workload,
                        "time_cost": time.time() - self.start_time,
                    },
                    "records": self.record_list,
                },
            )
            if self.vllm_worker:
                await self.vllm_worker.shutdown()
            print("[MANAGER] Program finished.")

    def start(self) -> None:
        try:
            asyncio.run(asyncio.wait_for(self.run(), float(self.manager_config["max_waiting_time"])))
        except asyncio.TimeoutError:
            print("[MANAGER] Maximum waiting time exceeded.")
            if self.vllm_worker:
                self.vllm_worker.force_shutdown()


if __name__ == "__main__":
    config_path = Path(__file__).with_name("config.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        manager = AdaptiveManager(yaml.safe_load(handle))
    manager.start()
