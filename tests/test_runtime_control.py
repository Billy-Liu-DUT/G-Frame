import asyncio
import unittest

from g_frame.actions import ActionExecutor, RuntimeState
from g_frame.clients import LimiterBoundChatClient, ScriptedChatClient
from g_frame.decision import DecisionAgent
from g_frame.orchestration import AdaptiveRuntimeController
from g_frame.prompts import PromptCatalog
from g_frame.schemas import AgentMessage, EngineTelemetry, TaskTelemetry
from g_frame.telemetry import (
    DynamicSemaphore,
    TaskTelemetryProducer,
    TaskTelemetryTracker,
    TelemetryQueueMerger,
    VLLMMetricsProducer,
    metrics_url_from_openai_base_url,
)


class _BlockingClient:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, messages, model, role):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            await self.release.wait()
        finally:
            self.active -= 1
        return "{}"


async def _wait_until(predicate, attempts: int = 200) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


class RuntimeControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_limiter_bound_client_enforces_and_adapts_live_concurrency(self):
        raw_client = _BlockingClient()
        limiter = DynamicSemaphore(1)
        client = LimiterBoundChatClient(raw_client, limiter)
        messages = [AgentMessage("user", "test")]
        requests = [
            asyncio.create_task(client.complete(messages, model="scripted", role=f"worker-{index}"))
            for index in range(3)
        ]

        await raw_client.started.wait()
        await asyncio.sleep(0)
        self.assertEqual(raw_client.active, 1)
        self.assertEqual(limiter.running_tasks, 1)

        await limiter.set_limit(2)
        await _wait_until(lambda: raw_client.active == 2)
        self.assertEqual(limiter.limit, 2)
        self.assertLessEqual(raw_client.max_active, 2)

        raw_client.release.set()
        await asyncio.gather(*requests)
        self.assertEqual(raw_client.calls, 3)
        self.assertEqual(raw_client.max_active, 2)
        self.assertEqual(limiter.running_tasks, 0)

    async def test_vllm_metrics_producer_uses_derived_endpoint_without_sdk_dependencies(self):
        observed_urls = []

        async def fetcher(url):
            observed_urls.append(url)
            return "\n".join(
                (
                    "# HELP vllm:gpu_cache_usage_perc KV cache utilization",
                    'vllm:gpu_cache_usage_perc{model_name="smoke"} 0.83',
                    "vllm:num_requests_waiting 4",
                    "vllm:num_requests_running 2",
                    "vllm:request_latency_seconds_sum 6.0",
                    "vllm:request_latency_seconds_count 3",
                )
            )

        queue: asyncio.Queue = asyncio.Queue()
        producer = VLLMMetricsProducer(
            queue,
            base_url="http://127.0.0.1:8002/v1",
            fetcher=fetcher,
        )
        telemetry = await producer.publish(7)

        self.assertEqual(observed_urls, ["http://127.0.0.1:8002/metrics"])
        self.assertEqual(telemetry.timestep, 7)
        self.assertEqual(telemetry.kv_cache_usage, 0.83)
        self.assertEqual(telemetry.pending_requests, 4)
        self.assertEqual(telemetry.active_requests, 2)
        self.assertEqual(telemetry.average_latency_s, 2.0)
        self.assertEqual(await queue.get(), telemetry)
        self.assertEqual(metrics_url_from_openai_base_url("http://host:9/v1/"), "http://host:9/metrics")

    async def test_task_tracker_is_an_injectable_task_telemetry_source(self):
        tracker = TaskTelemetryTracker()
        await tracker.enqueue(2)
        await tracker.start()
        await tracker.complete()
        queue: asyncio.Queue = asyncio.Queue()
        producer = TaskTelemetryProducer(queue, tracker.snapshot)

        telemetry = await producer.publish(5)

        self.assertEqual(telemetry.timestep, 5)
        self.assertEqual(telemetry.queued_tasks, 1)
        self.assertEqual(telemetry.completed_tasks, 1)
        self.assertEqual(telemetry.failed_tasks, 0)
        self.assertEqual(telemetry.completion_rate, 0.5)
        self.assertEqual(await queue.get(), telemetry)

    async def test_queue_merger_preserves_out_of_order_matching_timesteps(self):
        engine_queue: asyncio.Queue = asyncio.Queue()
        task_queue: asyncio.Queue = asyncio.Queue()
        merger = TelemetryQueueMerger(engine_queue, task_queue)
        await engine_queue.put(EngineTelemetry(2, 0.4, pending_requests=1))
        await task_queue.put(TaskTelemetry(1, 0.5, queued_tasks=1, completed_tasks=1))
        await engine_queue.put(EngineTelemetry(1, 0.3, pending_requests=0))
        await task_queue.put(TaskTelemetry(2, 0.5, queued_tasks=1, completed_tasks=1))
        try:
            states = [await merger.next_state(timeout_s=0.5), await merger.next_state(timeout_s=0.5)]
        finally:
            await merger.aclose()

        self.assertEqual({state.timestep for state in states}, {1, 2})

    async def test_persistent_controller_applies_two_decision_plans(self):
        engine_snapshots = iter(
            (
                {"kv_cache_usage": 0.8, "pending_requests": 3, "active_requests": 1},
                {"kv_cache_usage": 0.2, "pending_requests": 0, "active_requests": 2},
            )
        )
        task_snapshots = iter(
            (
                {"queued_tasks": 2, "completed_tasks": 1, "total_tasks": 3},
                {"queued_tasks": 0, "completed_tasks": 3, "total_tasks": 3},
            )
        )
        engine_queue: asyncio.Queue = asyncio.Queue()
        task_queue: asyncio.Queue = asyncio.Queue()
        limiter = DynamicSemaphore(1)
        executor = ActionExecutor(RuntimeState(concurrency=1, max_concurrency=2))
        executor.bind_concurrency_handler(limiter.set_limit)
        executed_stages = []

        async def stage_handler(action):
            executed_stages.append(action.action)

        executor.bind_stage_handler("qa_synth", stage_handler)
        client = ScriptedChatClient(
            {
                "decisional": [
                    '{"rationale":"queue pressure", "plan_of_action":[{"action":"set_concurrency", "arguments":{"value":2}}, {"action":"qa_synth", "arguments":{}}]}',
                    '{"rationale":"queue cleared", "plan_of_action":[{"action":"set_concurrency", "arguments":{"value":1}}, {"action":"qa_synth", "arguments":{}}]}',
                ]
            }
        )
        controller = AdaptiveRuntimeController(
            engine_producer=VLLMMetricsProducer(engine_queue, snapshot_provider=lambda: next(engine_snapshots)),
            task_producer=TaskTelemetryProducer(task_queue, lambda: next(task_snapshots)),
            telemetry_merger=TelemetryQueueMerger(engine_queue, task_queue),
            decision_agent=DecisionAgent(client, PromptCatalog(), executor, model="scripted"),
            action_executor=executor,
        )
        try:
            results = await controller.run(max_cycles=2, start_timestep=10)
        finally:
            await controller.aclose()

        self.assertEqual([result.state.timestep for result in results], [10, 11])
        self.assertEqual(executed_stages, ["qa_synth", "qa_synth"])
        self.assertEqual(executor.state.concurrency, 1)
        self.assertEqual(limiter.limit, 1)
        self.assertEqual(len(client.requests), 2)
