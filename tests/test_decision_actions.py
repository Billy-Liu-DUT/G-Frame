import unittest

from g_frame.actions import ActionExecutor, RuntimeState
from g_frame.clients import ScriptedChatClient
from g_frame.decision import DecisionAgent
from g_frame.prompts import PromptCatalog
from g_frame.schemas import EngineTelemetry, PlanAction, SchemaError, SystemState, TaskTelemetry
from g_frame.telemetry import DynamicSemaphore, TelemetryMerger


class DecisionActionTests(unittest.IsolatedAsyncioTestCase):
    async def test_decision_response_is_validated_then_dispatched(self):
        executor = ActionExecutor(RuntimeState(concurrency=16))
        client = ScriptedChatClient(
            {
                "decisional": '{"rationale": "High pending pressure needs throttling.", "plan_of_action": [{"action": "set_concurrency", "arguments": {"value": 8}}, {"action": "qa_synth", "arguments": {}}]}',
            }
        )
        agent = DecisionAgent(client, PromptCatalog(), executor, model="scripted")
        state = SystemState(
            timestep=3,
            engine=EngineTelemetry(3, 0.92, pending_requests=10, active_requests=16),
            tasks=TaskTelemetry(3, 0.5, queued_tasks=10, completed_tasks=10),
        )
        artifact = await agent.decide(state)
        executions = executor.execute_plan(artifact.plan)
        self.assertEqual(executor.state.concurrency, 8)
        self.assertEqual(executions[0].status, "applied")
        self.assertIn("qa_synth", executor.state.scheduled_stages)
        self.assertIn("pending_requests", client.requests[0]["messages"][1]["content"])

    async def test_decision_retries_once_after_an_invalid_plan(self):
        executor = ActionExecutor()
        client = ScriptedChatClient(
            {
                "decisional": [
                    "not JSON",
                    '{"rationale":"keep one worker", "plan_of_action":[{"action":"set_concurrency", "arguments":{"value":1}}]}',
                ]
            }
        )
        agent = DecisionAgent(client, PromptCatalog(), executor, model="scripted", max_attempts=2)
        state = SystemState(
            timestep=4,
            engine=EngineTelemetry(4, 0.5, pending_requests=0),
            tasks=TaskTelemetry(4, 1.0, queued_tasks=0, completed_tasks=1),
        )
        artifact = await agent.decide(state)
        self.assertEqual(artifact.plan.actions[0].arguments["value"], 1)
        self.assertEqual(len(client.requests), 2)

    async def test_decision_retries_when_concurrency_exceeds_the_runtime_cap(self):
        executor = ActionExecutor(RuntimeState(concurrency=1, max_concurrency=8))
        client = ScriptedChatClient(
            {
                "decisional": [
                    '{"rationality":"increase throughput", "plan_of_action":[{"action":"set_concurrency", "arguments":{"value":300}}]}',
                    '{"explanation":"stay within the smoke limit", "plan_of_action":[{"action":"set_concurrency", "arguments":{"value":8}}]}',
                ]
            }
        )
        agent = DecisionAgent(client, PromptCatalog(), executor, model="scripted", max_attempts=2)
        state = SystemState(
            timestep=5,
            engine=EngineTelemetry(5, 0.5, pending_requests=0),
            tasks=TaskTelemetry(5, 1.0, queued_tasks=0, completed_tasks=1),
        )
        artifact = await agent.decide(state)
        self.assertEqual(artifact.plan.rationale, "stay within the smoke limit")
        self.assertEqual(artifact.plan.actions[0].arguments["value"], 8)
        self.assertEqual(len(client.requests), 2)
        self.assertIn("integer from 1 to 8", client.requests[1]["messages"][-1]["content"])

    async def test_smoke_prompt_restricts_the_plan_to_the_live_smoke_contract(self):
        executor = ActionExecutor(RuntimeState(concurrency=1, max_concurrency=8))
        client = ScriptedChatClient(
            {
                "decisional": (
                    '{"rationale":"one worker is safe", "plan_of_action":['
                    '{"action":"set_concurrency","arguments":{"value":1}},'
                    '{"action":"qa_synth","arguments":{}}]}'
                )
            }
        )
        agent = DecisionAgent(
            client,
            PromptCatalog(),
            executor,
            model="scripted",
            prompt_name="smoke",
        )
        state = SystemState(
            timestep=6,
            engine=EngineTelemetry(6, 0.5, pending_requests=0),
            tasks=TaskTelemetry(6, 1.0, queued_tasks=0, completed_tasks=1),
        )
        await agent.decide(state)
        prompt = client.requests[0]["messages"][1]["content"]
        self.assertIn("exactly two actions", prompt)
        self.assertIn("set_learning_rate", prompt)

    async def test_unknown_action_is_rejected_before_execution(self):
        executor = ActionExecutor()
        with self.assertRaisesRegex(SchemaError, "unsupported"):
            executor.execute(PlanAction("shell", {"command": "rm -rf /"}))

    def test_telemetry_merges_only_matching_timesteps(self):
        merger = TelemetryMerger()
        self.assertIsNone(merger.ingest_engine(EngineTelemetry(1, 0.5, 0)))
        self.assertIsNone(merger.ingest_task(TaskTelemetry(2, 0.0, 1, 0)))
        state = merger.ingest_task(TaskTelemetry(1, 1.0, 0, 1))
        self.assertIsNotNone(state)
        self.assertEqual(state.timestep, 1)

    async def test_live_concurrency_handler_is_updated_before_action_is_reported(self):
        limiter = DynamicSemaphore(16)
        executor = ActionExecutor(RuntimeState(concurrency=16))
        executor.bind_concurrency_handler(limiter.set_limit)
        result = await executor.execute_async(PlanAction("set_concurrency", {"value": 8}))
        self.assertEqual(result.status, "applied")
        self.assertEqual(limiter.limit, 8)
        self.assertEqual(executor.state.concurrency, 8)

    def test_set_concurrency_uses_the_configured_runtime_ceiling(self):
        executor = ActionExecutor(RuntimeState(concurrency=1, max_concurrency=8))
        execution = executor.execute(PlanAction("set_concurrency", {"value": 8}))
        self.assertEqual(execution.status, "applied")
        self.assertEqual(executor.state.concurrency, 8)
        self.assertEqual(executor.state.to_dict()["max_concurrency"], 8)
        with self.assertRaisesRegex(SchemaError, "integer from 1 to 8"):
            executor.execute(PlanAction("set_concurrency", {"value": 9}))

    async def test_registered_stage_handler_executes_without_shell_access(self):
        executed = []

        async def handler(action):
            executed.append(action.action)

        executor = ActionExecutor()
        executor.bind_stage_handler("qa_synth", handler)
        result = await executor.execute_async(PlanAction("qa_synth", {"idempotency_key": "batch-1"}))
        self.assertEqual(result.status, "executed")
        self.assertEqual(executed, ["qa_synth"])
        duplicate = await executor.execute_async(PlanAction("qa_synth", {"idempotency_key": "batch-1"}))
        self.assertEqual(duplicate.status, "unchanged")

    def test_plan_supports_paper_casing_and_markdown_json_fence(self):
        from g_frame.schemas import DecisionPlan

        plan = DecisionPlan.from_json(
            '```json\n{"reasoning":"reduce queue pressure", "Plan_of_Action":[{"action":"set_concurrency", "arguments":{"value":8}}]}\n```'
        )
        self.assertEqual(plan.actions[0].arguments["value"], 8)

    def test_plan_supports_common_rationale_aliases(self):
        from g_frame.schemas import DecisionPlan

        for field in ("rationality", "explanation"):
            plan = DecisionPlan.from_json(
                f'{{"{field}":"compatible rationale", "plan_of_action":[{{"action":"set_concurrency", "arguments":{{"value":1}}}}]}}'
            )
            self.assertEqual(plan.rationale, "compatible rationale")

    def test_plan_retains_an_explicit_marker_when_model_omits_rationale(self):
        from g_frame.schemas import DecisionPlan

        plan = DecisionPlan.from_json(
            '{"plan_of_action":[{"action":"set_concurrency", "arguments":{"value":1}}]}'
        )
        self.assertIn("omitted rationale", plan.rationale)
