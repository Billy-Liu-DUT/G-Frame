import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from g_frame.clients import LimiterBoundChatClient, ScriptedChatClient
from scripts import live_gpu_smoke
from scripts.live_gpu_smoke import build_parser


class _ScriptedLiveClient(ScriptedChatClient):
    def __init__(self, **kwargs):
        super().__init__(
            {
                "teacher": '{"question": "What absorption trend follows from the source?"}',
                "student": '{"reasoning": "The source states a red shift.", "answer": "Longer conjugation often red-shifts absorption."}',
                "reviewer": '{"approved": true, "feedback": "Grounded in the source."}',
                "judger": '{"final_answer": "Longer conjugation often red-shifts absorption.", "approved": true, "feedback": "Approved."}',
                "decisional": '{"rationale": "Use two workers for the smoke.", "plan_of_action":[{"action":"set_concurrency","arguments":{"value":2}},{"action":"qa_synth","arguments":{}}]}',
            }
        )


class _RecordingLimiterClient(LimiterBoundChatClient):
    created = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.created.append(self)


class _DelayedScriptedLiveClient(_ScriptedLiveClient):
    created = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active_team_requests = 0
        self.max_team_requests = 0
        self.__class__.created.append(self)

    async def complete(self, messages, model, role):
        if role == "decisional":
            return await super().complete(messages, model, role)
        self.active_team_requests += 1
        self.max_team_requests = max(self.max_team_requests, self.active_team_requests)
        try:
            await asyncio.sleep(0.03)
            return await super().complete(messages, model, role)
        finally:
            self.active_team_requests -= 1


class LiveGPUSmokeTests(unittest.TestCase):
    def test_live_smoke_defaults_to_a_small_concurrency_cap(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.max_concurrency, 8)
        self.assertEqual(args.max_work_items, 2)

    def test_live_smoke_accepts_an_explicit_concurrency_cap(self):
        args = build_parser().parse_args(["--max-concurrency", "4"])
        self.assertEqual(args.max_concurrency, 4)

    def test_live_smoke_accepts_a_separate_decision_model(self):
        args = build_parser().parse_args(["--model", "team-game", "--decision-model", "decider"])
        self.assertEqual(args.model, "team-game")
        self.assertEqual(args.decision_model, "decider")

    def test_live_smoke_accepts_a_separate_decision_endpoint(self):
        args = build_parser().parse_args(
            ["--base-url", "http://team/v1", "--decision-base-url", "http://decision/v1"]
        )
        self.assertEqual(args.base_url, "http://team/v1")
        self.assertEqual(args.decision_base_url, "http://decision/v1")

    def test_fixture_live_smoke_uses_the_limiter_bound_team_client_and_persistent_loop(self):
        _RecordingLimiterClient.created = []
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = build_parser().parse_args(
                [
                    "--telemetry-source",
                    "fixture",
                    "--telemetry-cycles",
                    "2",
                    "--max-work-items",
                    "3",
                    "--output-dir",
                    str(Path(temporary_directory) / "smoke"),
                ]
            )
            with patch.object(live_gpu_smoke, "OpenAICompatibleClient", _ScriptedLiveClient), patch.object(
                live_gpu_smoke, "LimiterBoundChatClient", _RecordingLimiterClient
            ):
                output_dir = asyncio.run(live_gpu_smoke.run_live_smoke(args))

            manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(len(_RecordingLimiterClient.created), 1)
        self.assertEqual(_RecordingLimiterClient.created[0].limiter.limit, 2)
        self.assertEqual(manifest["telemetry"]["source"], "fixture")
        self.assertEqual(len(manifest["control_cycles"]), 2)
        self.assertEqual(manifest["worker_limiter"]["limit"], 2)
        self.assertEqual(manifest["workload"]["submitted_work_items"], 3)
        self.assertEqual(manifest["workload"]["completed_work_items"], 3)
        self.assertEqual(manifest["workload"]["approved_records"], 3)
        self.assertEqual([item["status"] for item in manifest["submissions"]], ["submitted"] * 3)
        self.assertEqual(
            [item["action"] for item in manifest["submissions"]],
            ["initial_work_item", "qa_synth", "qa_synth"],
        )
        self.assertEqual(len(manifest["records"]), 3)

    def test_controller_increases_the_live_team_game_request_limit_before_submitting_work(self):
        _DelayedScriptedLiveClient.created = []
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = build_parser().parse_args(
                [
                    "--telemetry-source",
                    "fixture",
                    "--telemetry-cycles",
                    "1",
                    "--initial-concurrency",
                    "1",
                    "--max-concurrency",
                    "2",
                    "--initial-work-items",
                    "1",
                    "--max-work-items",
                    "2",
                    "--output-dir",
                    str(Path(temporary_directory) / "smoke"),
                ]
            )
            with patch.object(live_gpu_smoke, "OpenAICompatibleClient", _DelayedScriptedLiveClient):
                output_dir = asyncio.run(live_gpu_smoke.run_live_smoke(args))

            manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(len(_DelayedScriptedLiveClient.created), 1)
        self.assertEqual(_DelayedScriptedLiveClient.created[0].max_team_requests, 2)
        self.assertEqual(manifest["worker_limiter"]["limit"], 2)
        self.assertEqual(manifest["workload"]["submitted_work_items"], 2)
        self.assertEqual(manifest["failures"], [])
        self.assertEqual(
            [execution["status"] for execution in manifest["control_cycles"][0]["executions"]],
            ["applied", "executed"],
        )
