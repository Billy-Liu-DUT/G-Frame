"""CPU-safe command line entry points for G-Frame v2."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional, Sequence

from .actions import ActionExecutor, RuntimeState
from .clients import LimiterBoundChatClient, ScriptedChatClient
from .data import SFTDatasetBuilder, write_training_records
from .decision import DecisionAgent
from .orchestration import GFrameOrchestrator
from .prompts import PromptCatalog
from .schemas import EngineTelemetry, TaskTelemetry
from .sft import SFTJobConfig
from .team_game import TeamGame
from .telemetry import DynamicSemaphore


async def _offline_smoke(output_dir: Path) -> Path:
    prompts = PromptCatalog()
    client = ScriptedChatClient(
        {
            "teacher": '{"question": "Why does increasing conjugation red-shift absorption?"}',
            "student": '{"reasoning": "A longer conjugated path reduces the HOMO-LUMO gap.", "answer": "Increasing conjugation usually lowers the electronic transition energy and red-shifts absorption."}',
            "reviewer": [
                '{"approved": false, "feedback": "State that this is a qualitative trend and depends on molecular structure."}',
                '{"approved": true, "feedback": "The qualification is now present."}',
            ],
            "rectifier": '{"reasoning": "A longer conjugated path often reduces the HOMO-LUMO gap, although substituents and geometry matter.", "answer": "Increasing conjugation often lowers transition energy and red-shifts absorption; the magnitude depends on structure and geometry."}',
            "judger": '{"final_answer": "Increasing conjugation often lowers transition energy and red-shifts absorption, subject to structural and geometric effects.", "approved": true, "feedback": "Final answer remains grounded."}',
            "decisional": '{"rationale": "Low KV-cache use allows one additional synthesis worker; the reviewed record should proceed to SFT preparation.", "plan_of_action": [{"action": "set_concurrency", "arguments": {"value": 2}}, {"action": "qa_synth", "arguments": {}}]}',
        }
    )
    executor = ActionExecutor(RuntimeState(concurrency=1, learning_rate=2e-5))
    worker_limiter = DynamicSemaphore(initial_limit=1)
    stage_queue: asyncio.Queue = asyncio.Queue()
    executor.bind_concurrency_handler(worker_limiter.set_limit)
    executor.bind_stage_handler("qa_synth", stage_queue.put)
    team_game = TeamGame(
        client=LimiterBoundChatClient(client, worker_limiter),
        prompts=prompts,
        model="offline-scripted",
    )
    decision_agent = DecisionAgent(client=client, prompts=prompts, executor=executor, model="offline-scripted")
    orchestrator = GFrameOrchestrator(team_game, decision_agent, executor)
    result = await orchestrator.run_cycle(
        task_id="offline-smoke-001",
        source_id="public-fixture-001",
        source="Conjugated organic chromophores commonly show lower transition energies as conjugation increases.",
        engine=EngineTelemetry(timestep=1, kv_cache_usage=0.42, pending_requests=0, active_requests=1),
        tasks=TaskTelemetry(timestep=1, completion_rate=1.0, queued_tasks=0, completed_tasks=1),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_training_records([result.record], output_dir / "training_records.jsonl")
    SFTDatasetBuilder.write_jsonl([result.record], output_dir / "sft_messages.jsonl")
    manifest = result.to_dict()
    manifest["runtime_state"] = executor.state.to_dict()
    manifest["worker_limiter"] = {"limit": worker_limiter.limit, "running_tasks": worker_limiter.running_tasks}
    manifest["queued_stages"] = [stage_queue.get_nowait().to_dict()] if not stage_queue.empty() else []
    manifest["workflow"] = ["team_game", "telemetry_merge", "decision", "action_execution", "sft_serialization"]
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return output_dir


def _cmd_offline_smoke(args: argparse.Namespace) -> int:
    output_dir = asyncio.run(_offline_smoke(Path(args.output_dir)))
    print(f"offline smoke completed: {output_dir}")
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    if not args.offline:
        raise SystemExit("only --offline is available until a GPU run is explicitly authorized")
    return _cmd_offline_smoke(args)


def _cmd_validate_sft(args: argparse.Namespace) -> int:
    config = SFTJobConfig.from_json_file(Path(args.config))
    print(json.dumps(config.to_dict(), indent=2))
    print("DeepSpeed command template:", " ".join(config.deepspeed_command(str(Path(args.config)), gpu_count=args.gpu_count)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gframe", description="G-Frame v2 workflow utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    offline_smoke = subparsers.add_parser("offline-smoke", help="run the CPU-only Team Game and decision-loop smoke test")
    offline_smoke.add_argument("--output-dir", default="runs/offline-smoke")
    offline_smoke.set_defaults(func=_cmd_offline_smoke)
    smoke = subparsers.add_parser("smoke", help="run a smoke test")
    smoke.add_argument("--offline", action="store_true", help="run only the deterministic CPU-only closed loop")
    smoke.add_argument("--output-dir", default="runs/offline-smoke")
    smoke.set_defaults(func=_cmd_smoke)
    validate_sft = subparsers.add_parser(
        "validate-sft",
        help="inspect a full-parameter SFT command without starting training",
    )
    validate_sft.add_argument("--config", required=True)
    validate_sft.add_argument("--gpu-count", type=int, default=1)
    validate_sft.set_defaults(func=_cmd_validate_sft)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
