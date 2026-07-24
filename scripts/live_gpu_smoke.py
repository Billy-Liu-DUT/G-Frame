"""Run the live vLLM-backed G-Frame v2 loop after the server is started.

This command does not launch vLLM or training itself. It calls a caller-owned
OpenAI-compatible endpoint and writes only public smoke artifacts under the
specified output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, List, Optional, Sequence

from g_frame.actions import ActionExecutor, RuntimeState
from g_frame.clients import LimiterBoundChatClient, OpenAICompatibleClient
from g_frame.data import SFTDatasetBuilder, write_training_records
from g_frame.decision import DecisionAgent
from g_frame.orchestration import AdaptiveRuntimeController
from g_frame.prompts import PromptCatalog
from g_frame.schemas import PlanAction, TrainingRecord
from g_frame.team_game import TeamGame
from g_frame.telemetry import (
    DynamicSemaphore,
    TaskTelemetryProducer,
    TaskTelemetryTracker,
    TelemetryQueueMerger,
    VLLMMetricsProducer,
)


PUBLIC_SMOKE_SOURCE = (
    "Increasing effective conjugation length often lowers electronic transition energy and red-shifts absorption."
)


async def run_live_smoke(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    decision_model = args.decision_model or args.model
    decision_base_url = args.decision_base_url or args.base_url
    worker_limiter = DynamicSemaphore(args.initial_concurrency)
    raw_team_client = OpenAICompatibleClient(
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0.0,
        json_mode=True,
        max_tokens=args.max_tokens,
    )
    team_client = LimiterBoundChatClient(raw_team_client, worker_limiter)
    decision_client = (
        raw_team_client
        if decision_base_url == args.base_url and args.decision_api_key is None
        else OpenAICompatibleClient(
            base_url=decision_base_url,
            api_key=args.decision_api_key or args.api_key,
            temperature=0.0,
            json_mode=True,
            max_tokens=args.max_tokens,
        )
    )
    prompts = PromptCatalog()
    executor = ActionExecutor(
        RuntimeState(
            concurrency=args.initial_concurrency,
            max_concurrency=args.max_concurrency,
            learning_rate=2e-5,
        )
    )
    executor.bind_concurrency_handler(worker_limiter.set_limit)
    team_game = TeamGame(
        client=team_client,
        prompts=prompts,
        model=args.model,
        max_revisions=args.max_revisions,
    )
    decision_agent = DecisionAgent(
        client=decision_client,
        prompts=prompts,
        executor=executor,
        model=decision_model,
        max_attempts=args.decision_attempts,
        prompt_name="smoke",
    )
    task_tracker = TaskTelemetryTracker()
    work_tasks: List["asyncio.Task[TrainingRecord]"] = []
    submissions: List[dict[str, Any]] = []
    failures: List[dict[str, str]] = []
    submission_lock = asyncio.Lock()

    async def run_work_item(task_id: str, source_id: str) -> TrainingRecord:
        await task_tracker.start()
        try:
            record = await team_game.run(
                task_id=task_id,
                source_id=source_id,
                source=PUBLIC_SMOKE_SOURCE,
            )
        except BaseException as error:
            await task_tracker.fail()
            failures.append(
                {
                    "task_id": task_id,
                    "source_id": source_id,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                }
            )
            raise
        await task_tracker.complete()
        return record

    async def submit_work_item(action: Optional[PlanAction], origin: str) -> bool:
        """Create a bounded Team Game job and yield so the limiter can admit it."""

        async with submission_lock:
            action_name = action.action if action is not None else "initial_work_item"
            if len(work_tasks) >= args.max_work_items:
                submissions.append(
                    {
                        "action": action_name,
                        "origin": origin,
                        "status": "capacity_reached",
                        "max_work_items": args.max_work_items,
                    }
                )
                return False
            item_number = len(work_tasks) + 1
            task_id = f"live-gpu-smoke-{item_number:03d}"
            source_id = f"public-gpu-fixture-{item_number:03d}"
            await task_tracker.enqueue()
            work_tasks.append(asyncio.create_task(run_work_item(task_id, source_id), name=task_id))
            submissions.append(
                {
                    "action": action_name,
                    "origin": origin,
                    "status": "submitted",
                    "task_id": task_id,
                    "source_id": source_id,
                }
            )
        await asyncio.sleep(0)
        return True

    async def handle_qa_synth(action: PlanAction) -> None:
        await submit_work_item(action, "adaptive_controller")

    executor.bind_stage_handler("qa_synth", handle_qa_synth)

    engine_queue: asyncio.Queue = asyncio.Queue()
    task_queue: asyncio.Queue = asyncio.Queue()
    if args.telemetry_source == "fixture":
        engine_producer = VLLMMetricsProducer(
            engine_queue,
            snapshot_provider=lambda: {
                "kv_cache_usage": args.kv_cache_usage,
                "pending_requests": args.pending_requests,
                "active_requests": worker_limiter.running_tasks,
            },
        )
    else:
        engine_producer = VLLMMetricsProducer(
            engine_queue,
            base_url=args.base_url,
            metrics_url=args.metrics_url,
            timeout_s=args.metrics_timeout_s,
        )
    task_producer = TaskTelemetryProducer(task_queue, task_tracker.snapshot)
    controller = AdaptiveRuntimeController(
        engine_producer=engine_producer,
        task_producer=task_producer,
        telemetry_merger=TelemetryQueueMerger(engine_queue, task_queue),
        decision_agent=decision_agent,
        action_executor=executor,
    )
    try:
        for _ in range(args.initial_work_items):
            await submit_work_item(None, "initial")
        control_cycles = await controller.run(
            start_timestep=1,
            poll_interval_s=args.telemetry_interval_s,
            max_cycles=args.telemetry_cycles,
            timeout_s=args.metrics_timeout_s,
        )
    finally:
        await controller.aclose()
    if not control_cycles:
        raise RuntimeError("live telemetry loop did not produce a decision cycle")
    latest_cycle = control_cycles[-1]
    work_results = await asyncio.gather(*work_tasks, return_exceptions=True)
    records = [result for result in work_results if not isinstance(result, BaseException)]
    approved_records = [record for record in records if record.approved]
    task_snapshot = await task_tracker.snapshot()

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "record": records[0].to_dict() if records else None,
        "records": [record.to_dict() for record in records],
        **latest_cycle.to_dict(),
        "control_cycles": [cycle.to_dict() for cycle in control_cycles],
    }
    manifest.update(
        {
            "mode": "live_vllm_gpu_smoke",
            "model": args.model,
            "decision_model": decision_model,
            "decision_base_url": decision_base_url,
            "worker_limiter": {"limit": worker_limiter.limit, "running_tasks": worker_limiter.running_tasks},
            "telemetry": {
                "source": args.telemetry_source,
                "metrics_url": engine_producer.metrics_url,
                "cycles": args.telemetry_cycles,
            },
            "workload": {
                "initial_work_items": args.initial_work_items,
                "max_work_items": args.max_work_items,
                "submitted_work_items": sum(item["status"] == "submitted" for item in submissions),
                "completed_work_items": len(records),
                "approved_records": len(approved_records),
                "task_telemetry": {
                    "queued_tasks": task_snapshot.queued_tasks,
                    "running_tasks": task_snapshot.running_tasks,
                    "completed_tasks": task_snapshot.completed_tasks,
                    "failed_tasks": task_snapshot.failed_tasks,
                    "total_tasks": task_snapshot.total_tasks,
                },
            },
            "submissions": submissions,
            "failures": failures,
        }
    )
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    write_training_records(records, output_dir / "training_records.jsonl")
    rows = SFTDatasetBuilder.write_jsonl(records, output_dir / "sft_messages.jsonl")
    if rows != len(approved_records):
        raise RuntimeError("approved SFT row count did not match completed Team Game records")
    if failures:
        raise RuntimeError("one or more live Team Game work items failed")
    if not approved_records:
        raise RuntimeError("no approved live Team Game record can enter the SFT smoke")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--api-key", default="unused")
    parser.add_argument("--model", default="gframe-smoke")
    parser.add_argument("--decision-model", help="model for the decision loop; defaults to --model")
    parser.add_argument(
        "--decision-base-url",
        help="OpenAI-compatible decision endpoint; defaults to --base-url",
    )
    parser.add_argument(
        "--decision-api-key",
        help="API key for --decision-base-url; defaults to --api-key",
    )
    parser.add_argument("--output-dir", default="runs/gpu-smoke")
    parser.add_argument("--initial-concurrency", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--initial-work-items", type=int, default=1)
    parser.add_argument("--max-work-items", type=int, default=2)
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--decision-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument(
        "--telemetry-source",
        choices=("metrics", "fixture"),
        default="metrics",
        help="read vLLM /metrics by default; fixture is only for an endpoint-free smoke invocation",
    )
    parser.add_argument("--metrics-url", help="override the /metrics URL derived from --base-url")
    parser.add_argument("--metrics-timeout-s", type=float, default=3.0)
    parser.add_argument("--telemetry-cycles", type=int, default=1)
    parser.add_argument("--telemetry-interval-s", type=float, default=0.0)
    parser.add_argument("--kv-cache-usage", type=float, default=0.42)
    parser.add_argument("--pending-requests", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be at least 1")
    if not 1 <= args.initial_concurrency <= args.max_concurrency:
        raise SystemExit("--initial-concurrency must be between 1 and --max-concurrency")
    if not 1 <= args.initial_work_items <= args.max_work_items <= 8:
        raise SystemExit("--initial-work-items must be at least 1 and no greater than --max-work-items (maximum 8)")
    if not 0.0 <= args.kv_cache_usage <= 1.0:
        raise SystemExit("--kv-cache-usage must be between 0 and 1")
    if args.metrics_timeout_s <= 0:
        raise SystemExit("--metrics-timeout-s must be positive")
    if args.telemetry_cycles < 1:
        raise SystemExit("--telemetry-cycles must be at least 1")
    if args.telemetry_interval_s < 0:
        raise SystemExit("--telemetry-interval-s must be non-negative")
    output_dir = asyncio.run(run_live_smoke(args))
    print(f"live GPU smoke completed: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
