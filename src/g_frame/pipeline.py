"""Executable mock-data handlers for the v2 algorithm flow.

The handlers are intentionally small and file-oriented.  They exercise the
same typed action boundary as a deployed workflow while keeping generated
fixtures inside an operator-selected run directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .actions import ActionExecutor
from .augmentation import AugmentedSource, SixStyleAugmenter
from .clients import AsyncChatClient, LimiterBoundChatClient
from .data import FileDatasetProvider, SFTDatasetBuilder, write_training_records
from .prompts import PromptCatalog
from .schemas import AgentMessage, DecisionPlan, PlanAction, TrainingRecord
from .team_game import TeamGame
from .telemetry import DynamicSemaphore


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"required stage artifact does not exist: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL artifact {path}:{line_number}: {error.msg}") from error
            if not isinstance(value, Mapping):
                raise RuntimeError(f"invalid JSONL artifact {path}:{line_number}: expected object")
            rows.append(dict(value))
    if not rows:
        raise RuntimeError(f"required stage artifact has no rows: {path}")
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    if count == 0:
        raise RuntimeError(f"refusing to write an empty stage artifact: {path}")
    return count


def _json_text(raw_response: str, field: str) -> str:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise ValueError(f"model response is not JSON for {field}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"model response for {field} must be an object")
    value = payload.get(field, payload.get("text"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"model response must contain non-empty {field}")
    return value.strip()


def _clean_text(value: str) -> str:
    """Perform deterministic character cleanup and whitespace normalization."""

    cleaned = value.replace("\x00", "").replace("\ufffd", "")
    return " ".join(cleaned.split())


@dataclass(frozen=True)
class PipelineArtifacts:
    root: Path

    @property
    def input_sources(self) -> Path:
        return self.root / "input_sources.jsonl"

    @property
    def clean(self) -> Path:
        return self.root / "clean.jsonl"

    @property
    def augmentation(self) -> Path:
        return self.root / "augmentation.jsonl"

    @property
    def cot_distill(self) -> Path:
        return self.root / "cot_distill.jsonl"

    @property
    def cot_synth(self) -> Path:
        return self.root / "cot_synth.jsonl"

    @property
    def qa_records(self) -> Path:
        return self.root / "qa_records.jsonl"

    @property
    def sft_stage1(self) -> Path:
        return self.root / "sft_stage1.jsonl"

    @property
    def sft_stage2(self) -> Path:
        return self.root / "sft_stage2.jsonl"

    @property
    def sft_eval(self) -> Path:
        return self.root / "sft_eval.jsonl"

    @property
    def pretrain_train(self) -> Path:
        return self.root / "pretrain_train.jsonl"

    @property
    def pretrain_eval(self) -> Path:
        return self.root / "pretrain_eval.jsonl"

    @property
    def health(self) -> Path:
        return self.root / "system_health.json"

    @property
    def manifest(self) -> Path:
        return self.root / "pipeline_manifest.json"

    def to_dict(self) -> Dict[str, str]:
        return {
            name: str(getattr(self, name).relative_to(self.root))
            for name in (
                "input_sources",
                "clean",
                "augmentation",
                "cot_distill",
                "cot_synth",
                "qa_records",
                "sft_stage1",
                "sft_stage2",
                "sft_eval",
                "pretrain_train",
                "pretrain_eval",
                "health",
            )
        }


class MockPipelineClient:
    """Deterministic JSON client used by the run-local mock workflow."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, str]] = []
        self._review_count = 0

    async def complete(self, messages: Sequence[AgentMessage], model: str, role: str) -> str:
        self.requests.append({"model": model, "role": role})
        if role.startswith("augment_"):
            style = role.removeprefix("augment_").replace("_", " ")
            return json.dumps(
                {"augmented_text": f"Mock {style} text preserves the explicit source relationship."}
            )
        if role == "data_clean":
            return json.dumps(
                {"cleaned_text": "The mock material states that concentration changes the observed signal."}
            )
        if role == "cot_distill":
            return json.dumps({"reasoning": "The mock source is converted into a concise grounded trace."})
        if role == "cot_synth":
            return json.dumps({"reasoning": "The synthesized trace retains only the stated mock relationship."})
        if role == "teacher":
            return json.dumps({"question": "What relationship does the source state?"})
        if role == "student":
            return json.dumps(
                {
                    "reasoning": "The source contains one explicit mock relationship.",
                    "answer": "The source states the explicit mock relationship.",
                }
            )
        if role == "reviewer":
            self._review_count += 1
            if self._review_count % 2:
                return json.dumps({"approved": False, "feedback": "State the source relationship directly."})
            return json.dumps({"approved": True, "feedback": "The answer is grounded in the source."})
        if role == "rectifier":
            return json.dumps(
                {
                    "reasoning": "The revised answer follows the reviewer direction.",
                    "answer": "The source states the explicit mock relationship directly.",
                }
            )
        if role == "judger":
            return json.dumps(
                {
                    "final_answer": "The source states the explicit mock relationship directly.",
                    "approved": True,
                    "feedback": "Final answer accepted.",
                }
            )
        raise KeyError(f"no deterministic response is defined for role {role}")


class PipelineStageHandlers:
    """Bind concrete, traceable handlers to all pipeline stage actions."""

    def __init__(
        self,
        output_dir: Path,
        client: AsyncChatClient,
        *,
        model: str = "mock-pipeline",
        prompts: Optional[PromptCatalog] = None,
        source_path: Optional[Path] = None,
        limiter: Optional[DynamicSemaphore] = None,
    ) -> None:
        self.artifacts = PipelineArtifacts(output_dir.expanduser().resolve())
        self.artifacts.root.mkdir(parents=True, exist_ok=True)
        self.client: AsyncChatClient = LimiterBoundChatClient(client, limiter) if limiter is not None else client
        self.model = model
        self.prompts = prompts or PromptCatalog()
        self.source_path = (source_path or self.artifacts.input_sources).expanduser().resolve()
        self.executed: List[str] = []

    def bind(self, executor: ActionExecutor) -> None:
        executor.bind_stage_handler("data_clean", self.data_clean)
        executor.bind_stage_handler("data_augment", self.data_augment)
        executor.bind_stage_handler("cot_distill", self.cot_distill)
        executor.bind_stage_handler("cot_synth", self.cot_synth)
        executor.bind_stage_handler("qa_synth", self.qa_synth)
        executor.bind_stage_handler("model_train", self.model_train)
        executor.bind_stage_handler("system_health_check", self.system_health_check)

    async def data_clean(self, action: PlanAction) -> None:
        source_path = action.arguments.get("source_path", str(self.source_path))
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError("data_clean.source_path must be a non-empty string when provided")
        rows = []
        for record in FileDatasetProvider(Path(source_path)).iter_sources():
            character_normalized = _clean_text(record["source"])
            prompt = self.prompts.render(
                "cleaning",
                "semantic_complete",
                source_id=record["source_id"],
                source=character_normalized,
            )
            raw = await self.client.complete(
                [
                    AgentMessage("system", "Return valid JSON with a grounded cleaned_text field."),
                    AgentMessage("user", prompt),
                ],
                model=self.model,
                role="data_clean",
            )
            cleaned = _clean_text(_json_text(raw, "cleaned_text"))
            if cleaned:
                rows.append(
                    {
                        "source_id": record["source_id"],
                        "source": cleaned,
                        "lineage": {
                            "stage": "data_clean",
                            "input_source_id": record["source_id"],
                            "operations": [
                                "character_symbol_normalize",
                                "semantic_complete",
                                "format_standardize",
                            ],
                        },
                    }
                )
        _write_jsonl(self.artifacts.clean, rows)
        self.executed.append("data_clean")

    async def data_augment(self, action: PlanAction) -> None:
        del action
        augmenter = SixStyleAugmenter(self.client, self.prompts, self.model)
        rows: List[Dict[str, Any]] = []
        for source in _read_jsonl(self.artifacts.clean):
            source_id = str(source["source_id"])
            source_text = str(source["source"])
            generated: Sequence[AugmentedSource] = await augmenter.augment(source_id, source_text)
            for item in generated:
                row = item.to_dict()
                row["lineage"] = {"stage": "data_augment", "input_source_id": source_id}
                rows.append(row)
        _write_jsonl(self.artifacts.augmentation, rows)
        self.executed.append("data_augment")

    async def cot_distill(self, action: PlanAction) -> None:
        del action
        rows: List[Dict[str, Any]] = []
        for index, source in enumerate(_read_jsonl(self.artifacts.augmentation), start=1):
            source_id = str(source["source_id"])
            source_text = str(source["augmented_text"])
            prompt = self.prompts.render("reasoning", "distill", source_id=source_id, source=source_text)
            raw = await self.client.complete(
                [
                    AgentMessage("system", "Return valid JSON with a grounded reasoning field."),
                    AgentMessage("user", prompt),
                ],
                model=self.model,
                role="cot_distill",
            )
            rows.append(
                {
                    "record_id": f"distill-{index}",
                    "source_id": source_id,
                    "source": source_text,
                    "reasoning": _json_text(raw, "reasoning"),
                    "lineage": {"stage": "cot_distill", "augmentation_style": source["style"]},
                }
            )
        _write_jsonl(self.artifacts.cot_distill, rows)
        self.executed.append("cot_distill")

    async def cot_synth(self, action: PlanAction) -> None:
        del action
        rows: List[Dict[str, Any]] = []
        for source in _read_jsonl(self.artifacts.cot_distill):
            source_id = str(source["source_id"])
            source_text = str(source["source"])
            distilled = str(source["reasoning"])
            prompt = self.prompts.render(
                "reasoning", "synth", source_id=source_id, source=source_text, reasoning=distilled
            )
            raw = await self.client.complete(
                [
                    AgentMessage("system", "Return valid JSON with a grounded reasoning field."),
                    AgentMessage("user", prompt),
                ],
                model=self.model,
                role="cot_synth",
            )
            rows.append(
                {
                    "record_id": f"synth-{source['record_id']}",
                    "source_id": source_id,
                    "source": source_text,
                    "reasoning": _json_text(raw, "reasoning"),
                    "lineage": {"stage": "cot_synth", "distill_record_id": source["record_id"]},
                }
            )
        _write_jsonl(self.artifacts.cot_synth, rows)
        self.executed.append("cot_synth")

    async def qa_synth(self, action: PlanAction) -> None:
        del action
        team_game = TeamGame(self.client, self.prompts, self.model, max_revisions=1)
        records: List[TrainingRecord] = []
        for index, source in enumerate(_read_jsonl(self.artifacts.cot_synth), start=1):
            records.append(
                await team_game.run(
                    task_id=f"qa-{index}",
                    source_id=str(source["source_id"]),
                    source=f"{source['source']}\n\nReasoning trace: {source['reasoning']}",
                )
            )
        write_training_records(records, self.artifacts.qa_records)
        self.executed.append("qa_synth")

    async def model_train(self, action: PlanAction) -> None:
        del action
        records = [TrainingRecord.from_dict(row) for row in _read_jsonl(self.artifacts.qa_records)]
        approved = [record for record in records if record.approved]
        if not approved:
            raise RuntimeError("qa_synth produced no approved records for the training handoff")
        stage1_records = approved[:-1] or approved
        stage2_records = approved[-1:]
        SFTDatasetBuilder.write_jsonl(stage1_records, self.artifacts.sft_stage1)
        SFTDatasetBuilder.write_jsonl(stage2_records, self.artifacts.sft_stage2)
        held_out = {
            "record_id": "mock-held-out-1",
            "messages": [
                {"role": "system", "content": "You are a chemistry expert."},
                {"role": "user", "content": "State the mock held-out relationship."},
                {"role": "assistant", "content": "The held-out source states its explicit relationship."},
            ],
        }
        _write_jsonl(self.artifacts.sft_eval, [held_out])
        pretrain_rows = [
            {"text": str(row["source"])} for row in _read_jsonl(self.artifacts.clean)
        ] + [
            {"text": str(row["augmented_text"])} for row in _read_jsonl(self.artifacts.augmentation)
        ]
        _write_jsonl(self.artifacts.pretrain_train, pretrain_rows)
        _write_jsonl(
            self.artifacts.pretrain_eval,
            [{"text": "The held-out mock text contains a single explicit chemistry relationship."}],
        )
        self.executed.append("model_train")

    async def system_health_check(self, action: PlanAction) -> None:
        del action
        payload = {
            "status": "ok",
            "executed_stages": list(self.executed),
            "artifact_root": str(self.artifacts.root),
        }
        self.artifacts.health.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.executed.append("system_health_check")


def create_mock_input(path: Path) -> Path:
    """Create one small run-local source record for an offline workflow check."""

    resolved = path.expanduser().resolve()
    _write_jsonl(
        resolved,
        [
            {
                "source_id": "mock-source-1",
                "source": "The mock material states that concentration changes the observed signal.",
            }
        ],
    )
    return resolved


async def run_mock_pipeline(
    output_dir: Path,
    *,
    source_path: Optional[Path] = None,
    model: str = "mock-pipeline",
) -> Dict[str, Any]:
    """Run every stage handler once using run-local mock inputs and responses."""

    output_dir = output_dir.expanduser().resolve()
    artifacts = PipelineArtifacts(output_dir)
    input_path = source_path.expanduser().resolve() if source_path is not None else create_mock_input(artifacts.input_sources)
    executor = ActionExecutor()
    client = MockPipelineClient()
    handlers = PipelineStageHandlers(output_dir, client, model=model, source_path=input_path)
    handlers.bind(executor)
    plan = DecisionPlan(
        rationale="Run each local mock stage once to verify the typed handoff.",
        actions=[
            PlanAction("data_clean", {"idempotency_key": "mock-clean", "source_path": str(input_path)}),
            PlanAction("data_augment", {"idempotency_key": "mock-augment"}),
            PlanAction("cot_distill", {"idempotency_key": "mock-distill"}),
            PlanAction("cot_synth", {"idempotency_key": "mock-synth"}),
            PlanAction("qa_synth", {"idempotency_key": "mock-qa"}),
            PlanAction("model_train", {"idempotency_key": "mock-train-handoff"}),
            PlanAction("system_health_check", {"idempotency_key": "mock-health"}),
        ],
    )
    executions = await executor.execute_plan_async(plan)
    manifest = {
        "mode": "mock",
        "plan": plan.to_dict(),
        "executions": [item.to_dict() for item in executions],
        "stage_handlers": list(handlers.executed),
        "model_request_roles": [request["role"] for request in client.requests],
        "runtime_state": executor.state.to_dict(),
        "artifacts": artifacts.to_dict(),
        "counts": {
            "clean": len(_read_jsonl(artifacts.clean)),
            "augmentation": len(_read_jsonl(artifacts.augmentation)),
            "cot_distill": len(_read_jsonl(artifacts.cot_distill)),
            "cot_synth": len(_read_jsonl(artifacts.cot_synth)),
            "qa_records": len(_read_jsonl(artifacts.qa_records)),
            "sft_stage1": len(_read_jsonl(artifacts.sft_stage1)),
            "sft_stage2": len(_read_jsonl(artifacts.sft_stage2)),
            "sft_eval": len(_read_jsonl(artifacts.sft_eval)),
        },
    }
    artifacts.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "MockPipelineClient",
    "PipelineArtifacts",
    "PipelineStageHandlers",
    "create_mock_input",
    "run_mock_pipeline",
]
