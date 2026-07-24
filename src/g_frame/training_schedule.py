"""Run-local coordination for pretraining and two full-parameter SFT stages."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .actions import ActionExecutor, RuntimeState
from .pipeline import PipelineArtifacts, run_mock_pipeline
from .schemas import DecisionPlan, PlanAction


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"required training artifact does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"training artifact is not valid JSON: {path}: {error.msg}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"training artifact must be a JSON object: {path}")
    return value


def _read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise RuntimeError(f"required stage input does not exist: {path}") from error
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid stage input {path}:{line_number}: {error.msg}") from error
        if not isinstance(row, Mapping):
            raise RuntimeError(f"invalid stage input {path}:{line_number}: expected object")
        rows.append(dict(row))
    if not rows:
        raise RuntimeError(f"required stage input has no rows: {path}")
    return rows


def _normalize_data_mix(
    data_mix: Mapping[str, float], source_paths: Mapping[str, Path]
) -> Dict[str, float]:
    if not isinstance(data_mix, Mapping) or not data_mix:
        raise ValueError("data_mix must be a non-empty mapping")
    unknown = sorted(str(name) for name in data_mix if str(name) not in source_paths)
    if unknown:
        raise ValueError("data_mix names have no stage input source: " + ", ".join(unknown))
    weights: Dict[str, float] = {}
    for name in source_paths:
        value = data_mix.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"data_mix.{name} must be a finite non-negative number")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"data_mix.{name} must be a finite non-negative number")
        weights[name] = weight
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("data_mix weights must sum to a positive value")
    return {name: weight / total for name, weight in weights.items()}


def _materialize_weighted_jsonl_mix(
    output_path: Path,
    source_paths: Mapping[str, Path],
    data_mix: Mapping[str, float],
) -> Dict[str, Any]:
    """Build a deterministic, run-local weighted SFT input from typed artifacts."""

    weights = _normalize_data_mix(data_mix, source_paths)
    active_sources = [name for name, weight in weights.items() if weight > 0]
    source_rows = {name: _read_jsonl_rows(source_paths[name]) for name in active_sources}
    target_rows = max(10, sum(len(rows) for rows in source_rows.values()))
    exact_counts = {name: weights[name] * target_rows for name in active_sources}
    counts = {name: int(math.floor(exact_counts[name])) for name in active_sources}
    remaining = target_rows - sum(counts.values())
    for name in sorted(
        active_sources,
        key=lambda item: (-(exact_counts[item] - counts[item]), item),
    )[:remaining]:
        counts[name] += 1

    mixed_rows: List[Dict[str, Any]] = []
    for name in sorted(active_sources):
        rows = source_rows[name]
        for index in range(counts[name]):
            row = dict(rows[index % len(rows)])
            row["data_mix_source"] = name
            row["data_mix_weight"] = weights[name]
            mixed_rows.append(row)
    if not mixed_rows:
        raise RuntimeError("data_mix did not select any rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in mixed_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "dataset_path": str(output_path),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "weights": weights,
        "rows": len(mixed_rows),
    }


@dataclass(frozen=True)
class CheckpointScore:
    path: str
    step: int
    eval_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "step": self.step, "eval_loss": self.eval_loss}


@dataclass(frozen=True)
class StageSelection:
    stage: str
    candidates: Sequence[CheckpointScore]
    selected: CheckpointScore

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "selection_metric": "eval_loss",
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected": self.selected.to_dict(),
        }


class BoundedTrainingJudge:
    """Produce bounded next-stage settings after metric-based selection.

    The judge never selects a checkpoint.  It only proposes actions accepted by
    :class:`ActionExecutor`, which validates the numerical bounds before the
    coordinator puts the values into the next stage configuration.
    """

    def __init__(self, *, min_learning_rate: float = 1e-6, max_learning_rate: float = 1e-3) -> None:
        if not 0 < min_learning_rate <= max_learning_rate:
            raise ValueError("invalid learning-rate bounds")
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate

    def propose(
        self,
        *,
        selected: CheckpointScore,
        learning_rate: float,
        data_mix: Mapping[str, float],
    ) -> Dict[str, Any]:
        if not math.isfinite(selected.eval_loss):
            raise ValueError("judge requires a finite held-out loss")
        factor = 0.9 if selected.eval_loss > 1.0 else 1.0
        proposed_lr = min(self.max_learning_rate, max(self.min_learning_rate, learning_rate * factor))
        normalized_mix = {str(name): float(weight) for name, weight in data_mix.items()}
        executor = ActionExecutor(
            RuntimeState(
                learning_rate=learning_rate,
                data_mix=normalized_mix,
            )
        )
        plan = DecisionPlan(
            rationale="Use the measured held-out loss only to bound next-stage parameter advice.",
            actions=[
                PlanAction("set_learning_rate", {"value": proposed_lr}),
                PlanAction("set_data_mix", {"value": normalized_mix}),
            ],
        )
        executions = executor.execute_plan(plan)
        return {
            "selection_is_external": True,
            "input_eval_loss": selected.eval_loss,
            "bounds": {
                "learning_rate": [self.min_learning_rate, self.max_learning_rate],
                "actions": ["set_learning_rate", "set_data_mix"],
            },
            "plan": plan.to_dict(),
            "executions": [item.to_dict() for item in executions],
            "next_parameters": {
                "learning_rate": executor.state.learning_rate,
                "data_mix": executor.state.data_mix,
            },
        }


def _checkpoint_candidates(output_dir: Path) -> List[Path]:
    return sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(path.name.split("-", 1)[1]) if path.name.split("-", 1)[1].isdigit() else -1,
    )


def collect_checkpoint_scores(output_dir: Path) -> List[CheckpointScore]:
    """Read held-out losses emitted at each saved checkpoint by a stage runner."""

    output_dir = output_dir.expanduser().resolve()
    checkpoint_paths = _checkpoint_candidates(output_dir)
    raw_metrics: Dict[str, Mapping[str, Any]] = {}
    metrics_path = output_dir / "checkpoint_metrics.json"
    if metrics_path.is_file():
        payload = _read_json(metrics_path)
        for raw_path, metric in payload.items():
            if isinstance(raw_path, str) and isinstance(metric, Mapping):
                raw_metrics[str(Path(raw_path).expanduser().resolve())] = metric

    state_path = output_dir / "trainer_state.json"
    if state_path.is_file():
        state = _read_json(state_path)
        history = state.get("log_history", [])
        if isinstance(history, list):
            for event in history:
                if not isinstance(event, Mapping) or "eval_loss" not in event:
                    continue
                try:
                    step = int(event.get("step", 0))
                    eval_loss = float(event["eval_loss"])
                except (TypeError, ValueError):
                    continue
                checkpoint = output_dir / f"checkpoint-{step}"
                key = str((checkpoint if checkpoint.is_dir() else output_dir).resolve())
                raw_metrics.setdefault(key, {"step": step, "eval_loss": eval_loss})

    eval_path = output_dir / "eval_results.json"
    if eval_path.is_file():
        evaluation = _read_json(eval_path)
        if "eval_loss" in evaluation:
            raw_metrics.setdefault(
                str(output_dir),
                {"step": int(evaluation.get("step", 0)), "eval_loss": evaluation["eval_loss"]},
            )

    missing_checkpoint_metrics = [
        path for path in checkpoint_paths if str(path.resolve()) not in raw_metrics
    ]
    if missing_checkpoint_metrics:
        missing = ", ".join(str(path) for path in missing_checkpoint_metrics)
        raise RuntimeError(f"held-out evaluation is missing for checkpoint(s): {missing}")

    scores: List[CheckpointScore] = []
    for path in [*checkpoint_paths, output_dir]:
        metric = raw_metrics.get(str(path.resolve()))
        if metric is None:
            continue
        try:
            score = CheckpointScore(
                path=str(path.resolve()),
                step=int(metric.get("step", 0)),
                eval_loss=float(metric["eval_loss"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"invalid held-out metric for {path}") from error
        if math.isfinite(score.eval_loss):
            scores.append(score)
    if not scores:
        raise RuntimeError(f"no finite held-out loss was recorded for {output_dir}")
    return scores


def select_checkpoint(stage: str, output_dir: Path, *, require_checkpoint: bool = False) -> StageSelection:
    scores = collect_checkpoint_scores(output_dir)
    checkpoint_scores = [score for score in scores if Path(score.path).name.startswith("checkpoint-")]
    if require_checkpoint and not checkpoint_scores:
        raise RuntimeError(f"{stage} did not produce an evaluated checkpoint for resumption")
    selection_pool = checkpoint_scores or scores
    selected = min(selection_pool, key=lambda score: (score.eval_loss, -score.step, score.path))
    return StageSelection(stage=stage, candidates=scores, selected=selected)


def _run_limited(command: Sequence[str], *, cwd: Path, timeout_s: int, environment: Mapping[str, str]) -> Dict[str, Any]:
    """Run one GPU stage under a hard wall-clock limit and retain its log."""

    if timeout_s < 1 or timeout_s > 300:
        raise ValueError("stage timeout must be between 1 and 300 seconds")
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        raise RuntimeError(f"stage exceeded its {timeout_s}-second limit\n{output[-4000:]}")
    if process.returncode != 0:
        raise RuntimeError(f"stage command failed ({process.returncode})\n{output[-4000:]}")
    return {"command": list(command), "returncode": process.returncode, "output": output}


class MockTrainingCoordinator:
    """Coordinate a run-local pretrain -> SFT-1 -> SFT-2 handoff."""

    def __init__(
        self,
        output_dir: Path,
        model_path: str,
        *,
        repo_root: Optional[Path] = None,
        gpu_index: int = 1,
        stage_timeout_s: int = 270,
    ) -> None:
        if gpu_index < 0:
            raise ValueError("gpu_index must be non-negative")
        if stage_timeout_s < 1 or stage_timeout_s > 300:
            raise ValueError("stage_timeout_s must be between 1 and 300 seconds")
        self.output_dir = output_dir.expanduser().resolve()
        self.model_path = model_path
        self.repo_root = (repo_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
        self.gpu_index = gpu_index
        self.stage_timeout_s = stage_timeout_s
        self.artifacts = PipelineArtifacts(self.output_dir)
        self.config_dir = self.output_dir / "stage_configs"
        self.manifest_path = self.output_dir / "schedule_manifest.json"
        self.judge = BoundedTrainingJudge()
        self._stage_input_mixes: Dict[str, Dict[str, Any]] = {}

    @property
    def _pretrain_output(self) -> Path:
        return self.output_dir / "pretrain"

    @property
    def _sft_stage1_output(self) -> Path:
        return self.output_dir / "sft_stage1"

    @property
    def _sft_stage2_output(self) -> Path:
        return self.output_dir / "sft_stage2"

    def _ensure_mock_artifacts(self) -> None:
        required = (
            self.artifacts.pretrain_train,
            self.artifacts.pretrain_eval,
            self.artifacts.sft_stage1,
            self.artifacts.sft_stage2,
            self.artifacts.sft_eval,
        )
        if not all(path.is_file() for path in required):
            asyncio.run(run_mock_pipeline(self.output_dir))

    def _pretrain_config(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "tokenizer_path": self.model_path,
            "dataset_path": str(self.artifacts.pretrain_train),
            "dataset_format": "jsonl_text",
            "eval_dataset_path": str(self.artifacts.pretrain_eval),
            "eval_dataset_format": "jsonl_text",
            "eval_strategy": "steps",
            "eval_steps": 1,
            "output_dir": str(self._pretrain_output),
            "deepspeed_config": str(self.repo_root / "llm_training" / "ds_config_pretrain.json"),
            "run_name": "g-frame-v2-mock-pretrain",
            "learning_rate": 3e-5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 1,
            "logging_steps": 1,
            "save_strategy": "steps",
            "save_steps": 1,
            "save_total_limit": 2,
            "bf16": True,
            "gradient_checkpointing": False,
            "max_seq_length": 256,
            "seed": 42,
        }

    def _sft_config(
        self,
        *,
        stage: str,
        model_path: str,
        dataset_path: Path,
        output_dir: Path,
        learning_rate: float,
        max_steps: int,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "model_path": model_path,
            "dataset_path": str(dataset_path),
            "dataset_format": "jsonl_messages",
            "eval_dataset_path": str(self.artifacts.sft_eval),
            "eval_dataset_format": "jsonl_messages",
            "eval_strategy": "steps",
            "eval_steps": 1,
            "output_dir": str(output_dir),
            "deepspeed_config": str(self.repo_root / "llm_training" / "ds_config_sft_smoke.json"),
            "run_name": f"g-frame-v2-mock-{stage}",
            "learning_rate": learning_rate,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": max_steps,
            "logging_steps": 1,
            "save_strategy": "steps",
            "save_steps": 1,
            "save_total_limit": 2,
            "bf16": True,
            "gradient_checkpointing": False,
            "max_seq_length": 256,
            "seed": 42,
        }
        if resume_from_checkpoint is not None:
            config["resume_from_checkpoint"] = resume_from_checkpoint
        return config

    def _stage_input_sources(self) -> Dict[str, Path]:
        return {
            "general": self.artifacts.sft_stage1,
            "domain": self.artifacts.sft_stage2,
        }

    def _write_configs(
        self,
        *,
        stage1_model_path: Optional[str] = None,
        stage2_model_path: Optional[str] = None,
        stage2_resume: Optional[str] = None,
        stage1_lr: float = 5e-5,
        stage2_lr: float = 5e-5,
        stage1_data_mix: Optional[Mapping[str, float]] = None,
        stage2_data_mix: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, Path]:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._stage_input_mixes = {}
        paths = {
            "pretrain": self.config_dir / "pretrain.json",
            "sft_stage1": self.config_dir / "sft_stage1.json",
            "sft_stage2": self.config_dir / "sft_stage2.json",
        }
        stage1_dataset = self.artifacts.sft_stage1
        stage2_dataset = self.artifacts.sft_stage2
        source_paths = self._stage_input_sources()
        if stage1_data_mix is not None:
            metadata = _materialize_weighted_jsonl_mix(
                self.output_dir / "stage_inputs" / "sft_stage1.mixed.jsonl",
                source_paths,
                stage1_data_mix,
            )
            stage1_dataset = Path(metadata["dataset_path"])
            self._stage_input_mixes["sft_stage1"] = metadata
        if stage2_data_mix is not None:
            metadata = _materialize_weighted_jsonl_mix(
                self.output_dir / "stage_inputs" / "sft_stage2.mixed.jsonl",
                source_paths,
                stage2_data_mix,
            )
            stage2_dataset = Path(metadata["dataset_path"])
            self._stage_input_mixes["sft_stage2"] = metadata
        _write_json(paths["pretrain"], self._pretrain_config())
        _write_json(
            paths["sft_stage1"],
            self._sft_config(
                stage="sft-stage1",
                model_path=stage1_model_path or self.model_path,
                dataset_path=stage1_dataset,
                output_dir=self._sft_stage1_output,
                learning_rate=stage1_lr,
                max_steps=1,
            ),
        )
        _write_json(
            paths["sft_stage2"],
            self._sft_config(
                stage="sft-stage2",
                model_path=stage2_model_path or self.model_path,
                dataset_path=stage2_dataset,
                output_dir=self._sft_stage2_output,
                learning_rate=stage2_lr,
                max_steps=2 if stage2_resume is not None else 1,
                resume_from_checkpoint=stage2_resume,
            ),
        )
        return paths

    def _validate_configs(self, paths: Mapping[str, Path]) -> Dict[str, Any]:
        from llm_training.run_pretraining import load_config as load_pretrain_config
        from llm_training.run_pretraining import validate_job as validate_pretrain_job
        from llm_training.run_sft import load_sft_config, validate_job as validate_sft_job

        return {
            "pretrain": validate_pretrain_job(load_pretrain_config(paths["pretrain"]), check_input=True),
            "sft_stage1": validate_sft_job(load_sft_config(paths["sft_stage1"]), check_input=True),
            "sft_stage2": validate_sft_job(load_sft_config(paths["sft_stage2"]), check_input=True),
        }

    def _launcher_command(self, runner: str, config_path: Path) -> List[str]:
        return [
            sys.executable,
            "-m",
            "deepspeed.launcher.runner",
            "--include",
            f"localhost:{self.gpu_index}",
            runner,
            "--config",
            str(config_path),
            "--run",
        ]

    def _run_stage(self, stage: str, runner: str, config_path: Path) -> Dict[str, Any]:
        result = _run_limited(
            self._launcher_command(runner, config_path),
            cwd=self.repo_root,
            timeout_s=self.stage_timeout_s,
            environment=os.environ,
        )
        log_path = self.output_dir / "logs" / f"{stage}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(str(result.pop("output")), encoding="utf-8")
        result["log"] = str(log_path.relative_to(self.output_dir))
        return result

    def _persist(self, manifest: Mapping[str, Any]) -> None:
        _write_json(self.manifest_path, manifest)

    def validate(self) -> Dict[str, Any]:
        """Generate mock inputs/configs and validate every stage without GPUs."""

        self._ensure_mock_artifacts()
        config_paths = self._write_configs()
        manifest: Dict[str, Any] = {
            "mode": "validate",
            "artifact_root": str(self.output_dir),
            "stage_timeout_seconds": self.stage_timeout_s,
            "configs": {name: str(path.relative_to(self.output_dir)) for name, path in config_paths.items()},
            "validation": self._validate_configs(config_paths),
            "checkpoint_convention": {
                "stage_outputs": ["pretrain", "sft_stage1", "sft_stage2"],
                "selection_metric": "eval_loss",
                "stage1_model_source": "pretrain selected checkpoint",
                "stage2_resume_source": "sft_stage1 selected checkpoint",
            },
        }
        self._persist(manifest)
        return manifest

    def run(self) -> Dict[str, Any]:
        """Run all three limited stages and record metric-based selections."""

        validation_manifest = self.validate()
        config_paths = {name: self.output_dir / path for name, path in validation_manifest["configs"].items()}
        manifest: Dict[str, Any] = {
            "mode": "run",
            "artifact_root": str(self.output_dir),
            "stage_timeout_seconds": self.stage_timeout_s,
            "configs": validation_manifest["configs"],
            "validation": validation_manifest["validation"],
            "stages": {},
            "checkpoint_convention": validation_manifest["checkpoint_convention"],
        }
        self._persist(manifest)

        manifest["stages"]["pretrain"] = self._run_stage(
            "pretrain", "llm_training/run_pretraining.py", config_paths["pretrain"]
        )
        pretrain_selection = select_checkpoint("pretrain", self._pretrain_output)
        pretrain_advice = self.judge.propose(
            selected=pretrain_selection.selected,
            learning_rate=5e-5,
            data_mix={"general": 0.4, "domain": 0.6},
        )
        stage1_data_mix = dict(pretrain_advice["next_parameters"]["data_mix"])
        manifest["stages"]["pretrain"].update(
            {"selection": pretrain_selection.to_dict(), "next_stage_advice": pretrain_advice}
        )
        self._persist(manifest)

        config_paths = self._write_configs(
            stage1_model_path=pretrain_selection.selected.path,
            stage1_lr=float(pretrain_advice["next_parameters"]["learning_rate"]),
            stage1_data_mix=stage1_data_mix,
        )
        manifest["configs"] = {name: str(path.relative_to(self.output_dir)) for name, path in config_paths.items()}
        manifest["validation"] = self._validate_configs(config_paths)
        manifest["stage_input_mixes"] = dict(self._stage_input_mixes)
        manifest["stages"]["sft_stage1"] = self._run_stage(
            "sft_stage1", "llm_training/run_sft.py", config_paths["sft_stage1"]
        )
        stage1_selection = select_checkpoint("sft_stage1", self._sft_stage1_output, require_checkpoint=True)
        stage1_advice = self.judge.propose(
            selected=stage1_selection.selected,
            learning_rate=float(pretrain_advice["next_parameters"]["learning_rate"]),
            data_mix={"general": 0.4, "domain": 0.6},
        )
        stage2_data_mix = dict(stage1_advice["next_parameters"]["data_mix"])
        manifest["stages"]["sft_stage1"].update(
            {"selection": stage1_selection.to_dict(), "next_stage_advice": stage1_advice}
        )
        self._persist(manifest)

        config_paths = self._write_configs(
            stage1_model_path=pretrain_selection.selected.path,
            stage2_model_path=str(self._sft_stage1_output),
            stage2_resume=stage1_selection.selected.path,
            stage1_lr=float(pretrain_advice["next_parameters"]["learning_rate"]),
            stage2_lr=float(stage1_advice["next_parameters"]["learning_rate"]),
            stage1_data_mix=stage1_data_mix,
            stage2_data_mix=stage2_data_mix,
        )
        manifest["configs"] = {name: str(path.relative_to(self.output_dir)) for name, path in config_paths.items()}
        manifest["validation"] = self._validate_configs(config_paths)
        manifest["stage_input_mixes"] = dict(self._stage_input_mixes)
        manifest["stages"]["sft_stage2"] = self._run_stage(
            "sft_stage2", "llm_training/run_sft.py", config_paths["sft_stage2"]
        )
        stage2_selection = select_checkpoint("sft_stage2", self._sft_stage2_output, require_checkpoint=True)
        stage2_advice = self.judge.propose(
            selected=stage2_selection.selected,
            learning_rate=float(stage1_advice["next_parameters"]["learning_rate"]),
            data_mix={"general": 0.4, "domain": 0.6},
        )
        manifest["stages"]["sft_stage2"].update(
            {"selection": stage2_selection.to_dict(), "next_stage_advice": stage2_advice}
        )
        self._persist(manifest)
        return manifest


__all__ = [
    "BoundedTrainingJudge",
    "CheckpointScore",
    "MockTrainingCoordinator",
    "StageSelection",
    "collect_checkpoint_scores",
    "select_checkpoint",
]
