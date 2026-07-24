"""DeepSpeed full-parameter supervised fine-tuning for G-Frame v2.

This is the v2 replacement for the import-time v1 training script.  It keeps
the legacy tokenized Hugging Face dataset path while adding direct support for
the structured ``messages`` JSONL emitted by the v2 Team Game workflow.

Configuration validation is intentionally CPU-safe.  Training starts only
when ``--run`` is supplied, normally through the DeepSpeed launcher::

    deepspeed --num_gpus=1 llm_training/run_sft.py --config configs/smoke_sft.json --run

The implementation deliberately contains no LoRA, PEFT, adapter, or quantized
adapter path.  Every parameter of the loaded causal language model is trained.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DATASET_FORMAT_JSONL_MESSAGES = "jsonl_messages"
DATASET_FORMAT_TOKENIZED_DISK = "tokenized_disk"
SUPPORTED_DATASET_FORMATS = frozenset(
    {DATASET_FORMAT_JSONL_MESSAGES, DATASET_FORMAT_TOKENIZED_DISK}
)

_FORMAT_ALIASES = {
    "jsonl": DATASET_FORMAT_JSONL_MESSAGES,
    "v2_jsonl": DATASET_FORMAT_JSONL_MESSAGES,
    "v2_jsonl_messages": DATASET_FORMAT_JSONL_MESSAGES,
    "jsonl_messages": DATASET_FORMAT_JSONL_MESSAGES,
    "tokenized": DATASET_FORMAT_TOKENIZED_DISK,
    "tokenized_disk": DATASET_FORMAT_TOKENIZED_DISK,
    "legacy_tokenized": DATASET_FORMAT_TOKENIZED_DISK,
}
_FORBIDDEN_TUNING_TERMS = ("lora", "qlora", "peft", "adapter", "prefix_tuning")
_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True)
class SFTConfig:
    """Validated configuration for a full-parameter DeepSpeed SFT job."""

    model_path: str
    dataset_path: str
    output_dir: str
    deepspeed_config: str
    dataset_format: str = DATASET_FORMAT_JSONL_MESSAGES
    dataset_paths: Tuple[str, ...] = ()
    eval_dataset_path: str | None = None
    eval_dataset_format: str | None = None
    run_name: str = "g-frame-v2-sft"
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    max_steps: int = -1
    logging_steps: int = 1
    save_steps: int = 100
    save_strategy: str = "steps"
    save_total_limit: int = 1
    eval_strategy: str = "no"
    eval_steps: int = 1
    bf16: bool = True
    seed: int = 42
    max_seq_length: int = 2048
    preprocessing_num_proc: int = 1
    add_special_tokens: bool = False
    special_tokens: Tuple[str, ...] = ()
    trust_remote_code: bool = False
    gradient_checkpointing: bool = False
    report_to: Tuple[str, ...] = ()
    resume_from_checkpoint: str | None = None

    @property
    def all_dataset_paths(self) -> Tuple[str, ...]:
        return self.dataset_paths or (self.dataset_path,)

    def validate(self) -> None:
        _require_nonempty_string("model_path", self.model_path)
        for name in ("dataset_path", "output_dir", "deepspeed_config", "run_name"):
            _require_nonempty_string(name, getattr(self, name))
        if self.dataset_format not in SUPPORTED_DATASET_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_DATASET_FORMATS))
            raise ValueError(f"dataset_format must be one of: {supported}")
        if not self.all_dataset_paths:
            raise ValueError("at least one dataset path is required")
        for path in self.all_dataset_paths:
            _require_nonempty_string("dataset_path", path)
        if self.eval_dataset_path is not None:
            _require_nonempty_string("eval_dataset_path", self.eval_dataset_path)
        if self.eval_dataset_format is not None and self.eval_dataset_format not in SUPPORTED_DATASET_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_DATASET_FORMATS))
            raise ValueError(f"eval_dataset_format must be one of: {supported}")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.per_device_train_batch_size < 1:
            raise ValueError("per_device_train_batch_size must be at least 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between 0 and 1")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or a positive integer")
        if self.logging_steps < 1 or self.save_steps < 1:
            raise ValueError("logging_steps and save_steps must be at least 1")
        if self.save_strategy not in {"no", "steps", "epoch"}:
            raise ValueError("save_strategy must be 'no', 'steps', or 'epoch'")
        if self.save_total_limit < 1:
            raise ValueError("save_total_limit must be at least 1")
        if self.eval_strategy not in {"no", "steps", "epoch"}:
            raise ValueError("eval_strategy must be 'no', 'steps', or 'epoch'")
        if self.eval_steps < 1:
            raise ValueError("eval_steps must be at least 1")
        if self.eval_strategy != "no" and self.eval_dataset_path is None:
            raise ValueError("eval_dataset_path is required when eval_strategy is enabled")
        if self.max_seq_length < 8:
            raise ValueError("max_seq_length must be at least 8")
        if self.preprocessing_num_proc < 1:
            raise ValueError("preprocessing_num_proc must be at least 1")
        if not isinstance(self.bf16, bool):
            raise ValueError("bf16 must be a boolean")
        if not isinstance(self.add_special_tokens, bool):
            raise ValueError("add_special_tokens must be a boolean")
        if not isinstance(self.trust_remote_code, bool):
            raise ValueError("trust_remote_code must be a boolean")
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("gradient_checkpointing must be a boolean")
        if any(not isinstance(token, str) or not token.strip() for token in self.special_tokens):
            raise ValueError("special_tokens must contain non-empty strings")
        if any(not isinstance(target, str) or not target.strip() for target in self.report_to):
            raise ValueError("report_to must contain non-empty strings")
        if self.resume_from_checkpoint is not None:
            _require_nonempty_string("resume_from_checkpoint", self.resume_from_checkpoint)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "dataset_path": self.dataset_path,
            "dataset_paths": list(self.all_dataset_paths),
            "dataset_format": self.dataset_format,
            "eval_dataset_path": self.eval_dataset_path,
            "eval_dataset_format": self.eval_dataset_format,
            "output_dir": self.output_dir,
            "deepspeed_config": self.deepspeed_config,
            "run_name": self.run_name,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_steps": self.max_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_strategy": self.save_strategy,
            "save_total_limit": self.save_total_limit,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "bf16": self.bf16,
            "seed": self.seed,
            "max_seq_length": self.max_seq_length,
            "preprocessing_num_proc": self.preprocessing_num_proc,
            "add_special_tokens": self.add_special_tokens,
            "special_tokens": list(self.special_tokens),
            "trust_remote_code": self.trust_remote_code,
            "gradient_checkpointing": self.gradient_checkpointing,
            "report_to": list(self.report_to),
            "resume_from_checkpoint": self.resume_from_checkpoint,
        }


def _require_nonempty_string(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if "/path/to/" in value or value.startswith("<"):
        raise ValueError(f"{name} still contains a v1 placeholder")


def _reject_parameter_efficient_tuning(value: Any, location: str = "config") -> None:
    """Reject LoRA/PEFT-style configuration before importing training libraries."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = str(key).lower()
            if any(term in normalized_key for term in _FORBIDDEN_TUNING_TERMS):
                raise ValueError(
                    f"{location}.{key} requests LoRA/PEFT or another adapter method; "
                    "G-Frame v2 SFT is full-parameter only"
                )
            _reject_parameter_efficient_tuning(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_parameter_efficient_tuning(child, f"{location}[{index}]")
    elif isinstance(value, str) and value.strip().lower() in _FORBIDDEN_TUNING_TERMS:
        raise ValueError(
            f"{location} requests LoRA/PEFT or another adapter method; "
            "G-Frame v2 SFT is full-parameter only"
        )


def _as_string_tuple(name: str, value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    result = tuple(value)
    if any(not isinstance(item, str) for item in result):
        raise ValueError(f"{name} must contain strings")
    return result


def _resolve_path(value: str, base_directory: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return str(path.resolve())


def _resolve_model_reference(value: str, base_directory: Path) -> str:
    """Resolve an explicit local model path without corrupting Hugging Face IDs."""

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    if value.startswith(".") or (base_directory / candidate).exists():
        return str((base_directory / candidate).resolve())
    return value


def _normalize_dataset_format(value: Any, dataset_path: str, uses_legacy_plural: bool) -> str:
    if value is None:
        if uses_legacy_plural or not dataset_path.lower().endswith(".jsonl"):
            return DATASET_FORMAT_TOKENIZED_DISK
        return DATASET_FORMAT_JSONL_MESSAGES
    if not isinstance(value, str):
        raise ValueError("dataset_format must be a string")
    normalized = _FORMAT_ALIASES.get(value.strip().lower())
    if normalized is None:
        supported = ", ".join(sorted(SUPPORTED_DATASET_FORMATS))
        raise ValueError(f"dataset_format must be one of: {supported}")
    return normalized


def load_sft_config(config_path: str | Path) -> SFTConfig:
    """Load a JSON job description without importing ML or telemetry packages."""

    source = Path(config_path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"SFT config does not exist: {source}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"SFT config is not valid JSON: {source}: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("SFT config must be a JSON object")

    _reject_parameter_efficient_tuning(payload)
    raw = dict(payload)
    uses_legacy_plural = "dataset_paths" in raw
    legacy_paths = _as_string_tuple("dataset_paths", raw.pop("dataset_paths", None))
    explicit_dataset_path = raw.get("dataset_path")
    configured_path = explicit_dataset_path
    if configured_path is None and legacy_paths:
        configured_path = legacy_paths[0]
    if configured_path is None:
        raise ValueError("SFT config requires dataset_path or legacy dataset_paths")
    if not isinstance(configured_path, str):
        raise ValueError("dataset_path must be a string")

    raw["dataset_path"] = _resolve_path(configured_path, source.parent)
    resolved_paths = tuple(
        _resolve_path(path, source.parent) for path in (legacy_paths or (configured_path,))
    )
    if explicit_dataset_path is not None and legacy_paths and raw["dataset_path"] not in resolved_paths:
        raise ValueError("dataset_path must also appear in legacy dataset_paths when both are set")
    raw["dataset_paths"] = resolved_paths
    raw["dataset_format"] = _normalize_dataset_format(
        raw.get("dataset_format"), configured_path, uses_legacy_plural
    )
    if raw.get("eval_dataset_path") is not None:
        if not isinstance(raw["eval_dataset_path"], str):
            raise ValueError("eval_dataset_path must be a string")
        raw["eval_dataset_path"] = _resolve_path(raw["eval_dataset_path"], source.parent)
        raw["eval_dataset_format"] = _normalize_dataset_format(
            raw.get("eval_dataset_format"), raw["eval_dataset_path"], False
        )
    elif raw.get("eval_dataset_format") is not None:
        raise ValueError("eval_dataset_format requires eval_dataset_path")

    for required in ("model_path", "output_dir", "deepspeed_config"):
        if required not in raw:
            raise ValueError(f"SFT config requires {required}")
        if not isinstance(raw[required], str):
            raise ValueError(f"{required} must be a string")
    raw["model_path"] = _resolve_model_reference(raw["model_path"], source.parent)
    raw["output_dir"] = _resolve_path(raw["output_dir"], source.parent)
    raw["deepspeed_config"] = _resolve_path(raw["deepspeed_config"], source.parent)
    raw["special_tokens"] = _as_string_tuple("special_tokens", raw.get("special_tokens"))
    raw["report_to"] = _as_string_tuple("report_to", raw.get("report_to"))
    if raw.get("resume_from_checkpoint") is not None:
        if not isinstance(raw["resume_from_checkpoint"], str):
            raise ValueError("resume_from_checkpoint must be a string")
        raw["resume_from_checkpoint"] = _resolve_path(raw["resume_from_checkpoint"], source.parent)

    allowed_keys = set(SFTConfig.__dataclass_fields__)
    unknown_keys = sorted(set(raw) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"unsupported SFT config keys: {', '.join(unknown_keys)}")
    try:
        config = SFTConfig(**raw)
    except TypeError as error:
        raise ValueError(f"invalid SFT configuration: {error}") from error
    config.validate()
    return config


def load_deepspeed_config(path: str | Path) -> Dict[str, Any]:
    """Parse and check a DeepSpeed JSON file without importing DeepSpeed."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"DeepSpeed config does not exist: {source}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"DeepSpeed config is not valid JSON: {source}: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("DeepSpeed config must be a JSON object")
    _reject_parameter_efficient_tuning(payload, "deepspeed_config")
    zero = payload.get("zero_optimization")
    if not isinstance(zero, Mapping):
        raise ValueError("DeepSpeed config requires a zero_optimization object")
    stage = zero.get("stage")
    if not isinstance(stage, int) or stage not in {1, 2, 3}:
        raise ValueError("DeepSpeed zero_optimization.stage must be 1, 2, or 3")
    return payload


def _validate_message(message: Any, line_number: int, message_index: int) -> None:
    if not isinstance(message, Mapping):
        raise ValueError(f"line {line_number}, messages[{message_index}] must be an object")
    role = message.get("role")
    content = message.get("content")
    if role not in _ALLOWED_ROLES:
        allowed = ", ".join(sorted(_ALLOWED_ROLES))
        raise ValueError(
            f"line {line_number}, messages[{message_index}].role must be one of: {allowed}"
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"line {line_number}, messages[{message_index}].content must be non-empty")


def validate_messages_jsonl(path: str | Path) -> int:
    """Validate the v2 JSONL schema and return the number of training rows."""

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"v2 messages JSONL does not exist: {source}")
    count = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number} is not valid JSON: {error.msg}") from error
            if not isinstance(row, Mapping):
                raise ValueError(f"line {line_number} must be a JSON object")
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"line {line_number} must contain a non-empty messages list")
            for message_index, message in enumerate(messages):
                _validate_message(message, line_number, message_index)
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"line {line_number} must end with an assistant response")
            count += 1
    if count == 0:
        raise ValueError(f"v2 messages JSONL has no training rows: {source}")
    return count


def validate_job(config: SFTConfig, check_input: bool = False) -> Dict[str, Any]:
    """Run CPU-only checks used by CI and by the operator before a GPU job."""

    deepspeed_payload = load_deepspeed_config(config.deepspeed_config)
    summary: Dict[str, Any] = {
        "dataset_format": config.dataset_format,
        "dataset_paths": list(config.all_dataset_paths),
        "deepspeed_zero_stage": deepspeed_payload["zero_optimization"]["stage"],
        "full_parameter_sft": True,
        "input_checked": False,
        "eval_input_checked": False,
    }
    if check_input:
        if config.dataset_format == DATASET_FORMAT_JSONL_MESSAGES:
            counts = [validate_messages_jsonl(path) for path in config.all_dataset_paths]
            summary["training_rows"] = sum(counts)
        else:
            missing = [path for path in config.all_dataset_paths if not Path(path).is_dir()]
            if missing:
                raise ValueError("tokenized dataset directory does not exist: " + ", ".join(missing))
            summary["training_rows"] = "checked at runtime"
        summary["input_checked"] = True
        if config.eval_dataset_path is not None:
            if config.eval_dataset_format == DATASET_FORMAT_JSONL_MESSAGES:
                summary["evaluation_rows"] = validate_messages_jsonl(config.eval_dataset_path)
            elif not Path(config.eval_dataset_path).is_dir():
                raise ValueError(f"tokenized evaluation dataset directory does not exist: {config.eval_dataset_path}")
            else:
                summary["evaluation_rows"] = "checked at runtime"
            summary["eval_input_checked"] = True
        if config.resume_from_checkpoint is not None and not Path(config.resume_from_checkpoint).is_dir():
            raise ValueError(f"resume checkpoint directory does not exist: {config.resume_from_checkpoint}")
    return summary


def _resume_log_history_count(resume_from_checkpoint: str | None) -> int:
    """Return the persisted history length so resumed metrics stay stage-local."""

    if resume_from_checkpoint is None:
        return 0
    state_path = Path(resume_from_checkpoint) / "trainer_state.json"
    if not state_path.is_file():
        return 0
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"resume trainer state is not valid JSON: {state_path}") from error
    if not isinstance(state, Mapping):
        raise ValueError(f"resume trainer state must be a JSON object: {state_path}")
    history = state.get("log_history", [])
    if not isinstance(history, list):
        raise ValueError(f"resume trainer state log_history must be a list: {state_path}")
    return len(history)


def _checkpoint_metrics_from_current_run(
    output_dir: Path,
    log_history: Sequence[Mapping[str, Any]],
    *,
    resume_history_count: int,
    final_eval_metrics: Mapping[str, Any],
    global_step: int,
) -> Dict[str, Dict[str, Any]]:
    """Associate only this invocation's held-out metrics with its checkpoints."""

    if resume_history_count < 0:
        raise ValueError("resume_history_count cannot be negative")
    output_dir = output_dir.expanduser().resolve()
    events = list(log_history)
    current_events = events[resume_history_count:]
    checkpoint_metrics: Dict[str, Dict[str, Any]] = {}
    for event in current_events:
        if "eval_loss" not in event:
            continue
        step = int(event.get("step", global_step))
        checkpoint = output_dir / f"checkpoint-{step}"
        key = str((checkpoint if checkpoint.is_dir() else output_dir).resolve())
        checkpoint_metrics[key] = {
            "step": step,
            "eval_loss": float(event["eval_loss"]),
        }
    if "eval_loss" in final_eval_metrics:
        checkpoint_metrics[str(output_dir)] = {
            "step": int(global_step),
            "eval_loss": float(final_eval_metrics["eval_loss"]),
        }
    return checkpoint_metrics


def _lazy_training_dependencies() -> Tuple[Any, Any, Any, Any, Any, Any]:
    """Import optional GPU-training dependencies only after --run was requested."""

    try:
        import deepspeed
        import torch
        from datasets import concatenate_datasets, load_dataset, load_from_disk
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError(
            "Training requires deepspeed, torch, datasets, and transformers. "
            "Install the v2 training environment before using --run."
        ) from error
    return (
        torch,
        load_dataset,
        load_from_disk,
        concatenate_datasets,
        (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        ),
        deepspeed,
    )


def _build_jsonl_dataset(config: SFTConfig, tokenizer: Any, load_dataset: Any) -> Any:
    dataset = load_dataset("json", data_files=list(config.all_dataset_paths), split="train")

    def format_and_tokenize(example: Mapping[str, Any]) -> Dict[str, List[int]]:
        messages = example.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("every JSONL row must contain a non-empty messages list")
        try:
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
        except AttributeError as error:
            raise RuntimeError(
                "the selected tokenizer has no chat template; choose a chat model or "
                "pre-tokenize the data with preprocess_sft_data.py"
            ) from error
        full_tokenized = tokenizer(
            full_text,
            max_length=config.max_seq_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )
        prompt_tokenized = tokenizer(
            prompt_text,
            max_length=config.max_seq_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )
        input_ids = list(full_tokenized["input_ids"])
        prompt_length = min(len(prompt_tokenized["input_ids"]), len(input_ids))
        labels = list(input_ids)
        labels[:prompt_length] = [-100] * prompt_length
        if all(token == -100 for token in labels):
            raise ValueError("a JSONL row was truncated before its assistant response")
        full_tokenized["labels"] = labels
        return full_tokenized

    tokenized = dataset.map(
        format_and_tokenize,
        remove_columns=list(dataset.column_names),
        num_proc=config.preprocessing_num_proc,
        desc="Formatting v2 SFT JSONL",
    )
    if len(tokenized) == 0:
        raise ValueError("no tokenized v2 SFT rows were produced")
    return tokenized


def _load_training_dataset(
    config: SFTConfig,
    tokenizer: Any,
    load_dataset: Any,
    load_from_disk: Any,
    concatenate_datasets: Any,
) -> Any:
    if config.dataset_format == DATASET_FORMAT_JSONL_MESSAGES:
        return _build_jsonl_dataset(config, tokenizer, load_dataset)

    datasets = [load_from_disk(path) for path in config.all_dataset_paths]
    dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    required_columns = {"input_ids", "labels"}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"legacy tokenized SFT dataset is missing columns: {missing}")
    return dataset.shuffle(seed=config.seed)


def _load_evaluation_dataset(
    config: SFTConfig,
    tokenizer: Any,
    load_dataset: Any,
    load_from_disk: Any,
    concatenate_datasets: Any,
) -> Any | None:
    """Load the held-out input with the same schema handling as training."""

    if config.eval_dataset_path is None:
        return None
    if config.eval_dataset_format is None:
        raise ValueError("eval_dataset_format is required when eval_dataset_path is set")
    evaluation_config = replace(
        config,
        dataset_path=config.eval_dataset_path,
        dataset_paths=(),
        dataset_format=config.eval_dataset_format,
    )
    return _load_training_dataset(
        evaluation_config, tokenizer, load_dataset, load_from_disk, concatenate_datasets
    )


def _special_tokens(config: SFTConfig) -> List[str]:
    tokens = list(config.special_tokens)
    if config.add_special_tokens:
        tokens.extend(["<think>", "</think>"])
    return list(dict.fromkeys(tokens))


def _assert_full_parameter_model(model: Any) -> int:
    """Reject adapter models and ensure every base-model parameter is trainable."""

    identity = f"{model.__class__.__module__}.{model.__class__.__name__}".lower()
    if any(term in identity for term in _FORBIDDEN_TUNING_TERMS):
        raise ValueError("adapter/PEFT model instances are not supported by full-parameter SFT")
    if getattr(model, "peft_config", None):
        raise ValueError("loaded model has an active PEFT configuration; full-parameter SFT is required")
    trainable_parameters = 0
    frozen_names: List[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = True
        if not parameter.requires_grad:
            frozen_names.append(name)
        else:
            trainable_parameters += parameter.numel()
    if frozen_names:
        preview = ", ".join(frozen_names[:3])
        raise RuntimeError(f"could not unfreeze all model parameters: {preview}")
    if trainable_parameters == 0:
        raise RuntimeError("model exposes no trainable parameters")
    return trainable_parameters


def run_training(config: SFTConfig) -> Dict[str, Any]:
    """Execute a DeepSpeed full-parameter SFT job after explicit operator consent."""

    validation = validate_job(config, check_input=True)
    (
        torch,
        load_dataset,
        load_from_disk,
        concatenate_datasets,
        transformers,
        _deepspeed,
    ) = _lazy_training_dependencies()
    (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    ) = transformers

    if not torch.cuda.is_available():
        raise RuntimeError(
            "no CUDA device is available. Run --validate for CPU-only checks; "
            "start --run only in the approved GPU environment."
        )

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        dtype=torch.bfloat16 if config.bf16 else None,
        trust_remote_code=config.trust_remote_code,
    )
    tokens = _special_tokens(config)
    if tokens:
        added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        if added:
            model.resize_token_embeddings(len(tokenizer))
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    trainable_parameters = _assert_full_parameter_model(model)

    dataset = _load_training_dataset(
        config, tokenizer, load_dataset, load_from_disk, concatenate_datasets
    )
    eval_dataset = _load_evaluation_dataset(
        config, tokenizer, load_dataset, load_from_disk, concatenate_datasets
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        deepspeed=config.deepspeed_config,
        run_name=config.run_name,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_steps=config.max_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        bf16=config.bf16,
        report_to=list(config.report_to),
        remove_unused_columns=False,
        seed=config.seed,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    resume_history_count = _resume_log_history_count(config.resume_from_checkpoint)
    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    metrics = dict(result.metrics)
    metrics["trainable_parameters"] = trainable_parameters
    metrics["dataset_rows"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    eval_metrics: Dict[str, Any] = {}
    if eval_dataset is not None:
        eval_metrics = dict(trainer.evaluate())
        eval_metrics["dataset_rows"] = len(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    checkpoint_metrics = _checkpoint_metrics_from_current_run(
        Path(config.output_dir),
        trainer.state.log_history,
        resume_history_count=resume_history_count,
        final_eval_metrics=eval_metrics,
        global_step=int(trainer.state.global_step),
    )
    (Path(config.output_dir) / "checkpoint_metrics.json").write_text(
        json.dumps(checkpoint_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trainer.save_state()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    return {
        "validation": validation,
        "metrics": metrics,
        "eval": eval_metrics,
        "checkpoint_metrics": checkpoint_metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="G-Frame v2 DeepSpeed full-parameter SFT runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to an SFT JSON configuration")
    parser.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration and DeepSpeed JSON without loading ML dependencies",
    )
    parser.add_argument(
        "--check-input",
        action="store_true",
        help="With --validate, also validate local JSONL rows or tokenized dataset paths",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Start training. Use only through the approved DeepSpeed GPU launcher.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.validate and not args.run:
        parser.error("choose --validate for CPU-only checks or --run to start training")
    if args.check_input and not (args.validate or args.run):
        parser.error("--check-input requires --validate or --run")

    config = load_sft_config(args.config)
    if args.validate:
        summary = validate_job(config, check_input=args.check_input)
        print(json.dumps(summary, indent=2, sort_keys=True))
    if args.run:
        result = run_training(config)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
