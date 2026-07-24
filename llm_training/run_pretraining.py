"""Configuration-driven full-parameter continued pre-training.

The runner accepts either a legacy Hugging Face disk dataset or a small JSONL
``text`` fixture.  It never starts a job on import: ``--validate`` is CPU-safe
and ``--run`` is the explicit DeepSpeed training path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


DATASET_FORMAT_JSONL_TEXT = "jsonl_text"
DATASET_FORMAT_TOKENIZED_DISK = "tokenized_disk"
SUPPORTED_DATASET_FORMATS = frozenset({DATASET_FORMAT_JSONL_TEXT, DATASET_FORMAT_TOKENIZED_DISK})
_FORBIDDEN_TUNING_TERMS = ("lora", "qlora", "peft", "adapter", "prefix_tuning")


def _reject_parameter_efficient_tuning(value: Any, location: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if any(term in str(key).lower() for term in _FORBIDDEN_TUNING_TERMS):
                raise ValueError(f"{location}.{key} requests an adapter method; full-parameter training is required")
            _reject_parameter_efficient_tuning(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_parameter_efficient_tuning(child, f"{location}[{index}]")
    elif isinstance(value, str) and value.strip().lower() in _FORBIDDEN_TUNING_TERMS:
        raise ValueError(f"{location} requests an adapter method; full-parameter training is required")


def _require_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or value.startswith("<") or "/path/to/" in value:
        raise ValueError(f"{name} must be a non-placeholder string")
    return value.strip()


def _resolve_path(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def _resolve_model_reference(value: str, base: Path) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    if value.startswith(".") or (base / candidate).exists():
        return str((base / candidate).resolve())
    return value


def _normalize_dataset_format(value: Any, dataset_path: str) -> str:
    if value is None:
        return DATASET_FORMAT_JSONL_TEXT if dataset_path.lower().endswith(".jsonl") else DATASET_FORMAT_TOKENIZED_DISK
    if not isinstance(value, str) or value not in SUPPORTED_DATASET_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_DATASET_FORMATS))
        raise ValueError(f"dataset_format must be one of: {supported}")
    return value


def load_config(path: Path) -> Dict[str, Any]:
    """Load and normalize a pre-training job without importing ML libraries."""

    source = path.expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"pre-training config does not exist: {source}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"pre-training config is not valid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("pre-training config must be a JSON object")
    _reject_parameter_efficient_tuning(payload)

    required = {"model_path", "tokenizer_path", "dataset_path", "output_dir", "deepspeed_config"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"pre-training config is missing: {', '.join(sorted(missing))}")
    for field in required:
        _require_string(field, payload[field])

    config = dict(payload)
    config["dataset_format"] = _normalize_dataset_format(config.get("dataset_format"), config["dataset_path"])
    config["dataset_path"] = _resolve_path(config["dataset_path"], source.parent)
    config["output_dir"] = _resolve_path(config["output_dir"], source.parent)
    config["deepspeed_config"] = _resolve_path(config["deepspeed_config"], source.parent)
    config["model_path"] = _resolve_model_reference(config["model_path"], source.parent)
    config["tokenizer_path"] = _resolve_model_reference(config["tokenizer_path"], source.parent)
    if config.get("eval_dataset_path") is not None:
        _require_string("eval_dataset_path", config["eval_dataset_path"])
        config["eval_dataset_path"] = _resolve_path(config["eval_dataset_path"], source.parent)
    config.setdefault("eval_dataset_format", config["dataset_format"])
    if config["eval_dataset_format"] not in SUPPORTED_DATASET_FORMATS:
        raise ValueError("eval_dataset_format must be a supported dataset format")
    config.setdefault("eval_strategy", "no")
    if config["eval_strategy"] not in {"no", "steps", "epoch"}:
        raise ValueError("eval_strategy must be 'no', 'steps', or 'epoch'")
    if config["eval_strategy"] != "no" and not config.get("eval_dataset_path"):
        raise ValueError("eval_dataset_path is required when eval_strategy is enabled")
    return config


def validate_jsonl_text(path: str | Path) -> int:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"JSONL text input does not exist: {source}")
    rows = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number} is not valid JSON: {error.msg}") from error
            if not isinstance(row, Mapping) or not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"line {line_number} must contain a non-empty text field")
            rows += 1
    if rows == 0:
        raise ValueError(f"JSONL text input has no rows: {source}")
    return rows


def validate_job(config: Mapping[str, Any], check_input: bool = False) -> Dict[str, Any]:
    try:
        deepspeed_payload = json.loads(Path(config["deepspeed_config"]).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"DeepSpeed config does not exist: {config['deepspeed_config']}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"DeepSpeed config is not valid JSON: {error.msg}") from error
    if not isinstance(deepspeed_payload, Mapping) or not isinstance(deepspeed_payload.get("zero_optimization"), Mapping):
        raise ValueError("DeepSpeed config requires zero_optimization")
    summary: Dict[str, Any] = {
        "dataset_format": config["dataset_format"],
        "full_parameter_pretraining": True,
        "input_checked": False,
        "eval_input_checked": False,
    }
    if check_input:
        if config["dataset_format"] == DATASET_FORMAT_JSONL_TEXT:
            summary["training_rows"] = validate_jsonl_text(config["dataset_path"])
        elif not Path(config["dataset_path"]).is_dir():
            raise ValueError("tokenized pre-training dataset directory does not exist")
        else:
            summary["training_rows"] = "checked at runtime"
        summary["input_checked"] = True
        if config.get("eval_dataset_path"):
            if config["eval_dataset_format"] == DATASET_FORMAT_JSONL_TEXT:
                summary["eval_rows"] = validate_jsonl_text(config["eval_dataset_path"])
            elif not Path(config["eval_dataset_path"]).is_dir():
                raise ValueError("tokenized evaluation dataset directory does not exist")
            else:
                summary["eval_rows"] = "checked at runtime"
            summary["eval_input_checked"] = True
    return summary


def _resume_log_history_count(resume_from_checkpoint: object) -> int:
    """Return the persisted history length so resumed metrics stay stage-local."""

    if resume_from_checkpoint is None:
        return 0
    if not isinstance(resume_from_checkpoint, str) or not resume_from_checkpoint.strip():
        raise ValueError("resume_from_checkpoint must be a non-empty string when provided")
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
        checkpoint_metrics[key] = {"step": step, "eval_loss": float(event["eval_loss"])}
    if "eval_loss" in final_eval_metrics:
        checkpoint_metrics[str(output_dir)] = {
            "step": int(global_step),
            "eval_loss": float(final_eval_metrics["eval_loss"]),
        }
    return checkpoint_metrics


def _lazy_dependencies() -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from datasets import load_dataset, load_from_disk
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError("pre-training requires torch, datasets, and transformers") from error
    return (
        torch,
        load_dataset,
        load_from_disk,
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        (Trainer, TrainingArguments, set_seed),
    )


def _load_dataset(config: Mapping[str, Any], prefix: str, tokenizer: Any, load_dataset: Any, load_from_disk: Any) -> Any:
    path = config[f"{prefix}dataset_path"] if prefix else config["dataset_path"]
    dataset_format = config[f"{prefix}dataset_format"] if prefix else config["dataset_format"]
    if dataset_format == DATASET_FORMAT_JSONL_TEXT:
        dataset = load_dataset("json", data_files=path, split="train")
    else:
        dataset = load_from_disk(path)
    if "input_ids" in dataset.column_names:
        return dataset
    if "text" not in dataset.column_names:
        raise ValueError("pre-training input must contain text or input_ids")
    max_seq_length = int(config.get("max_seq_length", 2048))

    def tokenize(examples: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
        return tokenizer(examples["text"], truncation=True, max_length=max_seq_length)

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=list(dataset.column_names),
        num_proc=int(config.get("preprocessing_num_proc", config.get("num_proc", 1))),
        desc="Tokenizing pre-training input",
    )


def run_pretraining(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Execute a full-parameter DeepSpeed pre-training job after explicit consent."""

    validation = validate_job(config, check_input=True)
    (
        torch,
        load_dataset,
        load_from_disk,
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        trainer_components,
    ) = _lazy_dependencies()
    Trainer, TrainingArguments, set_seed = trainer_components
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device is available; use --validate for CPU-only checks")

    set_seed(int(config.get("seed", 42)))
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], trust_remote_code=bool(config.get("trust_remote_code", False)))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        dtype=torch.bfloat16 if bool(config.get("bf16", True)) else None,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad = True
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if not trainable_parameters:
        raise RuntimeError("model exposes no trainable parameters")

    train_dataset = _load_dataset(config, "", tokenizer, load_dataset, load_from_disk)
    eval_dataset = None
    if config.get("eval_dataset_path"):
        eval_config = dict(config)
        eval_config["dataset_path"] = config["eval_dataset_path"]
        eval_config["dataset_format"] = config["eval_dataset_format"]
        eval_dataset = _load_dataset(eval_config, "", tokenizer, load_dataset, load_from_disk)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        run_name=config.get("run_name", "g-frame-pretraining"),
        num_train_epochs=float(config.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        learning_rate=float(config.get("learning_rate", 3e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        warmup_ratio=float(config.get("warmup_ratio", 0.0)),
        logging_steps=int(config.get("logging_steps", 1)),
        save_steps=int(config.get("save_steps", 1)),
        save_strategy=config.get("save_strategy", "no"),
        save_total_limit=int(config.get("save_total_limit", 1)),
        eval_strategy=config.get("eval_strategy", "no"),
        eval_steps=int(config.get("eval_steps", 1)),
        max_steps=int(config.get("max_steps", -1)),
        bf16=bool(config.get("bf16", True)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        deepspeed=config["deepspeed_config"],
        report_to=[],
        remove_unused_columns=False,
        seed=int(config.get("seed", 42)),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    resume_from_checkpoint = config.get("resume_from_checkpoint")
    resume_history_count = _resume_log_history_count(resume_from_checkpoint)
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    train_metrics = dict(train_result.metrics)
    train_metrics.update({"trainable_parameters": trainable_parameters, "training_rows": len(train_dataset)})
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    eval_metrics: Dict[str, Any] = {}
    if eval_dataset is not None:
        eval_metrics = dict(trainer.evaluate())
        eval_metrics["evaluation_rows"] = len(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    checkpoint_metrics = _checkpoint_metrics_from_current_run(
        Path(config["output_dir"]),
        trainer.state.log_history,
        resume_history_count=resume_history_count,
        final_eval_metrics=eval_metrics,
        global_step=int(trainer.state.global_step),
    )
    (Path(config["output_dir"]) / "checkpoint_metrics.json").write_text(
        json.dumps(checkpoint_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trainer.save_state()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    return {
        "validation": validation,
        "train": train_metrics,
        "eval": eval_metrics,
        "checkpoint_metrics": checkpoint_metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G-Frame full-parameter pre-training runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--validate", action="store_true", help="Validate config and inputs without loading ML dependencies")
    parser.add_argument("--check-input", action="store_true", help="Validate local input rows with --validate")
    parser.add_argument("--run", action="store_true", help="Run explicit full-parameter training")
    parser.add_argument("--dry-run", action="store_true", help="Legacy alias for --validate")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate = args.validate or args.dry_run
    if not validate and not args.run:
        parser = build_parser()
        parser.error("choose --validate for CPU checks or --run to start training")
    config = load_config(args.config)
    if validate:
        print(json.dumps(validate_job(config, check_input=args.check_input), indent=2, sort_keys=True))
    if args.run:
        print(json.dumps(run_pretraining(config), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
