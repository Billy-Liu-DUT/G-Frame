"""CPU-safe DeepSpeed full-parameter SFT planning utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


@dataclass(frozen=True)
class SFTJobConfig:
    model_path: str
    dataset_path: str
    output_dir: str
    deepspeed_config: str
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = -1
    bf16: bool = True
    seed: int = 42
    extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_json_file(cls, path: Path) -> "SFTJobConfig":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("SFT config must be a JSON object")
        if _contains_forbidden_tuning(payload):
            raise ValueError("v2 smoke uses full-parameter SFT; LoRA and PEFT are not supported")
        known_fields = set(cls.__dataclass_fields__) - {"extra"}
        options = {key: value for key, value in payload.items() if key not in known_fields}
        config = cls(
            **{key: value for key, value in payload.items() if key in known_fields},
            extra=options,
        )
        config.validate()
        return config

    def validate(self) -> None:
        for name in ("model_path", "dataset_path", "output_dir", "deepspeed_config"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "<" in value or "/path/to/" in value:
                raise ValueError(f"{name} must be configured with a real path or model identifier")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.per_device_train_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch size and gradient accumulation must be at least 1")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or a positive integer")

    def to_dict(self) -> Dict[str, Any]:
        value = {
            "model_path": self.model_path,
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "deepspeed_config": self.deepspeed_config,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
            "bf16": self.bf16,
            "seed": self.seed,
        }
        value.update(self.extra)
        return value

    def deepspeed_command(
        self,
        config_path: str = "configs/smoke_sft.json",
        entrypoint: str = "llm_training/run_sft.py",
        gpu_count: int = 1,
    ) -> List[str]:
        if gpu_count < 1:
            raise ValueError("gpu_count must be at least 1")
        return ["deepspeed", f"--num_gpus={gpu_count}", entrypoint, "--config", config_path]


def _contains_forbidden_tuning(value: Any) -> bool:
    forbidden = ("lora", "qlora", "peft", "adapter", "prefix_tuning")
    if isinstance(value, Mapping):
        return any(
            any(term in str(key).lower() for term in forbidden) or _contains_forbidden_tuning(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_tuning(item) for item in value)
    return isinstance(value, str) and value.strip().lower() in forbidden
