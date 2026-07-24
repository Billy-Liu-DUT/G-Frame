import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from llm_training.preprocess_sft_data import INPUT_FORMAT_V2_JSONL, collect_v2_jsonl_rows, main as preprocess_main
from llm_training.run_sft import (
    DATASET_FORMAT_TOKENIZED_DISK,
    _checkpoint_metrics_from_current_run as sft_checkpoint_metrics,
    _resume_log_history_count as sft_resume_history_count,
    load_sft_config,
    main as sft_main,
    validate_job,
)
from llm_training.run_pretraining import (
    _checkpoint_metrics_from_current_run as pretraining_checkpoint_metrics,
    _resume_log_history_count as pretraining_resume_history_count,
    load_config as load_pretraining_config,
    validate_job as validate_pretraining_job,
)


class TrainingEntrypointTests(unittest.TestCase):
    def _write_deepspeed_config(self, directory: Path) -> Path:
        path = directory / "deepspeed.json"
        path.write_text(
            json.dumps({"zero_optimization": {"stage": 2}, "bf16": {"enabled": True}}),
            encoding="utf-8",
        )
        return path

    def _write_v2_messages(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "messages.jsonl"
        path.write_text(
            json.dumps(
                {
                    "record_id": "record-1",
                    "messages": [
                        {"role": "system", "content": "You are a chemistry assistant."},
                        {"role": "user", "content": "State the unit of pressure."},
                        {"role": "assistant", "content": "The SI unit is pascal."},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_v2_jsonl_config_validates_without_ml_imports(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            deepspeed_path = self._write_deepspeed_config(directory)
            messages_path = self._write_v2_messages(directory)
            config_path = directory / "job.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
                        "dataset_path": str(messages_path),
                        "dataset_format": "jsonl_messages",
                        "output_dir": str(directory / "output"),
                        "deepspeed_config": str(deepspeed_path),
                        "save_strategy": "no",
                    }
                ),
                encoding="utf-8",
            )
            config = load_sft_config(config_path)
            summary = validate_job(config, check_input=True)
            self.assertEqual(summary["training_rows"], 1)
            self.assertEqual(summary["deepspeed_zero_stage"], 2)
            self.assertTrue(summary["full_parameter_sft"])
            self.assertEqual(sft_main(["--config", str(config_path), "--validate", "--check-input"]), 0)

    def test_sft_held_out_messages_are_validated_without_ml_imports(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            deepspeed_path = self._write_deepspeed_config(directory)
            messages_path = self._write_v2_messages(directory)
            eval_path = self._write_v2_messages(directory / "eval")
            config_path = directory / "job.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
                        "dataset_path": str(messages_path),
                        "dataset_format": "jsonl_messages",
                        "eval_dataset_path": str(eval_path),
                        "eval_dataset_format": "jsonl_messages",
                        "eval_strategy": "steps",
                        "eval_steps": 1,
                        "output_dir": str(directory / "output"),
                        "deepspeed_config": str(deepspeed_path),
                    }
                ),
                encoding="utf-8",
            )
            summary = validate_job(load_sft_config(config_path), check_input=True)
            self.assertEqual(summary["evaluation_rows"], 1)
            self.assertTrue(summary["eval_input_checked"])

    def test_pretraining_jsonl_text_config_validates_without_ml_imports(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            deepspeed_path = self._write_deepspeed_config(directory)
            train_path = directory / "pretrain.jsonl"
            eval_path = directory / "pretrain_eval.jsonl"
            train_path.write_text('{"text":"mock training text"}\n', encoding="utf-8")
            eval_path.write_text('{"text":"mock held-out text"}\n', encoding="utf-8")
            config_path = directory / "pretrain.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
                        "tokenizer_path": "Qwen/Qwen2.5-0.5B-Instruct",
                        "dataset_path": str(train_path),
                        "dataset_format": "jsonl_text",
                        "eval_dataset_path": str(eval_path),
                        "eval_dataset_format": "jsonl_text",
                        "eval_strategy": "steps",
                        "output_dir": str(directory / "output"),
                        "deepspeed_config": str(deepspeed_path),
                    }
                ),
                encoding="utf-8",
            )
            summary = validate_pretraining_job(load_pretraining_config(config_path), check_input=True)
            self.assertEqual(summary["training_rows"], 1)
            self.assertEqual(summary["eval_rows"], 1)

    def test_legacy_dataset_paths_remain_supported_and_lora_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            deepspeed_path = self._write_deepspeed_config(directory)
            legacy_path = directory / "legacy-tokenized"
            config_path = directory / "legacy.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
                        "dataset_paths": [str(legacy_path)],
                        "output_dir": str(directory / "output"),
                        "deepspeed_config": str(deepspeed_path),
                    }
                ),
                encoding="utf-8",
            )
            config = load_sft_config(config_path)
            self.assertEqual(config.dataset_format, DATASET_FORMAT_TOKENIZED_DISK)

            config_path.write_text(
                json.dumps(
                    {
                        "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
                        "dataset_path": str(legacy_path),
                        "output_dir": str(directory / "output"),
                        "deepspeed_config": str(deepspeed_path),
                        "tuning_method": "lora",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "LoRA/PEFT"):
                load_sft_config(config_path)

    def test_preprocess_validation_accepts_v2_jsonl_without_tokenizer(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            messages_path = self._write_v2_messages(directory)
            rows = collect_v2_jsonl_rows(messages_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                preprocess_main(
                    [
                        "--input",
                        str(messages_path),
                        "--input-format",
                        INPUT_FORMAT_V2_JSONL,
                        "--validate",
                    ]
                ),
                0,
            )

    def test_help_is_safe_without_training_dependencies(self):
        repository = Path(__file__).resolve().parents[1]
        for script in ("llm_training/run_sft.py", "llm_training/preprocess_sft_data.py"):
            completed = subprocess.run(
                [sys.executable, script, "--help"],
                cwd=repository,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("usage:", completed.stdout)

    def test_imports_do_not_load_training_dependencies(self):
        repository = Path(__file__).resolve().parents[1]
        guard_script = """
import builtins

blocked = {"torch", "datasets", "transformers", "deepspeed", "wandb"}
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise AssertionError("unexpected training import: " + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import llm_training.run_sft
import llm_training.preprocess_sft_data
"""
        completed = subprocess.run(
            [sys.executable, "-c", guard_script],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_resumed_checkpoint_metrics_exclude_prior_history_and_use_current_final_eval(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            resume_dir = directory / "resume"
            resume_dir.mkdir()
            resume_dir.joinpath("trainer_state.json").write_text(
                json.dumps({"log_history": [{"step": 1, "eval_loss": 9.0}]}),
                encoding="utf-8",
            )
            output_dir = directory / "output"
            (output_dir / "checkpoint-2").mkdir(parents=True)
            history = [
                {"step": 1, "eval_loss": 9.0},
                {"step": 2, "eval_loss": 0.3},
            ]
            for history_count, build_metrics in (
                (sft_resume_history_count, sft_checkpoint_metrics),
                (pretraining_resume_history_count, pretraining_checkpoint_metrics),
            ):
                metrics = build_metrics(
                    output_dir,
                    history,
                    resume_history_count=history_count(str(resume_dir)),
                    final_eval_metrics={"eval_loss": 0.2},
                    global_step=2,
                )
                self.assertEqual(
                    metrics[str((output_dir / "checkpoint-2").resolve())],
                    {"step": 2, "eval_loss": 0.3},
                )
                self.assertEqual(
                    metrics[str(output_dir.resolve())],
                    {"step": 2, "eval_loss": 0.2},
                )

                # A truncated current Trainer history cannot safely be linked to
                # the resumed stage; old evaluation events must remain excluded.
                truncated_metrics = build_metrics(
                    output_dir,
                    [{"step": 1, "eval_loss": 9.0}],
                    resume_history_count=history_count(str(resume_dir)),
                    final_eval_metrics={},
                    global_step=2,
                )
                self.assertEqual(truncated_metrics, {})


if __name__ == "__main__":
    unittest.main()
