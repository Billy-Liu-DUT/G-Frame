from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from g_frame.training_schedule import (
    BoundedTrainingJudge,
    CheckpointScore,
    MockTrainingCoordinator,
    StageSelection,
    select_checkpoint,
)


class TrainingScheduleTests(unittest.TestCase):
    def test_checkpoint_selection_uses_minimum_held_out_loss_and_prefers_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "stage"
            checkpoint_one = output_dir / "checkpoint-1"
            checkpoint_two = output_dir / "checkpoint-2"
            checkpoint_one.mkdir(parents=True)
            checkpoint_two.mkdir()
            (output_dir / "checkpoint_metrics.json").write_text(
                json.dumps(
                    {
                        str(checkpoint_one.resolve()): {"step": 1, "eval_loss": 1.4},
                        str(checkpoint_two.resolve()): {"step": 2, "eval_loss": 0.9},
                        str(output_dir.resolve()): {"step": 2, "eval_loss": 0.8},
                    }
                ),
                encoding="utf-8",
            )
            selection = select_checkpoint("stage", output_dir, require_checkpoint=True)
            self.assertEqual(Path(selection.selected.path).name, "checkpoint-2")
            self.assertEqual(selection.selected.eval_loss, 0.9)

    def test_judge_only_proposes_values_inside_hard_bounds(self):
        judge = BoundedTrainingJudge(min_learning_rate=1e-5, max_learning_rate=1e-4)
        advice = judge.propose(
            selected=CheckpointScore(path="checkpoint-1", step=1, eval_loss=2.0),
            learning_rate=1e-4,
            data_mix={"general": 4, "domain": 6},
        )
        self.assertTrue(advice["selection_is_external"])
        self.assertAlmostEqual(advice["next_parameters"]["learning_rate"], 9e-5)
        self.assertEqual(advice["next_parameters"]["data_mix"], {"general": 0.4, "domain": 0.6})

    def test_checkpoint_selection_rejects_an_unevaluated_saved_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "stage"
            (output_dir / "checkpoint-1").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "held-out evaluation is missing"):
                select_checkpoint("stage", output_dir, require_checkpoint=True)

    def test_validate_generates_mock_inputs_and_all_three_configs_without_gpu(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            coordinator = MockTrainingCoordinator(
                Path(temporary_directory) / "run",
                "Qwen/Qwen2.5-0.5B-Instruct",
                repo_root=repository,
            )
            manifest = coordinator.validate()
            self.assertEqual(manifest["mode"], "validate")
            self.assertEqual(set(manifest["validation"]), {"pretrain", "sft_stage1", "sft_stage2"})
            self.assertTrue((coordinator.output_dir / "pretrain_train.jsonl").is_file())
            self.assertTrue((coordinator.output_dir / "sft_eval.jsonl").is_file())
            self.assertTrue((coordinator.output_dir / "stage_configs" / "pretrain.json").is_file())
            self.assertTrue((coordinator.output_dir / "stage_configs" / "sft_stage1.json").is_file())
            self.assertTrue((coordinator.output_dir / "stage_configs" / "sft_stage2.json").is_file())

    def test_schedule_script_validates_with_only_src_on_pythonpath(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "run"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(repository / "src")
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_mock_training_schedule.py",
                    "--output-dir",
                    str(output_dir),
                    "--model-path",
                    "mock-model",
                    "--gpu-index",
                    "0",
                ],
                cwd=repository,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output_dir / "schedule_manifest.json").is_file())

    def test_selected_pretrain_checkpoint_and_bounded_mix_drive_stage_one_input(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            coordinator = MockTrainingCoordinator(
                Path(temporary_directory) / "run",
                "Qwen/Qwen2.5-0.5B-Instruct",
                repo_root=repository,
            )
            pretrain_checkpoint = coordinator.output_dir / "pretrain" / "checkpoint-7"
            stage1_checkpoint = coordinator.output_dir / "sft_stage1" / "checkpoint-3"
            stage2_checkpoint = coordinator.output_dir / "sft_stage2" / "checkpoint-5"
            for path in (pretrain_checkpoint, stage1_checkpoint, stage2_checkpoint):
                path.mkdir(parents=True)
            selections = [
                StageSelection(
                    "pretrain",
                    [CheckpointScore(str(pretrain_checkpoint), 7, 0.7)],
                    CheckpointScore(str(pretrain_checkpoint), 7, 0.7),
                ),
                StageSelection(
                    "sft_stage1",
                    [CheckpointScore(str(stage1_checkpoint), 3, 0.6)],
                    CheckpointScore(str(stage1_checkpoint), 3, 0.6),
                ),
                StageSelection(
                    "sft_stage2",
                    [CheckpointScore(str(stage2_checkpoint), 5, 0.5)],
                    CheckpointScore(str(stage2_checkpoint), 5, 0.5),
                ),
            ]
            with patch("g_frame.training_schedule.select_checkpoint", side_effect=selections), patch.object(
                coordinator,
                "_run_stage",
                return_value={"command": ["mock"], "returncode": 0, "output": ""},
            ):
                manifest = coordinator.run()

            stage1_config = json.loads(
                (coordinator.output_dir / "stage_configs" / "sft_stage1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stage1_config["model_path"], str(pretrain_checkpoint))
            mixed_input = Path(stage1_config["dataset_path"])
            self.assertTrue(mixed_input.is_file())
            rows = [json.loads(line) for line in mixed_input.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(Counter(row["data_mix_source"] for row in rows), {"general": 4, "domain": 6})
            self.assertEqual(manifest["stage_input_mixes"]["sft_stage1"]["weights"], {"general": 0.4, "domain": 0.6})
            self.assertEqual(manifest["validation"]["sft_stage1"]["dataset_paths"], [str(mixed_input)])


if __name__ == "__main__":
    unittest.main()
