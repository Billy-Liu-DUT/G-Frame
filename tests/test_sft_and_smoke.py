import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from g_frame.cli import _offline_smoke
from g_frame.sft import SFTJobConfig


class SFTAndSmokeTests(unittest.TestCase):
    def test_sft_planner_rejects_lora_and_never_launches_training(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "job.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
                        "dataset_path": "runs/live-smoke/sft_messages.jsonl",
                        "output_dir": "runs/live-smoke/checkpoint",
                        "deepspeed_config": "llm_training/ds_config_sft_smoke.json",
                        "max_steps": 3,
                    }
                ),
                encoding="utf-8",
            )
            config = SFTJobConfig.from_json_file(config_path)
            self.assertIn("deepspeed", config.deepspeed_command())
            config_path.write_text('{"lora": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "LoRA"):
                SFTJobConfig.from_json_file(config_path)

    def test_sft_planner_accepts_the_complete_gpu_smoke_config(self):
        repository = Path(__file__).resolve().parents[1]
        config = SFTJobConfig.from_json_file(repository / "configs" / "gpu_smoke_sft.json")
        self.assertEqual(config.to_dict()["dataset_format"], "jsonl_messages")
        self.assertIn("--config", config.deepspeed_command("configs/gpu_smoke_sft.json"))

    def test_offline_smoke_writes_traceable_artifacts_without_gpu(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "smoke"
            asyncio.run(_offline_smoke(output_path))
            manifest = json.loads((output_path / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["runtime_state"]["concurrency"], 2)
            self.assertEqual(manifest["worker_limiter"]["limit"], 2)
            self.assertEqual(manifest["workflow"][-1], "sft_serialization")
            self.assertEqual(manifest["queued_stages"][0]["action"], "qa_synth")
            self.assertTrue((output_path / "training_records.jsonl").exists())
            self.assertTrue((output_path / "sft_messages.jsonl").exists())
            roles = [item["role"] for item in manifest["record"]["agent_trace"]]
            self.assertEqual(roles, ["Teacher", "Student", "Reviewer", "Rectifier", "Reviewer", "Judger"])
