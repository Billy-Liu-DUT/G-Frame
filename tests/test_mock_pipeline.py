import json
from pathlib import Path
import tempfile
import unittest

from g_frame.pipeline import run_mock_pipeline
from llm_training.run_pretraining import validate_jsonl_text
from llm_training.run_sft import validate_messages_jsonl


class MockPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_stage_action_executes_and_writes_a_typed_handoff(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "mock-run"
            manifest = await run_mock_pipeline(output_dir)

            self.assertEqual(
                [item["status"] for item in manifest["executions"]],
                ["executed"] * 7,
            )
            self.assertEqual(
                manifest["stage_handlers"],
                [
                    "data_clean",
                    "data_augment",
                    "cot_distill",
                    "cot_synth",
                    "qa_synth",
                    "model_train",
                    "system_health_check",
                ],
            )
            self.assertEqual(manifest["counts"]["augmentation"], 6)
            self.assertEqual(manifest["counts"]["qa_records"], 6)
            self.assertGreaterEqual(manifest["counts"]["sft_stage1"], 1)
            self.assertEqual(manifest["counts"]["sft_stage2"], 1)
            self.assertEqual(manifest["counts"]["sft_eval"], 1)
            self.assertIn("data_clean", manifest["model_request_roles"])
            self.assertIn("cot_distill", manifest["model_request_roles"])
            self.assertIn("cot_synth", manifest["model_request_roles"])
            self.assertEqual(
                len([role for role in manifest["model_request_roles"] if role.startswith("augment_")]),
                6,
            )

            styles = {
                row["style"]
                for row in (json.loads(line) for line in (output_dir / "augmentation.jsonl").read_text(encoding="utf-8").splitlines())
            }
            self.assertEqual(len(styles), 6)
            self.assertEqual(validate_messages_jsonl(output_dir / "sft_stage1.jsonl"), 5)
            self.assertEqual(validate_messages_jsonl(output_dir / "sft_stage2.jsonl"), 1)
            self.assertEqual(validate_messages_jsonl(output_dir / "sft_eval.jsonl"), 1)
            self.assertEqual(validate_jsonl_text(output_dir / "pretrain_train.jsonl"), 7)
            self.assertEqual(validate_jsonl_text(output_dir / "pretrain_eval.jsonl"), 1)

            saved_manifest = json.loads((output_dir / "pipeline_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["mode"], "mock")
            self.assertTrue((output_dir / "system_health.json").is_file())


if __name__ == "__main__":
    unittest.main()
