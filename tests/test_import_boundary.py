import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class ImportBoundaryTests(unittest.TestCase):
    def test_core_modules_do_not_import_gpu_or_service_dependencies(self):
        blocked = {"torch", "deepspeed", "vllm", "openai", "transformers"}
        source_root = Path(__file__).resolve().parents[1] / "src"
        script = textwrap.dedent(
            """
            import builtins
            import importlib

            blocked = {"torch", "deepspeed", "vllm", "openai", "transformers"}
            original_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name.split(".")[0] in blocked:
                    raise AssertionError(f"unexpected heavy import: {name}")
                return original_import(name, *args, **kwargs)

            builtins.__import__ = guarded_import
            for module in (
                "g_frame.schemas",
                "g_frame.augmentation",
                "g_frame.prompts",
                "g_frame.telemetry",
                "g_frame.actions",
                "g_frame.decision",
                "g_frame.team_game",
                "g_frame.data",
                "g_frame.sft",
                "g_frame.orchestration",
                "g_frame.cli",
            ):
                importlib.import_module(module)
            """
        )
        environment = os.environ.copy()
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(source_root) + (os.pathsep + existing if existing else "")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
