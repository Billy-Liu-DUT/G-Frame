#!/usr/bin/env python3
"""Validate or run the local pretrain -> SFT stage handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from g_frame.training_schedule import MockTrainingCoordinator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coordinate the G-Frame v2 mock training schedule")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument(
        "--stage-timeout-seconds",
        type=int,
        default=270,
        help="Hard limit for each individual GPU stage; must not exceed 300 seconds.",
    )
    parser.add_argument("--run", action="store_true", help="Run the three DeepSpeed stages after CPU validation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    coordinator = MockTrainingCoordinator(
        args.output_dir,
        args.model_path,
        repo_root=REPOSITORY_ROOT,
        gpu_index=args.gpu_index,
        stage_timeout_s=args.stage_timeout_seconds,
    )
    manifest = coordinator.run() if args.run else coordinator.validate()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
