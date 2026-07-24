#!/usr/bin/env python3
"""Run the complete v2 data-flow handoff with generated local fixtures."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from g_frame.pipeline import run_mock_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local G-Frame v2 mock pipeline")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-path",
        type=Path,
        help="Optional caller-owned JSONL source input; omitted to generate a run-local fixture.",
    )
    parser.add_argument("--model", default="mock-pipeline", help="Trace label passed to the deterministic client")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = asyncio.run(
        run_mock_pipeline(args.output_dir, source_path=args.source_path, model=args.model)
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
