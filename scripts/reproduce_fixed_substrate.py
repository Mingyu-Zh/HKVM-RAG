#!/usr/bin/env python3
"""Run the fixed-substrate key-space comparison.

This public entry point keeps the reviewer-facing command aligned with the
paper terminology. It delegates execution to the HKVM runner while avoiding the
historical phase names used during development.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hkvm_mvp.runner import main as runner_main


CONFIGS = {
    "2wiki": ROOT / "configs" / "fixed_substrate_2wiki.yaml",
    "musique": ROOT / "configs" / "fixed_substrate_musique.yaml",
    "hotpotqa": ROOT / "configs" / "fixed_substrate_hotpotqa.yaml",
    "hotpotqa_structured": ROOT / "configs" / "fixed_substrate_hotpotqa_structured.yaml",
    "hotpotqa_dense": ROOT / "configs" / "fixed_substrate_hotpotqa_dense.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the fixed-substrate HKVM-RAG comparison.")
    parser.add_argument(
        "--setting",
        choices=sorted(CONFIGS),
        default="2wiki",
        help="Paper-facing evaluation setting to run.",
    )
    parser.add_argument("--datasets", default=None, help="Optional comma-separated dataset override.")
    parser.add_argument("--methods", default=None, help="Optional comma-separated method override.")
    parser.add_argument("--runs", type=int, default=None, help="Optional seed-count override.")
    parser.add_argument("--limit", type=int, default=None, help="Optional example limit for smoke tests.")
    parser.add_argument("--output_dir", default=None, help="Output directory for reproduced runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.setting]
    argv = ["hkvm-runner", "--config", str(config)]
    for key in ("datasets", "methods", "runs", "limit", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            argv.extend([f"--{key}", str(value)])
    old_argv = sys.argv
    try:
        sys.argv = argv
        runner_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
