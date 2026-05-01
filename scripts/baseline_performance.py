from __future__ import annotations

import argparse
import os
from pathlib import Path
from pprint import pprint

from loguru import logger

from src.experiments.baseline_performance import baseline_performance


def _repo_root() -> Path:
    # scripts/baseline_performance.py → repo root
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline_performance suite (committed experiment YAMLs).")
    parser.add_argument(
        "--configs-only",
        action="store_true",
        help="Only write resolved YAMLs under results/baseline_performance/<stamp>/configs; do not run experiments.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results") / "baseline_performance",
        help="Directory under which timestamped runs are stored (relative to repo root unless absolute).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = _repo_root()
    os.chdir(root)
    logger.info("CWD set to repo root: {}", str(root))

    results = baseline_performance(
        results_root=args.results_root,
        seed=args.seed,
        run=not args.configs_only,
    )
    pprint(results)


if __name__ == "__main__":
    main()
