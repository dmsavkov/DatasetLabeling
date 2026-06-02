#!/usr/bin/env python
"""Pairwise Cohen's kappa, McNemar, and per-class PR plots grouped by dataset + tier."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from src.error_analysis.agreement_analysis import run_agreement_analysis

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
DEFAULT_RESULTS = PROJECT_ROOT / "results"
DEFAULT_OUT = ANALYSIS_DIR / "plots" / "agreement"


def main() -> None:
    parser = argparse.ArgumentParser(description="Agreement / disagreement analysis across runs.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-groups", type=int, default=None, help="Limit dataset/tier groups (debug)")
    parser.add_argument("--max-runs-per-group", type=int, default=25)
    args = parser.parse_args()

    index = run_agreement_analysis(
        results_root=args.results_root,
        out_dir=args.out_dir,
        max_groups=args.max_groups,
        max_runs_per_group=args.max_runs_per_group,
    )
    logger.info("Agreement index: {} groups → {}", len(index), args.out_dir)


if __name__ == "__main__":
    main()
