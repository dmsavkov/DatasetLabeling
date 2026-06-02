#!/usr/bin/env python
"""Build ``analysis/leaderboard.csv`` and summary plots from all ``results/`` runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from src.error_analysis.leaderboard import write_leaderboard

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
DEFAULT_RESULTS = PROJECT_ROOT / "results"
DEFAULT_CSV = ANALYSIS_DIR / "leaderboard.csv"
DEFAULT_PLOTS = ANALYSIS_DIR / "plots" / "leaderboard"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build experiment leaderboard CSV + plots.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    df = write_leaderboard(
        results_root=args.results_root,
        csv_path=args.csv,
        plots_dir=args.plots_dir,
        write_plots=not args.no_plots,
    )
    if not df.empty and "f1_macro" in df.columns:
        best = df.sort_values("f1_macro", ascending=False, na_position="last").iloc[0]
        logger.info(
            "Best row: {} f1_macro={} ({})",
            best.get("run_key"),
            best.get("f1_macro"),
            best.get("series"),
        )


if __name__ == "__main__":
    main()
