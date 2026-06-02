#!/usr/bin/env python
"""Run full analysis pipeline: leaderboard + agreement + aggregated reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from src.error_analysis.agreement_analysis import run_agreement_analysis
from src.error_analysis.io import load_many
from src.error_analysis.leaderboard import write_leaderboard
from src.error_analysis.reports import aggregate_reports, plot_overview, save_artifacts

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
DEFAULT_RESULTS = PROJECT_ROOT / "results"
DEFAULT_CSV = ANALYSIS_DIR / "leaderboard.csv"
DEFAULT_PLOTS = ANALYSIS_DIR / "plots" / "leaderboard"
DEFAULT_AGREEMENT = ANALYSIS_DIR / "plots" / "agreement"
DEFAULT_REPORTS = ANALYSIS_DIR / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all analysis artifacts under analysis/.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--skip-agreement", action="store_true")
    parser.add_argument("--skip-reports", action="store_true")
    parser.add_argument("--max-agreement-groups", type=int, default=None)
    args = parser.parse_args()

    write_leaderboard(
        results_root=args.results_root,
        csv_path=DEFAULT_CSV,
        plots_dir=DEFAULT_PLOTS,
    )

    if not args.skip_agreement:
        run_agreement_analysis(
            results_root=args.results_root,
            out_dir=DEFAULT_AGREEMENT,
            max_groups=args.max_agreement_groups,
        )

    if not args.skip_reports:
        exps, load_table = load_many([str(args.results_root)], max_depth=8)
        reports_df = aggregate_reports(exps)
        save_artifacts(
            str(DEFAULT_REPORTS),
            exps=exps,
            reports_df=reports_df,
        )
        load_table.to_csv(DEFAULT_REPORTS / "discovery_index.csv", index=False)
        plot_overview(reports_df)
        logger.info("Wrote reports under {}", DEFAULT_REPORTS)

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
