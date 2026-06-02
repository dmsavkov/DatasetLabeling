from __future__ import annotations

from src.error_analysis.compare import (
    assert_same_dataset,
    build_comparison_df,
    confusions,
    disagreements,
    pairwise_agreement_matrix,
    multiwise_diff,
)
from src.error_analysis.io import LoadedExperiment, discover_experiments, load_experiment, load_many
from src.error_analysis.pairing import (
    build_paired_row_comparison,
    compare_prompt_eng_vs_google_eval,
)
from src.error_analysis.reports import aggregate_reports, plot_overview, save_artifacts
from src.error_analysis.row_metrics import (
    RowVoteMetricsConfig,
    add_model_correctness_flags,
    add_row_vote_metrics,
    calibration_bins,
    expected_calibration_error,
)

__all__ = [
    "LoadedExperiment",
    "discover_experiments",
    "load_experiment",
    "load_many",
    "compare_prompt_eng_vs_google_eval",
    "build_paired_row_comparison",
    "assert_same_dataset",
    "build_comparison_df",
    "disagreements",
    "multiwise_diff",
    "confusions",
    "pairwise_agreement_matrix",
    "aggregate_reports",
    "plot_overview",
    "save_artifacts",
    "RowVoteMetricsConfig",
    "add_row_vote_metrics",
    "add_model_correctness_flags",
    "calibration_bins",
    "expected_calibration_error",
]

