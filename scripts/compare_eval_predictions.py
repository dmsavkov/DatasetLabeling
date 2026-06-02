# pyright: basic
"""
Compare two evaluate_google_llm prediction CSVs (same sample_id column).

Example:
  uv run python scripts/compare_eval_predictions.py \\
    results/evaluate_google_llm/20260503_091005/.../predictions.csv \\
    results/evaluate_google_llm/20260518_111949/.../predictions.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.eval.label_compare import labels_equal_for_metrics


def compare_runs(
    path_a: Path,
    path_b: Path,
    *,
    dataset_name: str = "pubmed_20k_rct",
) -> dict[str, object]:
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    m = a.merge(b, on="sample_id", suffixes=("_a", "_b"), how="inner")
    if m.empty:
        raise ValueError("No overlapping sample_id rows between runs")

    pred_agree = m["pred_label_a"].astype(str) == m["pred_label_b"].astype(str)
    correct_a = [
        labels_equal_for_metrics(m["true_label_a"].iloc[i], m["pred_label_a"].iloc[i], dataset_name=dataset_name)
        for i in range(len(m))
    ]
    correct_b = [
        labels_equal_for_metrics(m["true_label_b"].iloc[i], m["pred_label_b"].iloc[i], dataset_name=dataset_name)
        for i in range(len(m))
    ]
    ca = pd.Series(correct_a)
    cb = pd.Series(correct_b)
    flipped = (~ca & cb) | (ca & ~cb)

    return {
        "path_a": str(path_a),
        "path_b": str(path_b),
        "n_overlap": int(len(m)),
        "pred_agreement": float(pred_agree.mean()),
        "pred_disagree_n": int((~pred_agree).sum()),
        "accuracy_a": float(ca.mean()),
        "accuracy_b": float(cb.mean()),
        "accuracy_delta_b_minus_a": float(cb.mean() - ca.mean()),
        "a_correct_b_wrong": int((ca & ~cb).sum()),
        "a_wrong_b_correct": int((~ca & cb).sum()),
        "flipped_n": int(flipped.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two eval prediction CSV files.")
    _ = ap.add_argument("predictions_a", type=Path)
    _ = ap.add_argument("predictions_b", type=Path)
    _ = ap.add_argument("--dataset", type=str, default="pubmed_20k_rct")
    args = ap.parse_args()
    out = compare_runs(args.predictions_a, args.predictions_b, dataset_name=str(args.dataset))
    print(json.dumps(out, indent=2))
    if out["pred_disagree_n"] and out["accuracy_delta_b_minus_a"]:
        print(
            f"\nNote: {out['pred_disagree_n']} rows changed predictions; "
            f"accuracy moved by {out['accuracy_delta_b_minus_a']:+.3f}. "
            "Identical configs can still differ (API/thinking non-determinism).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
