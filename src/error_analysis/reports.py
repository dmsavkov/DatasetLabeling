from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.error_analysis.io import LoadedExperiment


def _get(d: dict[str, Any] | None, *keys: str) -> object | None:
    cur: object | None = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def aggregate_reports(exps: list[LoadedExperiment]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for e in exps:
        r = e.report or {}
        c = e.config or {}
        model = c.get("model") if isinstance(c.get("model"), dict) else {}
        params = model.get("params") if isinstance(model, dict) and isinstance(model.get("params"), dict) else {}

        rows.append(
            {
                "exp_id": e.exp_id,
                "path": str(e.path),
                "series": e.meta.get("series"),
                "campaign": e.meta.get("campaign"),
                "suite": e.meta.get("suite"),
                "ok": len(e.errors) == 0,
                "dataset_name": e.meta.get("dataset_name")
                or r.get("dataset_name")
                or (
                    e.predictions_df["dataset_name"].iloc[0]
                    if e.predictions_df is not None
                    and "dataset_name" in e.predictions_df.columns
                    and len(e.predictions_df)
                    else None
                ),
                "tier_size": r.get("tier_size"),
                "predictor_name": r.get("predictor_name"),
                "f1_macro": _get(r, "metrics", "f1_macro"),
                "accuracy": _get(r, "metrics", "accuracy"),
                "infer_time_s": _get(r, "extras", "infer_time_s"),
                "train_time_s": _get(r, "extras", "train_time_s"),
                "model_kind": model.get("kind") if isinstance(model, dict) else None,
                "model_id": params.get("model_id") if isinstance(params, dict) else None,
                "prompt_id": params.get("prompt_id") if isinstance(params, dict) else None,
                "batch_size": params.get("batch_size") if isinstance(params, dict) else None,
                "max_concurrency": params.get("max_concurrency") if isinstance(params, dict) else None,
                "temperature": params.get("temperature") if isinstance(params, dict) else None,
                "max_tokens": params.get("max_tokens") if isinstance(params, dict) else None,
                "retries": params.get("retries") if isinstance(params, dict) else None,
                "in_tokens_total": _get(r, "extras", "usage", "in_tokens_total"),
                "out_tokens_total": _get(r, "extras", "usage", "out_tokens_total"),
                "n_warnings": len(e.warnings),
                "n_errors": len(e.errors),
                "has_confusion_stats": _get(r, "extras", "confusion_stats") is not None,
                "n_total": _get(r, "extras", "confusion_stats", "n_total"),
                "n_scored": _get(r, "extras", "confusion_stats", "n_scored"),
                "n_confusing": _get(r, "extras", "confusion_stats", "n_confusing"),
                "confusing_rate": _get(r, "extras", "confusion_stats", "confusing_rate"),
                "n_zero_labels": _get(r, "extras", "confusion_stats", "n_zero_labels"),
                "n_multi_labels": _get(r, "extras", "confusion_stats", "n_multi_labels"),
            }
        )
    return pd.DataFrame(rows)


def plot_overview(df: pd.DataFrame):
    """
    Minimal plotting helper.
    Returns matplotlib axes objects (if available).
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    if "infer_time_s" in df.columns and "f1_macro" in df.columns:
        ax[0].scatter(df["infer_time_s"], df["f1_macro"])
        ax[0].set_xlabel("infer_time_s")
        ax[0].set_ylabel("f1_macro")
        ax[0].set_title("Speed vs F1")

    if "f1_macro" in df.columns and "exp_id" in df.columns:
        top = df.sort_values("f1_macro", ascending=False).head(15)
        ax[1].barh(top["exp_id"].astype(str), top["f1_macro"])
        ax[1].invert_yaxis()
        ax[1].set_xlabel("f1_macro")
        ax[1].set_title("Top F1 (first 15)")

    fig.tight_layout()
    return ax


def save_artifacts(
    out_dir: str,
    *,
    exps: list[LoadedExperiment],
    reports_df: pd.DataFrame,
    comparison_df: pd.DataFrame | None = None,
    disagreements_df: pd.DataFrame | None = None,
    agreement_df: pd.DataFrame | None = None,
    save_comparison_wide: bool = False,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Loaded summary
    loaded_rows: list[dict[str, object]] = []
    for e in exps:
        loaded_rows.append(
            {
                "exp_id": e.exp_id,
                "path": str(e.path),
                "ok": len(e.errors) == 0,
                "has_predictions": e.predictions_df is not None,
                "has_report": e.report is not None,
                "has_config": e.config is not None,
                "n_warnings": len(e.warnings),
                "n_errors": len(e.errors),
                "warnings": "\n".join(e.warnings) if e.warnings else "",
                "errors": "\n".join(e.errors) if e.errors else "",
            }
        )
    pd.DataFrame(loaded_rows).to_csv(out / "loaded_experiments.csv", index=False)

    # Aggregated reports
    reports_df.to_csv(out / "reports_aggregated.csv", index=False)

    if comparison_df is not None and save_comparison_wide:
        comparison_df.to_csv(out / "comparison_wide.csv", index=False)

    if disagreements_df is not None:
        disagreements_df.to_csv(out / "disagreements.csv", index=False)

    if agreement_df is not None:
        agreement_df.to_csv(out / "pairwise_agreement.csv", index=True)

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "saved": sorted([p.name for p in out.iterdir() if p.is_file()]),
                "out_dir": str(out),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

