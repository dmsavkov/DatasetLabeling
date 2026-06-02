# pyright: basic
"""Multilabel / confusion-classification helpers for error analysis."""

from __future__ import annotations

import importlib
from typing import Any

import pandas as pd

from src.data_selection.label_utils import canonicalizer_for_dataset
from src.eval.metrics import compute_confusion_stats, compute_performance_excluding_confused
from src.error_analysis.io import LoadedExperiment
from src.error_analysis.labels import canonical_pred_value


def confusion_stats_from_predictions(df: pd.DataFrame) -> dict[str, int | float] | None:
    stats = compute_confusion_stats(df)
    if stats is None:
        return None
    return {
        "n_total": stats.n_total,
        "n_scored": stats.n_scored,
        "n_confusing": stats.n_confusing,
        "confusing_rate": stats.confusing_rate,
        "n_zero_labels": stats.n_zero_labels,
        "n_multi_labels": stats.n_multi_labels,
    }


def recompute_scored_metrics(
    df: pd.DataFrame,
    *,
    dataset_name: str,
) -> dict[str, float | int | None]:
    bundle = compute_performance_excluding_confused(df, dataset_name=dataset_name)
    stats = bundle.confusion_stats
    out: dict[str, float | int | None] = {
        "recomputed_accuracy": bundle.metrics.accuracy,
        "recomputed_f1_macro": bundle.metrics.f1_macro,
    }
    if stats is not None:
        out.update(
            {
                "n_total": stats.n_total,
                "n_scored": stats.n_scored,
                "n_confusing": stats.n_confusing,
                "confusing_rate": stats.confusing_rate,
            }
        )
    return out


def classification_report_dict(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    exclude_confusing: bool = True,
) -> dict[str, Any] | None:
    if "true_label" not in df.columns or "pred_label" not in df.columns:
        return None
    sub = df
    if exclude_confusing and "is_confusing" in sub.columns:
        sub = sub[~sub["is_confusing"].fillna(False).astype(bool)]
    sub = sub[sub["pred_label"].notna()].copy()
    if sub.empty:
        return None

    canon = canonicalizer_for_dataset(dataset_name)
    y_true = [canon(sub["true_label"].iloc[i]) for i in range(len(sub))]
    y_pred = [
        canon(canonical_pred_value(sub["pred_label"].iloc[i], dataset_name=dataset_name) or "")
        for i in range(len(sub))
    ]
    labels = sorted(set(y_true) | set(y_pred))

    sklearn_metrics = importlib.import_module("sklearn.metrics")
    classification_report = getattr(sklearn_metrics, "classification_report")
    return classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)


def classification_report_dataframe(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    exclude_confusing: bool = True,
) -> pd.DataFrame:
    report = classification_report_dict(
        df, dataset_name=dataset_name, exclude_confusing=exclude_confusing
    )
    if report is None:
        return pd.DataFrame()
    return pd.DataFrame(report).T


def per_experiment_metric_audit(e: LoadedExperiment) -> dict[str, object]:
    """Report.json metrics vs recomputed from predictions (scored rows only)."""
    df = e.predictions_df
    if df is None or df.empty:
        return {"exp_id": e.exp_id, "ok": False, "reason": "no predictions"}

    dataset_name = (
        e.meta.get("dataset_name")
        or (e.report or {}).get("dataset_name")
        or (str(df["dataset_name"].iloc[0]) if "dataset_name" in df.columns and len(df) else None)
    )
    if not isinstance(dataset_name, str):
        return {"exp_id": e.exp_id, "ok": False, "reason": "unknown dataset_name"}

    row: dict[str, object] = {"exp_id": e.exp_id, "dataset_name": dataset_name}
    row.update(recompute_scored_metrics(df, dataset_name=dataset_name))

    rpt = e.report or {}
    metrics = rpt.get("metrics") if isinstance(rpt.get("metrics"), dict) else {}
    if isinstance(metrics, dict):
        row["report_accuracy"] = metrics.get("accuracy")
        row["report_f1_macro"] = metrics.get("f1_macro")

    extras = rpt.get("extras") if isinstance(rpt.get("extras"), dict) else {}
    cs = extras.get("confusion_stats") if isinstance(extras, dict) else None
    if isinstance(cs, dict):
        for k in ("n_total", "n_scored", "n_confusing", "confusing_rate", "n_zero_labels", "n_multi_labels"):
            if k in cs:
                row[f"report_{k}"] = cs[k]

    pred_cs = confusion_stats_from_predictions(df)
    if pred_cs:
        for k, v in pred_cs.items():
            row[f"pred_{k}"] = v

    if row.get("report_accuracy") is not None and row.get("recomputed_accuracy") is not None:
        row["accuracy_delta"] = float(row["recomputed_accuracy"]) - float(row["report_accuracy"])

    return row
