# pyright: basic
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data_selection.label_utils import canonicalizer_for_dataset
from src.eval.label_compare import labels_equal_for_metrics, normalize_label_scalar


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    f1_macro: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class ConfusionStats:
    n_total: int
    n_scored: int
    n_confusing: int
    confusing_rate: float
    n_zero_labels: int = 0
    n_multi_labels: int = 0


@dataclass(frozen=True, slots=True)
class PerformanceWithConfusion:
    metrics: PerformanceMetrics
    confusion_stats: ConfusionStats | None


def compute_performance_metrics(
    df: pd.DataFrame,
    *,
    dataset_name: str | None = None,
) -> PerformanceMetrics:
    if "true_label" not in df.columns or "pred_label" not in df.columns:
        raise ValueError("df must contain true_label and pred_label")

    if dataset_name:
        mask = df["pred_label"].notna()
        sub = df[mask]
        if sub.empty:
            return PerformanceMetrics(f1_macro=0.0, accuracy=0.0)
        correct = [
            labels_equal_for_metrics(sub["true_label"].iloc[i], sub["pred_label"].iloc[i], dataset_name=dataset_name)
            for i in range(len(sub))
        ]
        accuracy = float(np.mean(correct)) if correct else 0.0
        y_true = [canonicalizer_for_dataset(dataset_name)(sub["true_label"].iloc[i]) for i in range(len(sub))]
        y_pred = [
            canonicalizer_for_dataset(dataset_name)(normalize_label_scalar(sub["pred_label"].iloc[i]) or "")
            for i in range(len(sub))
        ]
        sklearn_metrics = importlib.import_module("sklearn.metrics")
        f1_score = getattr(sklearn_metrics, "f1_score")
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        return PerformanceMetrics(f1_macro=f1_macro, accuracy=accuracy)

    # Legacy path (no dataset_name): raw string compare.
    # Nullable preds (abstain) often become float NaN in DataFrame columns; avoid
    # mixed float/str/object types that break sklearn label checks.
    y_true = df["true_label"].astype("string").fillna("").astype(str)
    y_pred = df["pred_label"].astype("string").fillna("").astype(str)

    sklearn_metrics = importlib.import_module("sklearn.metrics")
    f1_score = getattr(sklearn_metrics, "f1_score")
    accuracy_score = getattr(sklearn_metrics, "accuracy_score")

    return PerformanceMetrics(
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
    )


def compute_confusion_stats(df: pd.DataFrame, *, confusing_col: str = "is_confusing") -> ConfusionStats | None:
    if confusing_col not in df.columns:
        return None
    n_total = int(len(df))
    confusing = df[confusing_col].fillna(False).astype(bool)
    n_confusing = int(confusing.sum())
    n_scored = n_total - n_confusing
    n_zero = 0
    n_multi = 0
    if "n_pred_labels" in df.columns:
        for v in df["n_pred_labels"].tolist():
            try:
                k = int(v)
                if k == 0:
                    n_zero += 1
                elif k > 1:
                    n_multi += 1
            except Exception:
                continue
    rate = float(n_confusing / n_total) if n_total else 0.0
    return ConfusionStats(
        n_total=n_total,
        n_scored=n_scored,
        n_confusing=n_confusing,
        confusing_rate=rate,
        n_zero_labels=n_zero,
        n_multi_labels=n_multi,
    )


def compute_performance_excluding_confused(
    df: pd.DataFrame,
    *,
    true_col: str = "true_label",
    pred_col: str = "pred_label",
    confusing_col: str = "is_confusing",
    dataset_name: str | None = None,
) -> PerformanceWithConfusion:
    stats = compute_confusion_stats(df, confusing_col=confusing_col)
    if confusing_col in df.columns:
        scored = df[~df[confusing_col].fillna(False).astype(bool)].copy()
    else:
        scored = df
    if scored.empty or pred_col not in scored.columns:
        metrics = PerformanceMetrics(f1_macro=0.0, accuracy=0.0)
    else:
        sub = scored.rename(columns={true_col: "true_label", pred_col: "pred_label"})
        mask = sub["pred_label"].notna()
        sub = sub[mask]
        if sub.empty:
            metrics = PerformanceMetrics(f1_macro=0.0, accuracy=0.0)
        else:
            metrics = compute_performance_metrics(sub, dataset_name=dataset_name)
    return PerformanceWithConfusion(metrics=metrics, confusion_stats=stats)


def agreement_rate(pred_a: pd.Series, pred_b: pd.Series) -> float:
    a = pred_a.astype("string")
    b = pred_b.astype("string")
    mask = a.notna() & b.notna()
    if int(mask.sum()) == 0:
        return 0.0
    return float((a[mask] == b[mask]).mean())


def probs_if_accessible(df: pd.DataFrame) -> dict[str, float] | None:
    """
    If `probs` (dict-like) or `confidence` is available, compute simple aggregates.
    """

    if "probs" in df.columns:
        max_probs: list[float] = []
        entropies: list[float] = []
        for item in df["probs"].tolist():
            if not isinstance(item, dict) or not item:
                continue
            arr = np.array([float(v) for v in item.values()], dtype=float)
            arr = arr[arr >= 0]
            if arr.size == 0:
                continue
            s = float(arr.sum())
            if s > 0:
                p = arr / s
                ent = float(-(p * np.log(p + 1e-12)).sum())
                entropies.append(ent)
            max_probs.append(float(arr.max()))
        if not max_probs and not entropies:
            return None
        out: dict[str, float] = {}
        if max_probs:
            out["mean_max_prob"] = float(np.mean(max_probs))
        if entropies:
            out["mean_entropy"] = float(np.mean(entropies))
        return out

    if "confidence" in df.columns:
        vals: list[float] = []
        for v in df["confidence"].tolist():
            try:
                if v is None:
                    continue
                vals.append(float(v))
            except Exception:
                continue
        if not vals:
            return None
        return {"mean_confidence": float(np.mean(np.array(vals, dtype=float)))}

    return None


def summarize_usage(df: pd.DataFrame) -> dict[str, Any] | None:
    if "in_tokens" not in df.columns and "out_tokens" not in df.columns:
        return None
    in_total: int | None = None
    out_total: int | None = None

    if "in_tokens" in df.columns:
        s = 0
        seen = False
        for v in df["in_tokens"].tolist():
            try:
                if v is None:
                    continue
                s += int(v)
                seen = True
            except Exception:
                continue
        if seen:
            in_total = int(s)

    if "out_tokens" in df.columns:
        s = 0
        seen = False
        for v in df["out_tokens"].tolist():
            try:
                if v is None:
                    continue
                s += int(v)
                seen = True
            except Exception:
                continue
        if seen:
            out_total = int(s)

    if in_total is None and out_total is None:
        return None
    return {"in_tokens_total": in_total, "out_tokens_total": out_total}

