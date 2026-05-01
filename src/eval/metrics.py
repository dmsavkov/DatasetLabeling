# pyright: basic
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    f1_macro: float
    accuracy: float


def compute_performance_metrics(df: pd.DataFrame) -> PerformanceMetrics:
    if "true_label" not in df.columns or "pred_label" not in df.columns:
        raise ValueError("df must contain true_label and pred_label")
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

