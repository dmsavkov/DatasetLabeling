# pyright: basic
"""Dataset-aware gold vs pred comparison for eval metrics."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.data_selection.label_utils import canonicalizer_for_dataset


def normalize_label_scalar(value: object) -> str | None:
    """Stable string for metrics; maps float 4.0 -> '4'."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "?"}:
        return None
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        return s[:-2]
    return s


def labels_equal_for_metrics(
    true_label: object,
    pred_label: object,
    *,
    dataset_name: str,
) -> bool:
    pred_norm = normalize_label_scalar(pred_label)
    if pred_norm is None:
        return False
    canon = canonicalizer_for_dataset(dataset_name)
    return canon(true_label) == canon(pred_norm)


def correctness_series(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    true_col: str,
    pred_col: str = "pred_label",
    confusing_col: str = "is_confusing",
) -> pd.Series:
    """Row-wise correctness with optional multilabel confusion exclusion."""

    def row_ok(row: pd.Series) -> bool:
        if confusing_col in df.columns and bool(row.get(confusing_col)):
            return False
        gold = row.get(true_col)
        pred = row.get(pred_col)
        if gold is None or (isinstance(gold, float) and pd.isna(gold)):
            return False
        return labels_equal_for_metrics(gold, pred, dataset_name=dataset_name)

    return df.apply(row_ok, axis=1)
