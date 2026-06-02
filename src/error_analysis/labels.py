# pyright: basic
"""Dataset-aware label normalization for error-analysis comparisons."""

from __future__ import annotations

import pandas as pd

from src.data_selection.label_utils import canonicalizer_for_dataset
from src.eval.label_compare import normalize_label_scalar


def dataset_name_from_frame(df: pd.DataFrame) -> str | None:
    if "dataset_name" not in df.columns:
        return None
    vals = df["dataset_name"].dropna().astype(str).unique().tolist()
    if len(vals) == 1:
        return vals[0]
    return None


def canonical_pred_value(value: object, *, dataset_name: str) -> str | None:
    """Canonical class id/name for cross-run agreement; None if pred is absent."""
    norm = normalize_label_scalar(value)
    if norm is None:
        return None
    return canonicalizer_for_dataset(dataset_name)(norm)


def canonical_pred_series(series: pd.Series, *, dataset_name: str) -> pd.Series:
    return series.map(lambda v: canonical_pred_value(v, dataset_name=dataset_name))


def preds_equal(
    a: object,
    b: object,
    *,
    dataset_name: str,
) -> bool:
    ca = canonical_pred_value(a, dataset_name=dataset_name)
    cb = canonical_pred_value(b, dataset_name=dataset_name)
    if ca is None or cb is None:
        return False
    return ca == cb
