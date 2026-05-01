from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable

import pandas as pd


@dataclass(frozen=True, slots=True)
class ProcessedSampleSchema:
    """
    Universal processed sample schema used across all v1 datasets.

    Dataset-specific munging is allowed only inside dataset builders. Everything
    downstream (splitting, IO, evaluation) consumes only this schema.
    """

    sample_id: str = "sample_id"
    dataset_name: str = "dataset_name"
    text: str = "text"
    true_label: str = "true_label"
    meta_json: str = "meta_json"


SCHEMA: Final[ProcessedSampleSchema] = ProcessedSampleSchema()

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    SCHEMA.sample_id,
    SCHEMA.dataset_name,
    SCHEMA.text,
    SCHEMA.true_label,
)

OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (SCHEMA.meta_json,)


def validate_processed_samples_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Processed samples missing required columns: {missing}")


def required_columns() -> list[str]:
    return list(REQUIRED_COLUMNS)


def optional_columns() -> list[str]:
    return list(OPTIONAL_COLUMNS)


def all_known_columns() -> list[str]:
    return list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS)


def ensure_string_labels(df: pd.DataFrame, *, label_col: str = SCHEMA.true_label) -> pd.DataFrame:
    """
    Ensure `true_label` is a string label name (v1 assumption).
    Returns a shallow copy only when conversion is needed.
    """

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")
    if pd.api.types.is_string_dtype(df[label_col]):
        return df
    out = df.copy()
    out[label_col] = out[label_col].astype(str)
    return out


def ensure_unique_sample_ids(df: pd.DataFrame, *, id_col: str = SCHEMA.sample_id) -> None:
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found")
    dup = df[id_col].duplicated()
    if bool(dup.any()):
        raise ValueError("Duplicate sample_id detected in processed samples dataframe")


def stable_sort_for_determinism(
    df: pd.DataFrame, *, id_col: str = SCHEMA.sample_id, extra_cols: Iterable[str] = ()
) -> pd.DataFrame:
    """
    Deterministic ordering helper: sort by `sample_id` (and optionally other cols).
    """

    cols = [c for c in (list(extra_cols) + [id_col]) if c in df.columns]
    if not cols:
        return df
    return df.sort_values(cols, kind="mergesort").reset_index(drop=True)
