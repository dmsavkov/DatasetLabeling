# pyright: basic
"""Label-aware batch construction for batched DSPy examples."""

from __future__ import annotations

import pandas as pd

from src.data_selection.label_balanced_batching import (
    SentenceRow,
    TextBatch,
    batch_to_numbered_text,
    batches_to_manifest,
    build_label_balanced_batches,
    dataframe_to_sentence_rows as _dataframe_to_sentence_rows,
)
from src.dspy_gepa.labels import normalize_label_for_dataset


def dataframe_to_sentence_rows(df: pd.DataFrame, *, dataset_name: str) -> list[SentenceRow]:
    fn = lambda v: normalize_label_for_dataset(v, dataset_name=dataset_name)
    return _dataframe_to_sentence_rows(df, label_key_fn=fn)


__all__ = [
    "SentenceRow",
    "TextBatch",
    "batch_to_numbered_text",
    "batches_to_manifest",
    "build_label_balanced_batches",
    "dataframe_to_sentence_rows",
]
