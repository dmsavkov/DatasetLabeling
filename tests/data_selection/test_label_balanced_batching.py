# pyright: basic
from __future__ import annotations

import pandas as pd

from src.data_selection.label_balanced_batching import (
    build_label_balanced_batches,
    dataframe_to_sentence_rows,
)
from src.datasets.schema import SCHEMA


def _rows_df() -> pd.DataFrame:
    rows = []
    for lab in ("0", "1"):
        for i in range(30):
            rows.append(
                {
                    "sample_id": f"{lab}_{i}",
                    "dataset_name": "t",
                    "text": f"t {lab} {i}",
                    "true_label": lab,
                }
            )
    return pd.DataFrame(rows)


def test_batches_mixed_labels() -> None:
    sentence_rows = dataframe_to_sentence_rows(_rows_df(), label_key_fn=lambda x: str(x))
    batches = build_label_balanced_batches(sentence_rows, batch_size=5, seed=0)
    assert len(batches) == 12
    mixed = 0
    for b in batches:
        assert b.size == 5
        if len({r.label_key for r in b.rows}) >= 2:
            mixed += 1
    assert mixed >= 10
