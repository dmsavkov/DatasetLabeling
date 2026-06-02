# pyright: basic
from __future__ import annotations

import pandas as pd

from src.data_selection.huge_prediction_representatives import (
    build_shuffled_batches,
    shuffle_prediction_pool,
)
from src.datasets.schema import SCHEMA


def _df() -> pd.DataFrame:
    rows = []
    for lab in ("0", "0", "0", "1", "1"):
        for i in range(2):
            rows.append(
                {
                    "sample_id": f"{lab}_{i}",
                    "dataset_name": "t",
                    "text": f"t {lab}",
                    "true_label": lab,
                }
            )
    return pd.DataFrame(rows)


def test_shuffle_changes_row_order() -> None:
    df = _df()
    shuffled = shuffle_prediction_pool(df, seed=0)
    assert shuffled[SCHEMA.sample_id].tolist() != df[SCHEMA.sample_id].tolist()


def test_build_shuffled_batches_count() -> None:
    df = _df()
    batches = build_shuffled_batches(df, batch_size=3)
    assert len(batches) == 4
    assert all(len(chunk) <= 3 for _, chunk in batches)
