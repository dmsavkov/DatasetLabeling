from __future__ import annotations

import pandas as pd

from src.datasets.schema import SCHEMA
from src.datasets.splitter import stratified_take


def test_stratified_take_repeatable_on_processed_like_df() -> None:
    df = pd.DataFrame(
        {
            SCHEMA.sample_id: [f"id_{i}" for i in range(200)],
            SCHEMA.dataset_name: ["x"] * 200,
            SCHEMA.text: [f"t{i}" for i in range(200)],
            SCHEMA.true_label: (["a"] * 120) + (["b"] * 60) + (["c"] * 20),
        }
    )
    a1 = stratified_take(df, 20, SCHEMA.true_label, seed=123)[SCHEMA.sample_id].tolist()
    a2 = stratified_take(df, 20, SCHEMA.true_label, seed=123)[SCHEMA.sample_id].tolist()
    b = stratified_take(df, 20, SCHEMA.true_label, seed=124)[SCHEMA.sample_id].tolist()

    assert a1 == a2
    assert a1 != b

