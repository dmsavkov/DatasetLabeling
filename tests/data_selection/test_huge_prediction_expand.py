# pyright: basic
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_selection.gepa_optimizer_sets import expand_prediction_pool
from src.datasets.schema import SCHEMA


def _fake_embed(texts: list[str], *, model_name: str) -> np.ndarray:
    del model_name
    out = []
    for t in texts:
        h = abs(hash(t)) % 1000
        out.append([float(h % 7), float(h % 11), float(len(t) % 5)])
    return np.asarray(out, dtype=float)


def _pool(n_per_label: int = 30) -> pd.DataFrame:
    rows = []
    for lab in ("0", "1"):
        for i in range(n_per_label):
            rows.append(
                {
                    "sample_id": f"{lab}_{i}",
                    "dataset_name": "t",
                    "text": f"text {lab} {i}",
                    "true_label": lab,
                }
            )
    return pd.DataFrame(rows)


def test_expand_prediction_pool_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.data_selection.gepa_optimizer_sets.embed_texts",
        _fake_embed,
    )
    pool = _pool()
    centroids, prediction_pool = expand_prediction_pool(
        pool,
        label_ids=["0", "1"],
        n_centroids_per_label=5,
        prediction_size=20,
        embedding_model="fake",
        seed=1,
    )
    assert len(centroids) == 10
    assert len(prediction_pool) == 20
    assert set(prediction_pool[SCHEMA.true_label].astype(str)) == {"0", "1"}
