# pyright: basic
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_selection.gepa_optimizer_sets import (
    build_operating_frame,
    build_train_from_contrastive,
    build_val_holdout,
    centroid_samples_per_label,
    contrastive_neighbors_for_errors,
    resolve_focus_label,
    stratified_subset_df,
)
from src.datasets.schema import SCHEMA


def _fake_embed(texts: list[str], *, model_name: str) -> np.ndarray:
    del model_name
    out = []
    for t in texts:
        h = abs(hash(t)) % 1000
        out.append([float(h % 7), float(h % 11), float(len(t) % 5)])
    return np.asarray(out, dtype=float)


def _mini_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sample_id": "a1", "dataset_name": "t", "text": "methods endpoint design survival", "true_label": "2"},
            {"sample_id": "a2", "dataset_name": "t", "text": "methods endpoint design enrollment", "true_label": "2"},
            {"sample_id": "r1", "dataset_name": "t", "text": "results endpoint showed survival", "true_label": "3"},
            {"sample_id": "r2", "dataset_name": "t", "text": "results endpoint showed increase", "true_label": "3"},
        ]
    )


def test_stratified_subset_df_balanced() -> None:
    rows = []
    for lab in ("2", "3"):
        for i in range(20):
            rows.append(
                {
                    "sample_id": f"{lab}_{i}",
                    "dataset_name": "t",
                    "text": f"text {lab} {i}",
                    "true_label": lab,
                }
            )
    df = pd.DataFrame(rows)
    out = stratified_subset_df(df, n_total=10, labels=["2", "3"], seed=0)
    assert len(out) == 10
    assert out["true_label"].value_counts().to_dict() == {"2": 5, "3": 5}


def test_resolve_focus_label_pubmed_alias() -> None:
    assert resolve_focus_label("methods", labels_in_data=["0", "1", "2", "3", "4"]) == "2"


def test_centroid_samples_one_per_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.data_selection.gepa_optimizer_sets.embed_texts",
        _fake_embed,
    )
    df = _mini_df()
    out = centroid_samples_per_label(
        df,
        labels=["2", "3"],
        n_clusters=2,
        embedding_model="fake",
        seed=1,
        focus_label="2",
    )
    assert len(out) == 2
    assert set(out[SCHEMA.true_label].tolist()) == {"2"}


def test_contrastive_neighbor_other_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.data_selection.gepa_optimizer_sets.embed_texts",
        _fake_embed,
    )
    pool = _mini_df()
    errors = pool.iloc[[0]].copy()
    errors["pred_label"] = "3"
    edges = contrastive_neighbors_for_errors(errors, pool, embedding_model="fake", focus_label="2")
    assert len(edges) == 1
    assert edges.iloc[0]["anchor_true_label"] == "2"
    assert edges.iloc[0]["contrast_true_label"] != "2"


def test_operating_frame_includes_contrast_without_pred() -> None:
    pool = _mini_df()
    centroids = pool.iloc[[0, 1]].copy()  # a1, a2 only
    centroid_preds = centroids.copy()
    centroid_preds["pred_label"] = centroid_preds[SCHEMA.true_label]
    centroid_preds["correct"] = True
    centroid_preds.loc[0, "pred_label"] = "3"
    centroid_preds.loc[0, "correct"] = False
    edges = pd.DataFrame(
        [
            {
                "anchor_sample_id": "a1",
                "contrast_sample_id": "r1",
                "anchor_true_label": "2",
                "contrast_true_label": "3",
            }
        ]
    )
    op = build_operating_frame(pool, centroids, centroid_preds, edges)
    assert len(op) == 3
    r1 = op[op[SCHEMA.sample_id] == "r1"].iloc[0]
    assert pd.isna(r1["pred_label"]) or r1["pred_label"] is None
    assert bool(r1["correct"]) is False


def test_train_val_disjoint() -> None:
    pool = _mini_df()
    centroids = pool.copy()
    centroid_preds = centroids.copy()
    centroid_preds["pred_label"] = centroid_preds[SCHEMA.true_label]
    centroid_preds["correct"] = True
    centroid_preds.loc[0, "pred_label"] = "3"
    centroid_preds.loc[0, "correct"] = False
    operating = build_operating_frame(pool, centroids, centroid_preds, pd.DataFrame())

    contrastive = pd.DataFrame(
        [
            {
                "anchor_sample_id": "a1",
                "contrast_sample_id": "r1",
                "anchor_true_label": "2",
                "contrast_true_label": "3",
            }
        ]
    )
    operating = build_operating_frame(
        pool,
        centroids,
        centroid_preds,
        contrastive,
    )
    val_df, val_ids = build_val_holdout(
        operating,
        contrastive,
        labels=["2", "3"],
        n_total=2,
        contrastive_fraction=0.5,
        seed=0,
    )
    train_df = build_train_from_contrastive(operating, contrastive, val_ids=val_ids)
    overlap = set(val_df[SCHEMA.sample_id]) & set(train_df[SCHEMA.sample_id])
    assert not overlap
