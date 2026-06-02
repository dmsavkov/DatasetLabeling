# pyright: basic
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_selection.gepa_optimizer_sets import (
    build_contrastive_pair_table,
    build_gepa_train_val_from_huge_prediction,
)
from src.datasets.schema import SCHEMA


def _write_synthetic_huge_dir(tmp_path: Path) -> Path:
    rows = []
    for lab in ("0", "1"):
        for i in range(40):
            rows.append(
                {
                    "sample_id": f"{lab}_{i}",
                    "dataset_name": "tweet_eval_irony",
                    "text": f"text {lab} {i}",
                    "true_label": lab,
                    "pred_label": lab if i % 3 else ("1" if lab == "0" else "0"),
                    "correct": i % 3 != 0,
                }
            )
    pred = pd.DataFrame(rows)
    edges = []
    for lab in ("0", "1"):
        for j in range(8):
            edges.append(
                {
                    "anchor_sample_id": f"{lab}_{j * 3}",
                    "anchor_true_label": lab,
                    "contrast_sample_id": f"{'1' if lab == '0' else '0'}_{j}",
                    "contrast_true_label": "1" if lab == "0" else "0",
                }
            )
    out = tmp_path / "run"
    out.mkdir()
    pred.to_parquet(out / "predictions.parquet", index=False)
    pred.iloc[:10].to_parquet(out / "prediction_pool.parquet", index=False)
    pred.iloc[:4].to_parquet(out / "centroid_samples.parquet", index=False)
    pd.DataFrame(edges).to_parquet(out / "contrastive_edges.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps({"dataset_name": "tweet_eval_irony"}), encoding="utf-8")
    return out


def test_build_gepa_train_val_disjoint(tmp_path: Path, monkeypatch) -> None:
    huge_dir = _write_synthetic_huge_dir(tmp_path)

    def _fake_load(dataset_name: str, **kwargs):  # type: ignore[no-untyped-def]
        from src.data_selection.label_utils import DatasetContext

        del kwargs
        return DatasetContext(
            dataset_name=dataset_name,
            label_ids=["0", "1"],
            prompt_labels=["0", "1"],
            prompt_id="baseline_v1",
        )

    monkeypatch.setattr(
        "src.data_selection.label_utils.load_dataset_context",
        _fake_load,
    )
    result = build_gepa_train_val_from_huge_prediction(
        huge_dir,
        dataset_name="tweet_eval_irony",
        train_total=10,
        train_easy_fraction=0.2,
        val_total=8,
        seed=0,
    )
    train_ids = set(result.train_df[SCHEMA.sample_id].astype(str))
    val_ids = set(result.val_df[SCHEMA.sample_id].astype(str))
    assert not train_ids & val_ids
    assert len(result.train_df) == 10
    assert len(result.val_df) == 8

    pairs = build_contrastive_pair_table(pd.read_parquet(huge_dir / "contrastive_edges.parquet"))
    assert len(pairs) == 16


def test_build_gepa_hits_targets_when_contrast_outside_pool(tmp_path: Path, monkeypatch) -> None:
    """Contrast neighbors are often outside the 500-row prediction pool."""
    rows = []
    for lab in ("0", "1"):
        for i in range(30):
            rows.append(
                {
                    "sample_id": f"p_{lab}_{i}",
                    "dataset_name": "tweet_eval_irony",
                    "text": f"pool {lab} {i}",
                    "true_label": lab,
                    "pred_label": lab if i % 2 else ("1" if lab == "0" else "0"),
                    "correct": i % 2 != 0,
                }
            )
    pred = pd.DataFrame(rows)
    edges = []
    for lab in ("0", "1"):
        for j in range(10):
            aid = f"p_{lab}_{j * 2}"
            cid = f"ext_{lab}_{j}"
            edges.append(
                {
                    "anchor_sample_id": aid,
                    "anchor_true_label": lab,
                    "contrast_sample_id": cid,
                    "contrast_true_label": "1" if lab == "0" else "0",
                    "contrast_text": f"external contrast {lab} {j}",
                }
            )
    out = tmp_path / "run2"
    out.mkdir()
    pred.to_parquet(out / "predictions.parquet", index=False)
    pred.iloc[:8].to_parquet(out / "centroid_samples.parquet", index=False)
    pd.DataFrame(edges).to_parquet(out / "contrastive_edges.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps({"dataset_name": "tweet_eval_irony"}), encoding="utf-8")

    def _fake_load(dataset_name: str, **kwargs):  # type: ignore[no-untyped-def]
        from src.data_selection.label_utils import DatasetContext

        del kwargs
        return DatasetContext(
            dataset_name=dataset_name,
            label_ids=["0", "1"],
            prompt_labels=["0", "1"],
            prompt_id="baseline_v1",
        )

    monkeypatch.setattr(
        "src.data_selection.label_utils.load_dataset_context",
        _fake_load,
    )
    result = build_gepa_train_val_from_huge_prediction(
        out,
        dataset_name="tweet_eval_irony",
        train_total=10,
        train_easy_fraction=0.2,
        val_total=8,
        seed=1,
    )
    assert len(result.train_df) == 10
    assert len(result.val_df) == 8
    assert int(result.manifest.get("contrast_rows_enriched_from_edges", 0)) > 0
