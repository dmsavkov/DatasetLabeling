# pyright: basic
"""Train/val overlap from shared contrast endpoints across disjoint pairs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_selection.gepa_optimizer_sets import build_gepa_train_val_from_huge_prediction
from src.datasets.schema import SCHEMA


def test_disjoint_when_val_pairs_share_endpoint_with_train(tmp_path: Path, monkeypatch) -> None:
    """
    Train pair (a0, shared). Val pair (shared, b0) — same ``shared`` id, different pair_ids.
    """
    rows = []
    for i in range(30):
        lab = "0" if i % 2 == 0 else "1"
        rows.append(
            {
                "sample_id": f"s{i}",
                "dataset_name": "tweet_eval_irony",
                "text": f"t{i}",
                "true_label": lab,
                "pred_label": lab,
                "correct": True,
            }
        )
    rows[0]["correct"] = False
    rows[0]["pred_label"] = "1"
    rows[10]["correct"] = False
    rows[10]["pred_label"] = "0"

    pred = pd.DataFrame(rows)
    edges = [
        {
            "anchor_sample_id": "s0",
            "anchor_true_label": "0",
            "contrast_sample_id": "shared",
            "contrast_true_label": "1",
        },
        {
            "anchor_sample_id": "shared",
            "anchor_true_label": "1",
            "contrast_sample_id": "s10",
            "contrast_true_label": "0",
        },
    ]
    out = tmp_path / "overlap_run"
    out.mkdir()
    pred.to_parquet(out / "predictions.parquet", index=False)
    pred.iloc[:5].to_parquet(out / "centroid_samples.parquet", index=False)
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
        train_total=4,
        train_easy_fraction=0.0,
        val_total=4,
        seed=99,
    )
    train_ids = set(result.train_df[SCHEMA.sample_id].astype(str))
    val_ids = set(result.val_df[SCHEMA.sample_id].astype(str))
    assert not train_ids & val_ids
    assert int(result.manifest.get("train_val_overlap", -1)) == 0
