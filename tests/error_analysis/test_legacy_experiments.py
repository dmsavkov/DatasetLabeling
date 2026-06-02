# pyright: basic
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.error_analysis.legacy_experiments import discover_legacy_experiments
from src.error_analysis.leaderboard import build_leaderboard_df


def test_legacy_mvp4_and_hf_rows(tmp_path: Path) -> None:
    results = tmp_path / "results"
    mvp4 = results / "mvp4_results"
    mvp4.mkdir(parents=True)
    (mvp4 / "mvp4_final_results.json").write_text(
        json.dumps(
            [
                {
                    "dataset": "ag_news",
                    "model_family": "ML",
                    "model": "svm_tfidf",
                    "train_n": 100,
                    "test_n": 50,
                    "accuracy": 0.6,
                    "macro_f1": 0.55,
                }
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"text": ["a"], "true": ["Sports"], "pred": ["Sports"]}).to_csv(
        mvp4 / "preds_ag_news_svm_n100.csv", index=False
    )

    hf = results / "hf_llms_comparison"
    hf.mkdir()
    stamp = "20260425_120000"
    (hf / f"summary_{stamp}.json").write_text("{}", encoding="utf-8")
    pd.DataFrame(
        [{"dataset": "ag_news", "model": "llama_8b", "n_samples": 10, "accuracy": 0.8, "macro_f1": 0.75}]
    ).to_csv(hf / f"metrics_{stamp}.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "ag_news",
                "sample_id": "s1",
                "true_label": "Sports",
                "pred_label": "Sports",
                "model": "llama_8b",
            }
        ]
    ).to_csv(hf / f"predictions_long_{stamp}.csv", index=False)

    legacy = discover_legacy_experiments(results)
    assert len(legacy) == 2
    df = build_leaderboard_df(results)
    assert len(df) == 2
    assert set(df["series"]) == {"mvp4_results", "hf_llms_comparison"}
