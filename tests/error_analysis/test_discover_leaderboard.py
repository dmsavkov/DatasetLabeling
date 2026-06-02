# pyright: basic
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.error_analysis.discover import discover_all_runs
from src.error_analysis.leaderboard import build_leaderboard_df
from src.error_analysis.run_record import extract_run_record


def _write_harness_run(base: Path, *, name: str, dataset: str, f1: float) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "dataset_name": dataset,
                "text": "hi",
                "true_label": "a",
                "pred_label": "a",
            }
        ]
    ).to_csv(base / "predictions.csv", index=False)
    (base / "report.json").write_text(
        json.dumps(
            {
                "dataset_name": dataset,
                "tier_size": 10,
                "metrics": {"f1_macro": f1, "accuracy": f1},
            }
        ),
        encoding="utf-8",
    )
    (base / "config.resolved.json").write_text(
        json.dumps(
            {
                "name": name,
                "model": {"kind": "toy", "params": {"model_id": "m1", "thinking_level": "low"}},
            }
        ),
        encoding="utf-8",
    )
    return base


def test_discover_and_leaderboard_row(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = _write_harness_run(
        results / "evaluate_google_llm" / "20260501_120000" / "exp_a",
        name="exp_a",
        dataset="tweet_eval_irony",
        f1=0.9,
    )
    discovered = discover_all_runs(results)
    assert len(discovered) == 1
    rec = extract_run_record(discovered[0])
    assert rec is not None
    assert rec.row["dataset_name"] == "tweet_eval_irony"
    assert rec.row["f1_macro"] == pytest.approx(0.9)

    df = build_leaderboard_df(results)
    assert len(df) == 1
    assert df.iloc[0]["has_predictions"] is True


def test_prune_suite_parent_with_children(tmp_path: Path) -> None:
    results = tmp_path / "results"
    campaign = results / "prompt_eng" / "20260502_100000"
    _write_harness_run(
        campaign / "multilabel_confusion_probe" / "child_run",
        name="child_run",
        dataset="pubmed_20k_rct",
        f1=0.8,
    )
    (campaign / "summary.json").write_text(
        json.dumps({"planned": [{"output_dir": "x"}], "results": []}),
        encoding="utf-8",
    )
    discovered = discover_all_runs(results)
    rels = {d.rel_dir for d in discovered}
    assert any("child_run" in r for r in rels)
    assert not any(r.endswith("20260502_100000") for r in rels)
