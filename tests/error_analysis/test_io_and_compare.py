from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.error_analysis.compare import build_comparison_df, disagreements, pairwise_agreement_matrix
from src.error_analysis.io import LoadedExperiment, discover_experiments, load_experiment


def _write_exp(dir_: Path, *, exp_id: str, sample_ids: list[str], preds: list[str]) -> Path:
    p = dir_ / exp_id
    p.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "dataset_name": ["toy"] * len(sample_ids),
            "text": [f"t{i}" for i in range(len(sample_ids))],
            "true_label": ["A"] * len(sample_ids),
            "pred_label": preds,
        }
    )
    df.to_csv(p / "predictions.csv", index=False)
    (p / "report.json").write_text('{"dataset_name":"toy","metrics":{"f1_macro":0.1,"accuracy":0.2}}', encoding="utf-8")
    return p


def test_discover_experiments_run_dir_and_exp_dir(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _ = _write_exp(run, exp_id="e1", sample_ids=["s1"], preds=["A"])
    _ = _write_exp(run, exp_id="e2", sample_ids=["s2"], preds=["B"])

    exp_dirs = discover_experiments([str(run)])
    assert {p.name for p in exp_dirs} == {"e1", "e2"}

    exp_dirs2 = discover_experiments([str(run / "e1")])
    assert [p.name for p in exp_dirs2] == ["e1"]


def test_load_experiment_missing_files_warns(tmp_path: Path) -> None:
    exp = tmp_path / "e"
    exp.mkdir()
    loaded = load_experiment(exp)
    assert loaded.predictions_df is None
    assert any("Missing predictions.csv" in w for w in loaded.warnings)


def test_build_comparison_and_disagreements_and_agreement(tmp_path: Path) -> None:
    run = tmp_path / "run"
    e1 = _write_exp(run, exp_id="a", sample_ids=["s1", "s2", "s3"], preds=["A", "A", "B"])
    # Make sure there is a real disagreement on s3 (a predicts B, b predicts A).
    e2 = _write_exp(run, exp_id="b", sample_ids=["s2", "s3", "s4"], preds=["A", "A", "B"])

    la = LoadedExperiment(exp_id="a", path=e1, predictions_df=pd.read_csv(e1 / "predictions.csv"), report=None, config=None, warnings=[], errors=[])
    lb = LoadedExperiment(exp_id="b", path=e2, predictions_df=pd.read_csv(e2 / "predictions.csv"), report=None, config=None, warnings=[], errors=[])

    cmp = build_comparison_df([la, lb], join="inner")
    assert set(cmp["sample_id"]) == {"s2", "s3"}
    assert "pred_label__a" in cmp.columns
    assert "pred_label__b" in cmp.columns

    d = disagreements(cmp)
    assert set(d["sample_id"]) == {"s3"}

    m = pairwise_agreement_matrix(cmp)
    assert m.loc["a", "b"] < 1.0

