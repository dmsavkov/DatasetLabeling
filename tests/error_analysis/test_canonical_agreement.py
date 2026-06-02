# pyright: basic
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.error_analysis.compare import build_comparison_df, pairwise_agreement_matrix
from src.error_analysis.io import LoadedExperiment
from src.error_analysis.row_metrics import add_model_correctness_flags


def _write_pubmed_exp(dir_: Path, *, exp_id: str, preds: list[str]) -> Path:
    p = dir_ / exp_id
    p.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(len(preds))],
            "dataset_name": ["pubmed_20k_rct"] * len(preds),
            "text": ["t"] * len(preds),
            "true_label": ["2", "4", "0"],
            "pred_label": preds,
        }
    )
    df.to_csv(p / "predictions.csv", index=False)
    return p


def test_pairwise_agreement_canonicalizes_pubmed_ids_and_names(tmp_path: Path) -> None:
    run = tmp_path / "run"
    e1 = _write_pubmed_exp(run, exp_id="names", preds=["methods", "conclusions", "background"])
    e2 = _write_pubmed_exp(run, exp_id="ids", preds=["2", "4", "0"])

    la = LoadedExperiment(
        exp_id="names",
        path=e1,
        predictions_df=pd.read_csv(e1 / "predictions.csv"),
        report=None,
        config=None,
        warnings=[],
        errors=[],
    )
    lb = LoadedExperiment(
        exp_id="ids",
        path=e2,
        predictions_df=pd.read_csv(e2 / "predictions.csv"),
        report=None,
        config=None,
        warnings=[],
        errors=[],
    )

    cmp = build_comparison_df([la, lb])
    m = pairwise_agreement_matrix(cmp, dataset_name="pubmed_20k_rct")
    assert float(m.loc["names", "ids"]) == 1.0

    flagged = add_model_correctness_flags(cmp, dataset_name="pubmed_20k_rct")
    assert flagged["is_correct__names"].tolist() == [True, True, True]
    assert flagged["is_correct__ids"].tolist() == [True, True, True]


def test_pairwise_agreement_single_row_id_vs_name(tmp_path: Path) -> None:
    run = tmp_path / "run"
    e1 = _write_pubmed_exp(run, exp_id="a", preds=["2", "4", "0"])
    e2 = _write_pubmed_exp(run, exp_id="b", preds=["methods", "conclusions", "background"])
    la = LoadedExperiment(
        exp_id="a",
        path=e1,
        predictions_df=pd.read_csv(e1 / "predictions.csv"),
        report=None,
        config=None,
        warnings=[],
        errors=[],
    )
    lb = LoadedExperiment(
        exp_id="b",
        path=e2,
        predictions_df=pd.read_csv(e2 / "predictions.csv"),
        report=None,
        config=None,
        warnings=[],
        errors=[],
    )
    cmp = build_comparison_df([la, lb])
    # Legacy string compare would be 0; canonical should be 1.
    m = pairwise_agreement_matrix(cmp, dataset_name="pubmed_20k_rct")
    assert float(m.loc["a", "b"]) == 1.0
