# pyright: basic
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.error_analysis.io import discover_experiments, load_experiment
from src.error_analysis.pairing import compare_prompt_eng_vs_google_eval


def _write_exp(
    base: Path,
    *,
    name: str,
    dataset: str,
    model_id: str,
    kind: str,
    f1: float,
) -> None:
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "dataset_name": dataset,
                "text": "hello",
                "true_label": "0",
                "pred_label": "0",
                "correct": True,
            }
        ]
    ).to_csv(base / "predictions.csv", index=False)
    (base / "report.json").write_text(
        json.dumps(
            {
                "dataset_name": dataset,
                "tier_size": 200,
                "predictor_name": kind,
                "metrics": {"f1_macro": f1, "accuracy": f1},
            }
        ),
        encoding="utf-8",
    )
    (base / "config.resolved.json").write_text(
        json.dumps(
            {
                "name": name,
                "test_data": f"data/processed/{dataset}/test/tier_200/samples.parquet",
                "model": {
                    "kind": kind,
                    "params": {
                        "model_id": model_id,
                        "thinking_level": "high",
                        "batch_size": 10,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_discover_nested_prompt_eng(tmp_path: Path) -> None:
    root = tmp_path / "results" / "prompt_eng" / "20260519_134732"
    _write_exp(
        root / "multilabel_confusion_probe" / "multilabel_gemini31_flash_test200",
        name="multilabel_gemini31_flash_test200",
        dataset="implicit_hate",
        model_id="gemini-3.1-flash-lite-preview",
        kind="multilabel_confusion_probe",
        f1=0.75,
    )
    found = discover_experiments([str(root)])
    assert len(found) == 1
    assert found[0].name == "multilabel_gemini31_flash_test200"


def test_compare_prompt_eng_vs_eval(tmp_path: Path) -> None:
    pe_root = tmp_path / "results" / "prompt_eng" / "20260519_134732"
    ev_root = tmp_path / "results" / "evaluate_google_llm" / "20260519_191321"
    _write_exp(
        pe_root / "multilabel_confusion_probe" / "multilabel_x",
        name="multilabel_x",
        dataset="pubmed_20k_rct",
        model_id="gemma-4-31b-it",
        kind="multilabel_confusion_probe",
        f1=0.80,
    )
    _write_exp(
        ev_root / "google_genai_gemma4_bs10",
        name="google_genai_gemma4_bs10",
        dataset="pubmed_20k_rct",
        model_id="gemma-4-31b-it",
        kind="google_genai_chat",
        f1=0.77,
    )
    from src.error_analysis.io import load_many

    exps, _ = load_many([str(pe_root), str(ev_root)])
    cmp_df = compare_prompt_eng_vs_google_eval(exps)
    assert len(cmp_df) == 1
    assert bool(cmp_df.iloc[0]["eval_matched"])
    assert float(cmp_df.iloc[0]["delta_f1"]) == pytest.approx(0.03)
