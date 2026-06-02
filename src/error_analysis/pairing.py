# pyright: basic
"""Pair prompt_eng runs with evaluate_google_llm baselines on the same dataset/model/tier."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.error_analysis.compare import build_comparison_df
from src.error_analysis.io import LoadedExperiment


def _key_tuple(meta: dict[str, Any]) -> tuple[Any, ...]:
    return (
        meta.get("dataset_name"),
        meta.get("model_id"),
        meta.get("thinking_level"),
        meta.get("tier_size"),
    )


def _is_eval_baseline(e: LoadedExperiment) -> bool:
    return e.meta.get("model_kind") == "google_genai_chat" and e.meta.get("series") == "evaluate_google_llm"


def _is_prompt_eng(e: LoadedExperiment) -> bool:
    return e.meta.get("series") == "prompt_eng" and e.meta.get("model_kind") in (
        "multilabel_confusion_probe",
        "self_debate",
    )


def _pick_eval_baseline(
    candidates: list[LoadedExperiment],
    *,
    preferred_batch_size: int,
) -> LoadedExperiment | None:
    if not candidates:
        return None
    for e in candidates:
        if e.meta.get("batch_size") == preferred_batch_size:
            return e
    return max(
        candidates,
        key=lambda e: float((e.report or {}).get("metrics", {}).get("f1_macro") or -1.0),
    )


def compare_prompt_eng_vs_google_eval(
    exps: list[LoadedExperiment],
    *,
    preferred_batch_size: int = 10,
) -> pd.DataFrame:
    """
    Side-by-side metrics for each prompt_eng run vs the closest matching
    ``evaluate_google_llm`` baseline (same dataset, model_id, thinking, tier).
    """
    eval_by_key: dict[tuple[Any, ...], list[LoadedExperiment]] = {}
    prompt_eng: list[LoadedExperiment] = []

    for e in exps:
        if e.predictions_df is None or not e.report:
            continue
        k = _key_tuple(e.meta)
        if _is_eval_baseline(e):
            eval_by_key.setdefault(k, []).append(e)
        elif _is_prompt_eng(e):
            prompt_eng.append(e)

    rows: list[dict[str, object]] = []
    for pe in prompt_eng:
        k = _key_tuple(pe.meta)
        eval_candidates = eval_by_key.get(k, [])
        ev = _pick_eval_baseline(eval_candidates, preferred_batch_size=preferred_batch_size)
        pe_m = (pe.report or {}).get("metrics", {})
        row: dict[str, object] = {
            "dataset_name": pe.meta.get("dataset_name"),
            "model_id": pe.meta.get("model_id"),
            "thinking_level": pe.meta.get("thinking_level"),
            "tier_size": pe.meta.get("tier_size"),
            "prompt_eng_exp_id": pe.exp_id,
            "prompt_eng_kind": pe.meta.get("model_kind"),
            "prompt_eng_campaign": pe.meta.get("campaign"),
            "prompt_eng_f1": pe_m.get("f1_macro"),
            "prompt_eng_accuracy": pe_m.get("accuracy"),
            "eval_matched": ev is not None,
        }
        if ev is not None:
            ev_m = (ev.report or {}).get("metrics", {})
            row.update(
                {
                    "eval_exp_id": ev.exp_id,
                    "eval_batch_size": ev.meta.get("batch_size"),
                    "eval_f1": ev_m.get("f1_macro"),
                    "eval_accuracy": ev_m.get("accuracy"),
                    "delta_f1": (
                        float(pe_m.get("f1_macro")) - float(ev_m.get("f1_macro"))
                        if pe_m.get("f1_macro") is not None and ev_m.get("f1_macro") is not None
                        else None
                    ),
                    "delta_accuracy": (
                        float(pe_m.get("accuracy")) - float(ev_m.get("accuracy"))
                        if pe_m.get("accuracy") is not None and ev_m.get("accuracy") is not None
                        else None
                    ),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["dataset_name", "model_id", "prompt_eng_kind", "prompt_eng_exp_id"],
        na_position="last",
    )


def build_paired_row_comparison(
    prompt_eng_exp: LoadedExperiment,
    eval_exp: LoadedExperiment,
) -> pd.DataFrame:
    """Row-level join for one prompt_eng run vs one evaluate_google_llm baseline."""
    return build_comparison_df([eval_exp, prompt_eng_exp], join="inner")
