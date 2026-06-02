from __future__ import annotations

from src.experiments.suites.prompt_eng_suite import (
    DEFAULT_MODEL_MATRIX,
    build_prompt_eng_configs,
    filter_datasets,
    filter_kinds,
    filter_models,
)


def test_prompt_eng_full_matrix_count() -> None:
    cfgs = build_prompt_eng_configs(test_tier=200)
    # 4 datasets × 3 model specs (gemini low/high + gemma high) × 2 kinds
    assert len(cfgs) == 24
    kinds = {c.model.kind for c in cfgs}
    assert kinds == {"multilabel_confusion_probe", "self_debate"}


def test_gemini_has_low_and_high() -> None:
    levels = {m.thinking_level for m in DEFAULT_MODEL_MATRIX if m.model_id == "gemini-3.1-flash-lite-preview"}
    assert levels == {"low", "high"}


def test_tactical_filters() -> None:
    ds = filter_datasets({"banking-10"})
    ms = filter_models(model_ids={"gemini-3.1-flash-lite-preview"}, thinking_levels={"high"})
    kinds = filter_kinds({"multilabel"})
    cfgs = build_prompt_eng_configs(test_tier=20, datasets=ds, models=ms, kinds=kinds)
    assert len(cfgs) == 1
    assert cfgs[0].name.endswith("_test20")
    assert cfgs[0].model.kind == "multilabel_confusion_probe"
    assert cfgs[0].model.params.thinking_level == "high"
