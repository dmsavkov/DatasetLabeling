# pyright: basic
"""In-memory experiment matrix for prompt-eng (no YAML files required)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from src.experiments.config import ExperimentConfig, ThinkingLevelLiteral

ExperimentKind = Literal["multilabel_confusion_probe", "self_debate"]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    dataset_name: str
    folder_name: str


@dataclass(frozen=True, slots=True)
class PromptEngModelSpec:
    slug: str
    model_id: str
    thinking_level: ThinkingLevelLiteral
    include_thoughts: bool


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(dataset_name="banking-10", folder_name="banking-10"),
    DatasetSpec(dataset_name="tweet_eval_irony", folder_name="tweet_eval_irony"),
    DatasetSpec(dataset_name="implicit_hate", folder_name="implicit_hate"),
    DatasetSpec(dataset_name="pubmed_20k_rct", folder_name="pubmed_20k_rct"),
)

DEFAULT_MODEL_MATRIX: tuple[PromptEngModelSpec, ...] = (
    PromptEngModelSpec("gemini31_flash_think_low", "gemini-3.1-flash-lite-preview", "low", True),
    PromptEngModelSpec("gemini31_flash_think_high", "gemini-3.1-flash-lite-preview", "high", True),
    PromptEngModelSpec("gemma4_31b_think_high", "gemma-4-31b-it", "high", True),
)

EXPERIMENT_KINDS: tuple[ExperimentKind, ...] = ("multilabel_confusion_probe", "self_debate")


def _train_path(ds: DatasetSpec) -> str:
    return f"data/processed/{ds.folder_name}/train_seed/tier_10/samples.parquet"


def _test_path(ds: DatasetSpec, *, tier: int) -> str:
    return f"data/processed/{ds.folder_name}/test/tier_{int(tier)}/samples.parquet"


def _slug_ds(ds: DatasetSpec) -> str:
    return ds.dataset_name.replace("-", "")


def _config_name(kind: ExperimentKind, ms: PromptEngModelSpec, ds: DatasetSpec, *, test_tier: int) -> str:
    kind_short = "multilabel" if kind == "multilabel_confusion_probe" else "self_debate"
    return f"{kind_short}_{ms.slug}_{_slug_ds(ds)}_test{int(test_tier)}"


def _safe_name_segment(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_")


def _model_params(ms: PromptEngModelSpec) -> dict[str, Any]:
    return {
        "model_id": ms.model_id,
        "max_concurrency": 5,
        "max_tokens": None,
        "retries": 20,
        "thinking_level": ms.thinking_level,
        "include_thoughts": ms.include_thoughts,
    }


def build_prompt_eng_config(
    *,
    kind: ExperimentKind,
    dataset: DatasetSpec,
    model: PromptEngModelSpec,
    test_tier: int,
    seed: int = 42,
    output_dir: str | None = None,
) -> ExperimentConfig:
    train = _train_path(dataset)
    test = _test_path(dataset, tier=test_tier)
    name = _config_name(kind, model, dataset, test_tier=test_tier)
    slug = kind
    out = output_dir or f"results/prompt_eng/{slug}/{name}"

    if kind == "multilabel_confusion_probe":
        params = {"batch_size": 3, "temperature": 0.0, **_model_params(model)}
        model_block = {"kind": "multilabel_confusion_probe", "params": params}
    else:
        params = {
            "batch_size": 3,
            "temperature_a": 0.0,
            "pass_b_temperature": 0.5,
            **_model_params(model),
        }
        model_block = {"kind": "self_debate", "params": params}

    return ExperimentConfig.model_validate(
        {
            "name": name,
            "seed": int(seed),
            "train_data": train,
            "test_data": test,
            "output_dir": out,
            "model": model_block,
        }
    )


def build_prompt_eng_configs(
    *,
    test_tier: int = 200,
    seed: int = 42,
    datasets: tuple[DatasetSpec, ...] | None = None,
    models: tuple[PromptEngModelSpec, ...] | None = None,
    kinds: tuple[ExperimentKind, ...] | None = None,
) -> list[ExperimentConfig]:
    ds_list = datasets if datasets is not None else DATASETS
    ms_list = models if models is not None else DEFAULT_MODEL_MATRIX
    kind_list = kinds if kinds is not None else EXPERIMENT_KINDS
    out: list[ExperimentConfig] = []
    for ds in ds_list:
        for ms in ms_list:
            for kind in kind_list:
                out.append(
                    build_prompt_eng_config(
                        kind=kind,
                        dataset=ds,
                        model=ms,
                        test_tier=test_tier,
                        seed=seed,
                    )
                )
    return out


def filter_datasets(names: set[str]) -> tuple[DatasetSpec, ...]:
    if not names:
        return DATASETS
    return tuple(d for d in DATASETS if d.dataset_name in names)


def filter_models(
    *,
    model_ids: set[str],
    thinking_levels: set[str],
) -> tuple[PromptEngModelSpec, ...]:
    out: list[PromptEngModelSpec] = []
    for m in DEFAULT_MODEL_MATRIX:
        if model_ids and m.model_id not in model_ids:
            continue
        if thinking_levels and m.thinking_level not in thinking_levels:
            continue
        out.append(m)
    return tuple(out)


def filter_kinds(names: set[str]) -> tuple[ExperimentKind, ...]:
    if not names:
        return EXPERIMENT_KINDS
    aliases = {
        "multilabel": "multilabel_confusion_probe",
        "multilabel_confusion_probe": "multilabel_confusion_probe",
        "self_debate": "self_debate",
        "sdr": "self_debate",
    }
    resolved: list[ExperimentKind] = []
    for n in names:
        k = aliases.get(n.strip().lower())
        if k and k not in resolved:
            resolved.append(k)  # type: ignore[arg-type]
    return tuple(resolved) if resolved else EXPERIMENT_KINDS
