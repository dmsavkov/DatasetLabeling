# pyright: basic
"""Few-shot examples for LLM prompts (aligned with experiment train_seed tiers)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.data_selection.label_utils import DatasetContext
from src.datasets.io import processed_root
from src.datasets.schema import SCHEMA, stable_sort_for_determinism, validate_processed_samples_df

DEFAULT_FEW_SHOT_TIER = 10


def default_few_shot_parquet_path(
    dataset_name: str,
    *,
    tier: int = DEFAULT_FEW_SHOT_TIER,
    processed_root_path: Path | None = None,
) -> Path:
    pr = processed_root(processed_root_path)
    return pr / dataset_name / "train_seed" / f"tier_{int(tier)}" / "samples.parquet"


def few_shot_examples_from_df(
    df: pd.DataFrame,
    ctx: DatasetContext,
    *,
    n: int | None = None,
    exclude_sample_ids: frozenset[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Build ``(text, label)`` pairs for the prompt few-shot block.
    Labels are canonical **prompt** strings (e.g. PubMed names, not numeric ids).
    """
    validate_processed_samples_df(df)
    work = stable_sort_for_determinism(df.reset_index(drop=True))
    excl = exclude_sample_ids or frozenset()
    if excl:
        work = work[~work[SCHEMA.sample_id].astype(str).isin(excl)].copy()

    limit = int(n) if n is not None else len(work)
    limit = min(limit, len(work))
    out: list[tuple[str, str]] = []
    for i in range(limit):
        text = str(work[SCHEMA.text].iloc[i])
        lab = ctx.canonicalize(work[SCHEMA.true_label].iloc[i])
        out.append((text, lab))
    return out


def load_few_shot_examples(
    ctx: DatasetContext,
    *,
    few_shot_parquet: Path | None = None,
    tier: int = DEFAULT_FEW_SHOT_TIER,
    n: int | None = None,
    exclude_sample_ids: frozenset[str] | None = None,
    processed_root_path: Path | None = None,
) -> tuple[list[tuple[str, str]], Path | None]:
    """
    Load few-shot examples from ``train_seed/tier_{tier}`` (same as Gemma eval configs),
    unless ``few_shot_parquet`` is set.
    """
    path = few_shot_parquet
    if path is None:
        path = default_few_shot_parquet_path(
            ctx.dataset_name,
            tier=tier,
            processed_root_path=processed_root_path,
        )
    if not path.exists():
        logger.warning("Few-shot parquet missing: {} — running without few-shot", path)
        return [], None

    df = pd.read_parquet(path)
    examples = few_shot_examples_from_df(
        df,
        ctx,
        n=n if n is not None else DEFAULT_FEW_SHOT_TIER,
        exclude_sample_ids=exclude_sample_ids,
    )
    if not examples:
        logger.warning("No few-shot examples after exclusions from {}", path)
    else:
        logger.info("Loaded {} few-shot examples from {}", len(examples), path)
    return examples, path


def load_eval_aligned_few_shot(
    ctx: DatasetContext,
    *,
    seed: int,
    tier: int = DEFAULT_FEW_SHOT_TIER,
    n: int = DEFAULT_FEW_SHOT_TIER,
    exclude_sample_ids: frozenset[str] | None = None,
    few_shot_parquet: Path | None = None,
    processed_root_path: Path | None = None,
) -> tuple[list[tuple[str, str]], Path | None]:
    """
    Same few-shot contract as ``evaluate_google_llm`` / ``arun_experiment``:
    shuffle ``train_seed/tier_{tier}`` with ``seed``, labels are card ids (``"0"``…).
    """
    from src.experiments.run import _few_shot_from_train_df, _shuffle_llm_df

    path = few_shot_parquet or default_few_shot_parquet_path(
        ctx.dataset_name,
        tier=tier,
        processed_root_path=processed_root_path,
    )
    if not path.exists():
        logger.warning("Few-shot parquet missing: {} — running without few-shot", path)
        return [], None

    df = pd.read_parquet(path)
    validate_processed_samples_df(df)
    excl = exclude_sample_ids or frozenset()
    if excl:
        df = df[~df[SCHEMA.sample_id].astype(str).isin(excl)].copy()
    df = _shuffle_llm_df(df, seed=int(seed))
    examples = _few_shot_from_train_df(df, n=int(n))
    if not examples:
        logger.warning("No eval-aligned few-shot examples from {}", path)
    else:
        logger.info("Loaded {} eval-aligned few-shot examples from {}", len(examples), path)
    return examples, path
