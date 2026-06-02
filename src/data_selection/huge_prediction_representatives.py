# pyright: basic
"""
Huge prediction over representative samples: K-means centroids, expand to a fixed
prediction pool, then shuffled sequential batch LLM calls (aligned with Gemma eval).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.data_selection.few_shot import DEFAULT_FEW_SHOT_TIER, load_eval_aligned_few_shot
from src.data_selection.gepa_optimizer_sets import (
    GOLDEN_TEST_TIERS,
    collect_golden_test_sample_ids,
    contrastive_neighbors_for_errors,
    expand_prediction_pool,
    stratified_subset_df,
    utc_stamp,
)
from src.data_selection.gepa_pipeline_params import validate_or_raise
from src.data_selection.label_utils import DatasetContext, load_dataset_context
from src.datasets.schema import SCHEMA, stable_sort_for_determinism, validate_processed_samples_df
from src.models.llm.google_genai_batch import GoogleGenaiBatchParams, GoogleGenaiBatchPredictor
from src.prompts.baseline import BatchItem, normalize_label

ThinkingLevelStr = Literal["off", "low", "high"]

__all__ = [
    "HugePredictionResult",
    "run_huge_prediction_representatives",
    "run_huge_prediction_representatives_sync",
    "utc_stamp",
]


@dataclass(frozen=True, slots=True)
class HugePredictionResult:
    output_dir: Path
    manifest: dict[str, Any]
    centroids_df: pd.DataFrame
    prediction_pool_df: pd.DataFrame
    predictions_df: pd.DataFrame
    batch_log_df: pd.DataFrame
    contrastive_df: pd.DataFrame


def shuffle_prediction_pool(df: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Break label-clustered row order before batching (same idea as ``run._shuffle_llm_df``)."""
    return df.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)


def build_shuffled_batches(df: pd.DataFrame, *, batch_size: int) -> list[tuple[int, pd.DataFrame]]:
    """Consecutive chunks after shuffle; each batch uses item ids ``0`` … ``batch_size-1``."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches: list[tuple[int, pd.DataFrame]] = []
    n = len(df)
    for start in range(0, n, batch_size):
        chunk = df.iloc[start : start + batch_size].reset_index(drop=True)
        if chunk.empty:
            continue
        batches.append((len(batches), chunk))
    return batches


async def _run_shuffled_batch_predictions(
    df: pd.DataFrame,
    ctx: DatasetContext,
    *,
    model_id: str,
    batch_size: int,
    max_concurrency: int,
    seed: int,
    few_shot: list[tuple[str, str]] | None,
    thinking_level: ThinkingLevelStr,
    include_thoughts: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = shuffle_prediction_pool(df, seed=int(seed))
    batches = build_shuffled_batches(shuffled, batch_size=int(batch_size))

    params = GoogleGenaiBatchParams(
        model_id=model_id,
        prompt_id=ctx.prompt_id,
        few_shot=few_shot,
        batch_size=int(batch_size),
        max_concurrency=int(max_concurrency),
        thinking_level=thinking_level,
        include_thoughts=bool(include_thoughts),
    )
    predictor = GoogleGenaiBatchPredictor(params=params)

    pred_by_sample_id: dict[str, str | None] = {}
    batch_records: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def run_one(batch_id: int, chunk: pd.DataFrame) -> None:
        items = [
            BatchItem(id=str(i), text=str(chunk[SCHEMA.text].iloc[i])) for i in range(len(chunk))
        ]
        sample_ids = chunk[SCHEMA.sample_id].astype(str).tolist()
        true_ids = chunk[SCHEMA.true_label].astype(str).tolist()

        async with sem:
            preds_map = await predictor._apredict_one_batch(items, allowed_labels=ctx.label_ids)

        gold_labels: list[str] = []
        pred_labels: list[str] = []
        correct_n = 0
        prompt_toks = 0
        out_toks = 0
        elapsed = 0.0

        for i, sid in enumerate(sample_ids):
            true_id = true_ids[i]
            pred = preds_map.get(str(i))
            pred_raw = pred.pred_label if pred is not None else None
            pred_id = (
                normalize_label(str(pred_raw), ctx.label_ids) if pred_raw is not None else None
            )
            pred_by_sample_id[sid] = pred_id
            gold_labels.append(str(true_id))
            pred_labels.append(str(pred_id) if pred_id is not None else "")
            if ctx.labels_match(true_id, pred_id):
                correct_n += 1
            if pred is not None and pred.raw:
                elapsed = max(elapsed, float(pred.raw.get("elapsed_s", 0.0)))
                um = pred.raw.get("usage_metadata") or {}
                prompt_toks += int(um.get("prompt_token_count") or 0)
                out_toks += int(um.get("candidates_token_count") or 0) + int(um.get("thoughts_token_count") or 0)

        batch_records.append(
            {
                "batch_id": int(batch_id),
                "batch_size": int(len(chunk)),
                "sample_ids": sample_ids,
                "gold_labels": gold_labels,
                "pred_labels": pred_labels,
                "batch_accuracy": float(correct_n) / float(len(chunk)) if len(chunk) else 0.0,
                "correct_count": int(correct_n),
                "prompt_tokens": int(prompt_toks),
                "output_tokens": int(out_toks),
                "elapsed_s": float(elapsed),
            }
        )
        logger.info(
            "batch {}/{} acc={:.2f} n={} sample_ids={}",
            len(batch_records),
            len(batches),
            batch_records[-1]["batch_accuracy"],
            len(chunk),
            sample_ids,
        )

    tasks = [run_one(bid, chunk) for bid, chunk in batches]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM batches"):
        await coro

    out = df.reset_index(drop=True).copy()
    out["pred_label"] = out[SCHEMA.sample_id].astype(str).map(lambda s: pred_by_sample_id.get(s))
    out["correct"] = [
        ctx.labels_match(out[SCHEMA.true_label].iloc[i], out["pred_label"].iloc[i])
        for i in range(len(out))
    ]
    batch_log_df = pd.DataFrame(batch_records).sort_values("batch_id").reset_index(drop=True)
    return stable_sort_for_determinism(out), batch_log_df


async def run_huge_prediction_representatives(
    *,
    dataset_name: str,
    train_parquet: Path,
    output_dir: Path,
    model_id: str = "gemma-4-31b-it",
    prompt_id: str | None = None,
    batch_size: int = 10,
    max_concurrency: int = 5,
    n_centroids_per_label: int = 20,
    prediction_size: int = 500,
    pool_size: int = 5000,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    seed: int = 42,
    max_embedding_pool_rows: int | None = None,
    processed_root_path: Path | None = None,
    few_shot_tier: int = DEFAULT_FEW_SHOT_TIER,
    few_shot_parquet: Path | None = None,
    few_shot_n: int | None = None,
    use_few_shot: bool = True,
    thinking_level: ThinkingLevelStr = "high",
    include_thoughts: bool = True,
) -> HugePredictionResult:
    ctx = load_dataset_context(
        dataset_name,
        prompt_id=prompt_id,
        processed_root_path=processed_root_path,
    )
    n_centroids, samples_per_centroid = validate_or_raise(
        ctx,
        prediction_size=int(prediction_size),
        batch_size=int(batch_size),
        n_centroids_per_label=int(n_centroids_per_label),
    )

    source_df = pd.read_parquet(train_parquet)
    validate_processed_samples_df(source_df)
    source_df = stable_sort_for_determinism(source_df)

    golden_ids = collect_golden_test_sample_ids(dataset_name, root=processed_root_path)
    embedding_pool = source_df[~source_df[SCHEMA.sample_id].astype(str).isin(golden_ids)].copy()
    rows_after_golden = int(len(embedding_pool))

    if int(pool_size) > 0 and rows_after_golden > int(pool_size):
        embedding_pool = stratified_subset_df(
            embedding_pool,
            n_total=int(pool_size),
            labels=ctx.label_ids,
            seed=int(seed),
        )
    if max_embedding_pool_rows is not None:
        embedding_pool = embedding_pool.head(int(max_embedding_pool_rows)).copy()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Centroids: {} per label × {} labels = {}; {} samples/centroid → {} predictions",
        n_centroids_per_label,
        ctx.n_labels,
        n_centroids,
        samples_per_centroid,
        prediction_size,
    )

    centroids_df, prediction_pool = expand_prediction_pool(
        embedding_pool,
        label_ids=ctx.label_ids,
        n_centroids_per_label=int(n_centroids_per_label),
        prediction_size=int(prediction_size),
        embedding_model=embedding_model,
        seed=int(seed),
    )
    centroids_df.to_parquet(out_dir / "centroid_samples.parquet", index=False)
    prediction_pool.to_parquet(out_dir / "prediction_pool.parquet", index=False)

    pool_ids = frozenset(prediction_pool[SCHEMA.sample_id].astype(str).tolist())
    few_shot_examples: list[tuple[str, str]] = []
    few_shot_path: Path | None = None
    if use_few_shot:
        few_shot_examples, few_shot_path = load_eval_aligned_few_shot(
            ctx,
            seed=int(seed),
            few_shot_parquet=few_shot_parquet,
            tier=int(few_shot_tier),
            n=few_shot_n if few_shot_n is not None else DEFAULT_FEW_SHOT_TIER,
            exclude_sample_ids=pool_ids,
            processed_root_path=processed_root_path,
        )

    logger.info(
        "LLM: batch_size={} thinking_level={} include_thoughts={} few_shot={} allowed_labels={}",
        batch_size,
        thinking_level,
        include_thoughts,
        len(few_shot_examples),
        ctx.label_ids,
    )

    predictions_df, batch_log_df = await _run_shuffled_batch_predictions(
        prediction_pool,
        ctx,
        model_id=model_id,
        batch_size=int(batch_size),
        max_concurrency=int(max_concurrency),
        seed=int(seed),
        few_shot=few_shot_examples if few_shot_examples else None,
        thinking_level=thinking_level,
        include_thoughts=include_thoughts,
    )
    predictions_df.to_parquet(out_dir / "predictions.parquet", index=False)
    batch_log_df.to_parquet(out_dir / "prediction_batch_log.parquet", index=False)

    errors_df = predictions_df[predictions_df["correct"] == False].copy()  # noqa: E712
    contrastive_df = contrastive_neighbors_for_errors(
        errors_df,
        embedding_pool,
        embedding_model=embedding_model,
        focus_label=None,
    )
    contrastive_df.to_parquet(out_dir / "contrastive_edges.parquet", index=False)

    overall_acc = float(predictions_df["correct"].mean()) if len(predictions_df) else 0.0
    per_label_acc = {
        lab: float(
            predictions_df[predictions_df[SCHEMA.true_label].astype(str) == lab]["correct"].mean()
        )
        if len(predictions_df[predictions_df[SCHEMA.true_label].astype(str) == lab])
        else 0.0
        for lab in ctx.label_ids
    }

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "huge_prediction_representatives",
        "dataset_name": dataset_name,
        "train_parquet": str(train_parquet),
        "model_id": model_id,
        "prompt_id": ctx.prompt_id,
        "label_ids": ctx.label_ids,
        "prompt_labels": ctx.prompt_labels,
        "n_labels": ctx.n_labels,
        "batch_size": int(batch_size),
        "batching": "shuffled_sequential",
        "thinking_level": thinking_level,
        "include_thoughts": bool(include_thoughts),
        "n_centroids_per_label": int(n_centroids_per_label),
        "n_centroids": int(n_centroids),
        "samples_per_centroid": int(samples_per_centroid),
        "prediction_size": int(prediction_size),
        "n_batches": int(len(batch_log_df)),
        "pool_size": int(pool_size),
        "embedding_model": embedding_model,
        "seed": int(seed),
        "golden_test_tiers_excluded": list(GOLDEN_TEST_TIERS),
        "embedding_pool_rows": int(len(embedding_pool)),
        "centroid_rows": int(len(centroids_df)),
        "prediction_pool_rows": int(len(prediction_pool)),
        "prediction_rows": int(len(predictions_df)),
        "error_rows": int(len(errors_df)),
        "contrastive_edge_rows": int(len(contrastive_df)),
        "overall_accuracy": overall_acc,
        "per_label_accuracy": per_label_acc,
        "mean_batch_accuracy": float(batch_log_df["batch_accuracy"].mean()) if len(batch_log_df) else 0.0,
        "few_shot_enabled": bool(use_few_shot and few_shot_examples),
        "few_shot_count": len(few_shot_examples),
        "few_shot_parquet": str(few_shot_path) if few_shot_path is not None else None,
        "few_shot_tier": int(few_shot_tier) if use_few_shot else None,
        "few_shot_style": "eval_aligned_card_ids" if use_few_shot else None,
        "allowed_labels": ctx.label_ids,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return HugePredictionResult(
        output_dir=out_dir,
        manifest=manifest,
        centroids_df=centroids_df,
        prediction_pool_df=prediction_pool,
        predictions_df=predictions_df,
        batch_log_df=batch_log_df,
        contrastive_df=contrastive_df,
    )


def run_huge_prediction_representatives_sync(**kwargs: Any) -> HugePredictionResult:
    return asyncio.run(run_huge_prediction_representatives(**kwargs))
