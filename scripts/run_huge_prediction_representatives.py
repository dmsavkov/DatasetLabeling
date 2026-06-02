# pyright: basic
"""
Huge prediction over representative samples (centroids + expanded pool, shuffled batches).

See docs/gepa_pipeline.md for full multi-dataset spec.

Example (pubmed — defaults OK):
  uv run python scripts/run_huge_prediction_representatives.py --dataset pubmed_20k_rct

Example (2-class — use suggested grid):
  uv run python scripts/run_huge_prediction_representatives.py \\
    --dataset implicit_hate --prediction-size 400 --n-centroids-per-label 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from src.data_selection.gepa_pipeline_params import load_and_suggest
from src.data_selection.huge_prediction_representatives import (
    run_huge_prediction_representatives_sync,
    utc_stamp,
)
from src.experiments.suites.extended_suite import DATASETS
from src.utils.env import load_dotenv_if_present

load_dotenv_if_present()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dataset_folder(name: str) -> str:
    for ds in DATASETS:
        if ds.dataset_name == name:
            return ds.folder_name
    raise ValueError(f"Unknown dataset: {name}")


def _default_train_parquet(folder: str) -> Path:
    return _repo_root() / "data" / "processed" / folder / "train_seed" / "tier_5000" / "samples.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Huge prediction pass over representative samples (centroids + expanded pool).",
    )
    _ = ap.add_argument("--dataset", type=str, required=True)
    _ = ap.add_argument("--train-parquet", type=Path, default=None)
    _ = ap.add_argument("--output-dir", type=Path, default=None)
    _ = ap.add_argument("--model-id", type=str, default="gemma-4-31b-it")
    _ = ap.add_argument("--prompt-id", type=str, default=None)
    _ = ap.add_argument("--batch-size", type=int, default=10)
    _ = ap.add_argument("--max-concurrency", type=int, default=5)
    _ = ap.add_argument("--n-centroids-per-label", type=int, default=20)
    _ = ap.add_argument("--prediction-size", type=int, default=500)
    _ = ap.add_argument("--pool-size", type=int, default=5000)
    _ = ap.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    _ = ap.add_argument("--seed", type=int, default=42)
    _ = ap.add_argument("--max-embedding-pool-rows", type=int, default=None)
    _ = ap.add_argument(
        "--thinking-level",
        type=str,
        default="high",
        choices=("off", "low", "high"),
        help="Gemini/Gemma thinking (default high, matches gemma4_31b_think_high eval).",
    )
    _ = ap.add_argument(
        "--include-thoughts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include thought parts in API response (default on for baseline parity).",
    )
    _ = ap.add_argument(
        "--no-few-shot",
        action="store_true",
        help="Disable few-shot examples (default: train_seed/tier_<few-shot-tier>).",
    )
    _ = ap.add_argument(
        "--few-shot-tier",
        type=int,
        default=10,
        help="train_seed tier for few-shot (default 10, matches gemma eval).",
    )
    _ = ap.add_argument(
        "--few-shot-parquet",
        type=Path,
        default=None,
        help="Override few-shot source parquet.",
    )
    _ = ap.add_argument(
        "--few-shot-n",
        type=int,
        default=None,
        help="Max few-shot examples (default: all rows in tier file, usually 10).",
    )
    args = ap.parse_args()

    dataset_name = str(args.dataset).strip()
    folder = _dataset_folder(dataset_name)
    train_p = args.train_parquet or _default_train_parquet(folder)
    if not train_p.exists():
        raise FileNotFoundError(f"Train parquet not found: {train_p}")

    ctx, defaults = load_and_suggest(dataset_name)
    logger.info(
        "Dataset {} ({} labels). Suggested huge-pred defaults: prediction_size={} "
        "batch_size={} n_centroids_per_label={} — {}",
        dataset_name,
        ctx.n_labels,
        defaults.prediction_size,
        defaults.batch_size,
        defaults.n_centroids_per_label,
        defaults.note,
    )

    out_dir = args.output_dir or (
        _repo_root() / "data" / "huge_prediction_representatives" / dataset_name / utc_stamp()
    )

    logger.info("Huge prediction for {} → {}", dataset_name, out_dir)
    result = run_huge_prediction_representatives_sync(
        dataset_name=dataset_name,
        train_parquet=train_p,
        output_dir=out_dir,
        model_id=str(args.model_id),
        prompt_id=args.prompt_id,
        batch_size=int(args.batch_size),
        max_concurrency=int(args.max_concurrency),
        n_centroids_per_label=int(args.n_centroids_per_label),
        prediction_size=int(args.prediction_size),
        pool_size=int(args.pool_size),
        embedding_model=str(args.embedding_model),
        seed=int(args.seed),
        max_embedding_pool_rows=args.max_embedding_pool_rows,
        few_shot_tier=int(args.few_shot_tier),
        few_shot_parquet=args.few_shot_parquet,
        few_shot_n=args.few_shot_n,
        use_few_shot=not bool(args.no_few_shot),
        thinking_level=args.thinking_level,  # type: ignore[arg-type]
        include_thoughts=bool(args.include_thoughts),
    )
    print(json.dumps({"output_dir": str(result.output_dir), "manifest": result.manifest}, indent=2))


if __name__ == "__main__":
    main()
