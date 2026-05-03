# pyright: basic
"""
Run Google-native LLM evaluations via `google.genai` (thinking + usage_metadata).

Uses `ExperimentConfig` kind `google_genai_chat` and `arun_experiment` for consistent artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.experiments.config import ExperimentConfig
from src.experiments.run import arun_experiment
from src.utils.env import load_dotenv_if_present

load_dotenv_if_present()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True, slots=True)
class _Dataset:
    dataset_name: str
    folder_name: str


DATASETS: tuple[_Dataset, ...] = (
    _Dataset("banking-10", "banking-10"),
    _Dataset("tweet_eval_irony", "tweet_eval_irony"),
    _Dataset("implicit_hate", "implicit_hate"),
    _Dataset("pubmed_20k_rct", "pubmed_20k_rct"),
)


@dataclass(frozen=True, slots=True)
class _ModelSpec:
    slug: str
    model_id: str
    thinking_level: str  # off | low | high
    include_thoughts: bool


DEFAULT_MODEL_MATRIX: tuple[_ModelSpec, ...] = (
    _ModelSpec("gemma4_31b_think_high", "gemma-4-31b-it", "high", True),
    _ModelSpec("gemma3_4b", "gemma-3-4b-it", "off", False),
    _ModelSpec("gemma3_27b", "gemma-3-27b-it", "off", False),
    _ModelSpec("gemini31_flash_think_low", "gemini-3.1-flash-lite-preview", "low", True),
    _ModelSpec("gemini31_flash_think_high", "gemini-3.1-flash-lite-preview", "high", True),
)

# Each model × dataset run is repeated for these batch sizes (total runs × len).
BATCH_SIZES: tuple[int, ...] = (3, 5, 10)


def _train_path(ds: _Dataset) -> str:
    return f"data/processed/{ds.folder_name}/train_seed/tier_10/samples.parquet"


def _test_path(ds: _Dataset, *, test_tier: int) -> str:
    return f"data/processed/{ds.folder_name}/test/tier_{int(test_tier)}/samples.parquet"


def _safe_name_segment(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_")


async def arun_google_eval_suite(
    *,
    results_root: Path,
    test_tier: int,
    datasets: tuple[_Dataset, ...],
    models: tuple[_ModelSpec, ...],
    run: bool,
    seed: int,
) -> dict[str, Any]:
    stamp = _utc_stamp()
    out_root = results_root / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    planned: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for ds in datasets:
        train_p = _train_path(ds)
        test_p = _test_path(ds, test_tier=test_tier)
        for ms in models:
            for batch_size in BATCH_SIZES:
                exp_name = _safe_name_segment(
                    f"google_genai_{ms.slug}_{ds.dataset_name}_bs{batch_size}_trainseed10_test{test_tier}"
                )
                cfg_dict: dict[str, Any] = {
                    "name": exp_name,
                    "seed": seed,
                    "train_data": train_p,
                    "test_data": test_p,
                    "output_dir": str(out_root / exp_name),
                    "model": {
                        "kind": "google_genai_chat",
                        "params": {
                            "model_id": ms.model_id,
                            "prompt_id": "baseline_v1",
                            "batch_size": int(batch_size),
                            "max_concurrency": 5,
                            "temperature": 0.0,
                            "max_tokens": None,
                            "retries": 20,
                            "thinking_level": ms.thinking_level,
                            "include_thoughts": ms.include_thoughts,
                        },
                    },
                }
                planned.append({"name": exp_name, "config": cfg_dict})

                cfg = ExperimentConfig.model_validate(cfg_dict)
                resolved_path = out_root / "configs" / f"{exp_name}.json"
                resolved_path.parent.mkdir(parents=True, exist_ok=True)
                resolved_path.write_text(json.dumps(cfg.model_dump(mode="json"), indent=2), encoding="utf-8")

                if run:
                    logger.info(
                        "Google GenAI eval: {} ({} batch_size={})",
                        exp_name,
                        ms.model_id,
                        batch_size,
                    )
                    results.append(await arun_experiment(resolved_path))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(out_root),
        "test_tier": int(test_tier),
        "planned": planned,
        "ran": len(results),
        "results": results,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Google GenAI SDK evaluations (native thinking + tokens).")
    _ = ap.add_argument(
        "--results-root",
        type=Path,
        default=_repo_root() / "results" / "evaluate_google_llm",
        help="Root directory for timestamped run folders.",
    )
    _ = ap.add_argument(
        "--test-tier-only",
        action="store_true",
        help="Use test split tier_20 only (fast plumbing). Default is tier_200.",
    )
    _ = ap.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset_name keys (e.g. banking-10,tweet_eval_irony). Empty = all four.",
    )
    _ = ap.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model_id filter (e.g. gemma-3-4b-it,gemini-3.1-flash-lite-preview). Empty = full matrix.",
    )
    _ = ap.add_argument("--seed", type=int, default=42)
    _ = ap.add_argument("--dry-run", action="store_true", help="Print planned runs JSON only.")
    args = ap.parse_args()

    test_tier = 20 if args.test_tier_only else 200

    ds_filter = {x.strip() for x in args.datasets.split(",") if x.strip()}
    datasets = tuple(d for d in DATASETS if not ds_filter or d.dataset_name in ds_filter)

    mid_filter = {x.strip() for x in args.models.split(",") if x.strip()}
    models = tuple(m for m in DEFAULT_MODEL_MATRIX if not mid_filter or m.model_id in mid_filter)

    if args.dry_run:
        payload = {
            "test_tier": test_tier,
            "batch_sizes": list(BATCH_SIZES),
            "datasets": [d.dataset_name for d in datasets],
            "models": [{"slug": m.slug, "model_id": m.model_id, "thinking_level": m.thinking_level} for m in models],
            "runs_total_estimate": len(datasets) * len(models) * len(BATCH_SIZES),
        }
        print(json.dumps(payload, indent=2))
        return

    summary = asyncio.run(
        arun_google_eval_suite(
            results_root=args.results_root,
            test_tier=test_tier,
            datasets=datasets,
            models=models,
            run=True,
            seed=int(args.seed),
        )
    )
    print(json.dumps({"results_dir": summary["results_dir"], "ran": summary["ran"]}, indent=2))


if __name__ == "__main__":
    main()
