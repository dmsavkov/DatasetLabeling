# pyright: basic
"""
Prompt-eng evaluations: multi-label confusion probe + self-debate.

Builds ExperimentConfig in memory (no YAML generator). Filters by --datasets and --models
for tactical runs, matching scripts/evaluate_google_llm.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.experiments.run import arun_experiment
from src.experiments.suites.prompt_eng_suite import (
    DEFAULT_MODEL_MATRIX,
    DATASETS,
    build_prompt_eng_configs,
    filter_datasets,
    filter_kinds,
    filter_models,
)
from src.utils.env import load_dotenv_if_present

load_dotenv_if_present()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


async def arun_prompt_eng_suite(
    *,
    results_root: Path,
    test_tier: int,
    datasets: tuple[str, ...],
    model_ids: tuple[str, ...],
    thinking_levels: tuple[str, ...],
    experiment_kinds: tuple[str, ...],
    run: bool,
    seed: int,
) -> dict[str, Any]:
    ds_filter = {x.strip() for x in datasets if x.strip()}
    mid_filter = {x.strip() for x in model_ids if x.strip()}
    think_filter = {x.strip().lower() for x in thinking_levels if x.strip()}
    kind_filter = {x.strip() for x in experiment_kinds if x.strip()}

    ds_specs = filter_datasets(ds_filter)
    ms_specs = filter_models(model_ids=mid_filter, thinking_levels=think_filter)
    kinds = filter_kinds(kind_filter)

    cfgs = build_prompt_eng_configs(
        test_tier=test_tier,
        seed=seed,
        datasets=ds_specs,
        models=ms_specs,
        kinds=kinds,
    )

    stamp = _utc_stamp()
    out_root = results_root / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    planned: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    logger.info(
        "Prompt-eng suite: {} runs (datasets={}, models={}, kinds={}) → {}",
        len(cfgs),
        [d.dataset_name for d in ds_specs],
        [m.slug for m in ms_specs],
        list(kinds),
        str(out_root),
    )

    for cfg in cfgs:
        kind = cfg.model.kind
        slug = "multilabel_confusion_probe" if kind == "multilabel_confusion_probe" else "self_debate"
        run_out = out_root / slug / cfg.name
        cfg_run = cfg.model_copy(update={"output_dir": str(run_out)})
        planned.append({"name": cfg.name, "kind": kind, "output_dir": str(run_out)})

        resolved_path = out_root / "configs" / f"{cfg.name}.json"
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(json.dumps(cfg_run.model_dump(mode="json"), indent=2), encoding="utf-8")

        if run:
            logger.info("Prompt-eng run: {} ({})", cfg.name, kind)
            results.append(await arun_experiment(resolved_path, experiment_slug=slug))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(out_root),
        "test_tier": int(test_tier),
        "planned": planned,
        "ran": len(results),
        "results": results,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Prompt-eng suite done: ran={}", len(results))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prompt-eng eval: multilabel confusion probe + self-debate (google.genai, in-memory configs)."
    )
    _ = ap.add_argument(
        "--results-root",
        type=Path,
        default=_repo_root() / "results" / "prompt_eng",
        help="Root directory for timestamped run folders.",
    )
    _ = ap.add_argument(
        "--test-tier-only",
        action="store_true",
        help="Use test/tier_20 (fast plumbing). Default is tier_200.",
    )
    _ = ap.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset_name keys (e.g. banking-10,pubmed_20k_rct). Empty = all four.",
    )
    _ = ap.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model_id filter (e.g. gemini-3.1-flash-lite-preview,gemma-4-31b-it). Empty = full matrix.",
    )
    _ = ap.add_argument(
        "--thinking-levels",
        type=str,
        default="",
        help="Comma-separated thinking levels: off, low, high. Empty = all levels in the selected model matrix.",
    )
    _ = ap.add_argument(
        "--experiments",
        type=str,
        default="",
        help="Comma-separated kinds: multilabel, self_debate (aliases: sdr). Empty = both.",
    )
    _ = ap.add_argument("--seed", type=int, default=42)
    _ = ap.add_argument("--dry-run", action="store_true", help="Print planned runs JSON only.")
    args = ap.parse_args()

    test_tier = 20 if args.test_tier_only else 200

    ds_tuple = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    mid_tuple = tuple(x.strip() for x in args.models.split(",") if x.strip())
    think_tuple = tuple(x.strip().lower() for x in args.thinking_levels.split(",") if x.strip())
    exp_tuple = tuple(x.strip() for x in args.experiments.split(",") if x.strip())

    if args.dry_run:
        ds_specs = filter_datasets(set(ds_tuple))
        ms_specs = filter_models(model_ids=set(mid_tuple), thinking_levels=set(think_tuple))
        kinds = filter_kinds(set(exp_tuple))
        cfgs = build_prompt_eng_configs(
            test_tier=test_tier,
            seed=int(args.seed),
            datasets=ds_specs,
            models=ms_specs,
            kinds=kinds,
        )
        payload = {
            "test_tier": test_tier,
            "datasets": [d.dataset_name for d in ds_specs],
            "models": [
                {
                    "slug": m.slug,
                    "model_id": m.model_id,
                    "thinking_level": m.thinking_level,
                }
                for m in ms_specs
            ],
            "experiments": list(kinds),
            "runs_total": len(cfgs),
            "run_names": [c.name for c in cfgs],
            "default_model_matrix": [
                {"slug": m.slug, "model_id": m.model_id, "thinking_level": m.thinking_level}
                for m in DEFAULT_MODEL_MATRIX
            ],
            "all_datasets": [d.dataset_name for d in DATASETS],
        }
        print(json.dumps(payload, indent=2))
        return

    summary = asyncio.run(
        arun_prompt_eng_suite(
            results_root=args.results_root,
            test_tier=test_tier,
            datasets=ds_tuple,
            model_ids=mid_tuple,
            thinking_levels=think_tuple,
            experiment_kinds=exp_tuple,
            run=True,
            seed=int(args.seed),
        )
    )
    print(json.dumps({"results_dir": summary["results_dir"], "ran": summary["ran"]}, indent=2))


if __name__ == "__main__":
    main()
