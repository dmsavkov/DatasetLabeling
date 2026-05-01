# pyright: basic
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from .config import ExperimentConfig, load_experiment_config
from .run import arun_experiment


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


"""
Baseline suite: curated playlist of committed experiment YAMLs (`experiments/*.yaml`).
Grouping is expressed only here, not by folder convention under results.
"""
BASELINE_SUITE_YAMLS: tuple[str, ...] = (
    "experiments/gemini_banking10_test200.yaml",
    "experiments/gemini_tweet_irony_test200.yaml",
    "experiments/svm_banking10_test200.yaml",
    "experiments/svm_tweet_irony_test200.yaml",
)


def write_config(path: Path, cfg: ExperimentConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")


async def abaseline_performance(
    *,
    results_root: Path = Path("results") / "baseline_performance",
    seed: int = 42,
    run: bool = False,
) -> list[dict[str, Any]]:
    """
    Load each baseline YAML from the repo, optionally run `arun_experiment` with
    `output_dir` under `results_root/<stamp>/<config_name>/`.
    """

    root = _repo_root()
    stamp = _utc_stamp()
    out_root = results_root / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    logger.info("Baseline suite start: {} configs → {}", len(BASELINE_SUITE_YAMLS), str(out_root))

    for rel in BASELINE_SUITE_YAMLS:
        cfg_path = root / rel
        if not cfg_path.is_file():
            logger.error("Baseline suite config missing: {}", str(cfg_path))
            raise FileNotFoundError(f"Baseline suite config missing: {cfg_path}")

        cfg = load_experiment_config(cfg_path)
        cfg_run = cfg.model_copy(update={"seed": seed, "output_dir": str(out_root / cfg.name)})
        resolved_path = out_root / "configs" / f"{cfg_run.name}.yaml"
        write_config(resolved_path, cfg_run)
        logger.info("Prepared config: {} → {}", cfg_run.name, str(resolved_path))
        if run:
            logger.info("Running: {}", cfg_run.name)
            runs.append(await arun_experiment(resolved_path))

    logger.info("Baseline suite done: ran={}", len(runs))
    return runs


def baseline_performance(
    *,
    results_root: Path = Path("results") / "baseline_performance",
    seed: int = 42,
    run: bool = False,
) -> list[dict[str, Any]]:
    """Sync wrapper: `asyncio.run(abaseline_performance(...))` at this boundary only."""

    return asyncio.run(abaseline_performance(results_root=results_root, seed=seed, run=run))
