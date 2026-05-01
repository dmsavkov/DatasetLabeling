# pyright: basic
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.experiments.config import load_experiment_config
from src.experiments.run import arun_experiment
from src.experiments.suites.extended_suite import llm_suite_filenames, write_extended_suite_yamls


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_llm_suite_configs(config_dir: Path, *, force: bool = False) -> list[Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    if force:
        write_extended_suite_yamls(config_dir)
    expected = [config_dir / n for n in llm_suite_filenames()]
    missing = [p for p in expected if not p.is_file()]
    if missing:
        write_extended_suite_yamls(config_dir)
    return expected


async def arun_llm_suite(*, config_dir: Path, results_root: Path, run: bool) -> dict[str, Any]:
    cfg_paths = ensure_llm_suite_configs(config_dir)
    stamp = _utc_stamp()
    out_root = results_root / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info("LLM suite start: {} configs → {}", len(cfg_paths), str(out_root))
    results: list[dict[str, Any]] = []
    for p in cfg_paths:
        cfg = load_experiment_config(p)
        logger.info("Config loaded: {} ({})", cfg.name, str(p))
        resolved_path = out_root / "configs" / p.name
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        if run:
            cfg_run = cfg.model_copy(update={"output_dir": str(out_root / cfg.name)})
            resolved_path.write_text(json.dumps(cfg_run.model_dump(mode="json"), indent=2), encoding="utf-8")
            results.append(await arun_experiment(resolved_path))

    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "results_dir": str(out_root), "ran": len(results), "results": results}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("LLM suite done: ran={}", len(results))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the LLM-only extended evaluation suite.")
    _ = ap.add_argument("--config-dir", type=Path, default=_repo_root() / "experiments")
    _ = ap.add_argument("--results-root", type=Path, default=_repo_root() / "results" / "extended_evaluation_llm")
    _ = ap.add_argument("--configs-only", action="store_true", help="Generate/update suite YAMLs only.")
    _ = ap.add_argument("--print-suite", action="store_true", help="Print the exact config list and exit.")
    args = ap.parse_args()

    if args.print_suite:
        cfgs = ensure_llm_suite_configs(args.config_dir)
        print(json.dumps([str(p) for p in cfgs], indent=2))
        return

    if args.configs_only:
        cfgs = ensure_llm_suite_configs(args.config_dir, force=True)
        print(json.dumps({"generated": len(cfgs), "config_dir": str(args.config_dir)}, indent=2))
        return

    summary = asyncio.run(arun_llm_suite(config_dir=args.config_dir, results_root=args.results_root, run=True))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

