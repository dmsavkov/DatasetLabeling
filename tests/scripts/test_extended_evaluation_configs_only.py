from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypedDict, cast

from pytest import MonkeyPatch


class _RunSummary(TypedDict):
    created_at: str
    results_dir: str
    ran: int
    results: list[object]


def _should_not_run(*_a: object, **_k: object) -> object:
    raise RuntimeError("should not run")


def test_extended_evaluation_configs_only(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    cfg_dir = tmp_path / "experiments"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Configs-only mode should generate YAMLs + summary.json but not run experiments.
    from scripts.extended_evaluation_llm import arun_llm_suite
    from scripts.extended_evaluation_ml import arun_ml_suite

    # Ensure run=False does not call arun_experiment.
    monkeypatch.setattr(
        "scripts.extended_evaluation_llm.arun_experiment",
        _should_not_run,
    )
    monkeypatch.setattr(
        "scripts.extended_evaluation_ml.arun_experiment",
        _should_not_run,
    )

    llm_summary = cast(
        _RunSummary,
        cast(object, asyncio.run(arun_llm_suite(config_dir=cfg_dir, results_root=tmp_path / "results_llm", run=False))),
    )
    ml_summary = cast(
        _RunSummary,
        cast(object, asyncio.run(arun_ml_suite(config_dir=cfg_dir, results_root=tmp_path / "results_ml", run=False))),
    )

    assert llm_summary["ran"] == 0
    assert ml_summary["ran"] == 0
    assert (Path(llm_summary["results_dir"]) / "summary.json").exists()
    assert (Path(ml_summary["results_dir"]) / "summary.json").exists()
    assert list(cfg_dir.glob("*.yaml"))


def test_llm_and_ml_suite_prints(tmp_path: Path) -> None:
    from scripts.extended_evaluation_llm import ensure_llm_suite_configs
    from scripts.extended_evaluation_ml import ensure_ml_suite_configs

    cfg_dir = tmp_path / "experiments"
    llm = ensure_llm_suite_configs(cfg_dir, force=True)
    ml = ensure_ml_suite_configs(cfg_dir, force=False)
    assert llm
    assert ml

