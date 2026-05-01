from __future__ import annotations

from pathlib import Path

from src.experiments.baseline_performance import BASELINE_SUITE_YAMLS, baseline_performance


def test_baseline_suite_yaml_files_exist_in_repo() -> None:
    root = Path(__file__).resolve().parents[2]
    for rel in BASELINE_SUITE_YAMLS:
        assert (root / rel).is_file(), f"missing committed experiment config: {rel}"


def test_baseline_performance_writes_resolved_configs(tmp_path: Path) -> None:
    out = baseline_performance(results_root=tmp_path, seed=1, run=False)
    assert out == []
    subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert subdirs
    configs = list(subdirs[0].rglob("*.yaml"))
    assert len(configs) >= len(BASELINE_SUITE_YAMLS)
