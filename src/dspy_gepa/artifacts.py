# pyright: basic
"""Persist MIPROv2 / GEPA run artifacts under results/gepa_mipro/."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def begin_run_dir(*, repo_root: Path, dataset_name: str, run_kind: str) -> Path:
    run_dir = repo_root / "results" / "gepa_mipro" / dataset_name / run_kind / utc_stamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def save_run_manifest(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "run_manifest.json", payload)
