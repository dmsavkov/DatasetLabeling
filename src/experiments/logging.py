# pyright: basic
"""Minimal run logging for prompt-eng and other experiment groups."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PREDICTIONS_EXPORT_MAX = 500


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_run_manifest(
    out_dir: Path,
    *,
    experiment_slug: str,
    config_path: str | None,
    cfg_payload: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "experiment_slug": experiment_slug,
        "started_utc": utc_iso(),
        "config_path": config_path,
        "config": cfg_payload,
    }
    if extra:
        manifest.update(extra)
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def write_full_metadata(
    out_dir: Path,
    *,
    report: dict[str, Any],
    duration_seconds: float | None = None,
    notes: str | None = None,
) -> Path:
    metrics = report.get("metrics", {})
    extras = report.get("extras", {})
    payload: dict[str, Any] = {
        "saved_utc": utc_iso(),
        "duration_seconds": duration_seconds,
        "metrics": metrics,
        "extras": extras,
        "notes": notes,
    }
    path = out_dir / "full_metadata.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def write_predictions_json(out_dir: Path, predictions: list[dict[str, Any]]) -> Path | None:
    if len(predictions) > PREDICTIONS_EXPORT_MAX:
        note = out_dir / "full_predictions_skipped.txt"
        note.write_text(
            f"predictions not exported: n={len(predictions)} > {PREDICTIONS_EXPORT_MAX}\n",
            encoding="utf-8",
        )
        return None
    path = out_dir / "full_predictions.json"
    path.write_text(json.dumps(predictions, indent=2, default=str), encoding="utf-8")
    return path
