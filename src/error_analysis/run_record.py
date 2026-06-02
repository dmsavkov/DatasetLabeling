# pyright: basic
"""Normalize metadata from any run directory into leaderboard rows."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.error_analysis.discover import DiscoveredRun
from src.error_analysis.metadata import match_key_from_config

_TIER_RE = re.compile(r"tier_(\d+)")


def _safe_float(x: object) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _get(d: dict[str, Any] | None, *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _macro_f1_from_sklearn_report(cr: dict[str, Any]) -> float | None:
    macro = cr.get("macro avg")
    if isinstance(macro, dict):
        return _safe_float(macro.get("f1-score"))
    return None


def _predictions_source(run_dir: Path) -> str:
    if (run_dir / "predictions.csv").is_file():
        return "csv"
    if (run_dir / "full_predictions.json").is_file():
        if (run_dir / "full_predictions_skipped.txt").is_file():
            return "skipped"
        return "json"
    return "none"


# Exported column order for leaderboard.csv (18 focused fields).
LEADERBOARD_COLUMNS: tuple[str, ...] = (
    "run_key",
    "series",
    "campaign",
    "suite",
    "run_leaf",
    "experiment_slug",
    "predictor_name",
    "dataset_name",
    "tier_size",
    "n_samples",
    "model_kind",
    "model_id",
    "thinking_level",
    "batch_size",
    "phase",
    "accuracy",
    "f1_macro",
    "duration_seconds",
    "infer_time_s",
    "has_predictions",
    "predictions_source",
    "saved_utc",
    "run_dir",
    "format",
)


@dataclass(frozen=True, slots=True)
class RunRecord:
    row: dict[str, Any]

    def to_leaderboard_dict(self) -> dict[str, Any]:
        return {c: self.row.get(c) for c in LEADERBOARD_COLUMNS}


def _config_dict(run_dir: Path, manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    for name in ("config.resolved.json", "config.json"):
        cfg = _read_json(run_dir / name)
        if cfg:
            return cfg
    if manifest and isinstance(manifest.get("config"), dict):
        return manifest["config"]
    return None


def _report_from_sources(
    run_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, str]:
    report = _read_json(run_dir / "report.json")
    meta = _read_json(run_dir / "full_metadata.json") or _read_json(run_dir / "preliminary_metadata.json")
    metrics = _read_json(run_dir / "metrics.json")
    val_eval = _read_json(run_dir / "val_eval_macro_f1.json")
    if report:
        return report, meta, metrics, "harness_report"
    if metrics:
        return (
            {
                "dataset_name": metrics.get("dataset_name"),
                "tier_size": metrics.get("n_samples"),
                "metrics": {
                    "f1_macro": metrics.get("f1_macro") or metrics.get("macro_f1"),
                    "accuracy": metrics.get("accuracy"),
                },
                "extras": {},
            },
            meta,
            metrics,
            "metrics_json",
        )
    if val_eval:
        return (
            {
                "metrics": {
                    "f1_macro": val_eval.get("f1_macro"),
                    "accuracy": val_eval.get("accuracy"),
                },
                "tier_size": val_eval.get("n_samples"),
                "extras": {},
            },
            meta,
            val_eval,
            "gepa_val_eval",
        )
    if meta:
        return (
            {
                "metrics": meta.get("metrics") if isinstance(meta.get("metrics"), dict) else {},
                "extras": meta.get("extras") if isinstance(meta.get("extras"), dict) else {},
            },
            meta,
            metrics,
            "metadata_only",
        )
    return None, meta, metrics, "manifest_only"


def extract_run_record(discovered: DiscoveredRun) -> RunRecord | None:
    run_dir = discovered.run_dir
    manifest = _read_json(run_dir / "run_manifest.json")
    report, meta, metrics_file, fmt = _report_from_sources(run_dir)
    cfg = _config_dict(run_dir, manifest)

    if report is None and manifest is None and metrics_file is None:
        return None

    if report is None and manifest is not None and metrics_file is None:
        fmt = "manifest_only"

    metrics = (report or {}).get("metrics") if isinstance(report, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    extras = (report or {}).get("extras") if isinstance(report, dict) else {}
    if not isinstance(extras, dict):
        extras = meta.get("extras") if isinstance(meta, dict) and isinstance(meta.get("extras"), dict) else {}

    # sklearn-style report fallback
    if not metrics.get("f1_macro") and not metrics.get("macro_f1"):
        cr = _read_json(run_dir / "full_classification_report.json")
        if isinstance(cr, dict):
            mf1 = _macro_f1_from_sklearn_report(cr)
            if mf1 is not None:
                metrics = {**metrics, "f1_macro": mf1}
            if metrics.get("accuracy") is None and isinstance(cr.get("accuracy"), (int, float)):
                metrics = {**metrics, "accuracy": cr["accuracy"]}

    f1 = _safe_float(metrics.get("f1_macro") or metrics.get("macro_f1"))
    acc = _safe_float(metrics.get("accuracy"))
    match = match_key_from_config(cfg, report=report if isinstance(report, dict) else None)

    model = cfg.get("model") if isinstance(cfg, dict) else {}
    params = model.get("params") if isinstance(model, dict) and isinstance(model.get("params"), dict) else {}

    tier_size = match.get("tier_size")
    if tier_size is None and isinstance(report, dict):
        tier_size = report.get("tier_size")
    if tier_size is None and isinstance(metrics_file, dict):
        tier_size = metrics_file.get("n_samples")

    n_samples = metrics_file.get("n_samples") if isinstance(metrics_file, dict) else None
    if n_samples is None and isinstance(extras, dict):
        cs = extras.get("confusion_stats")
        if isinstance(cs, dict):
            n_samples = cs.get("n_total") or cs.get("n_scored")

    phase = "full" if (run_dir / "full_metadata.json").is_file() else "preliminary"
    if (run_dir / "report.json").is_file():
        phase = "full"
    if fmt == "manifest_only":
        phase = "started"

    experiment_slug = None
    if manifest:
        experiment_slug = manifest.get("experiment_slug") or manifest.get("experiment")
    if not experiment_slug and isinstance(cfg, dict):
        experiment_slug = cfg.get("name")

    pred_src = _predictions_source(run_dir)
    row: dict[str, Any] = {
        "run_key": discovered.rel_dir,
        "run_dir": discovered.rel_dir,
        "series": discovered.series,
        "campaign": discovered.campaign,
        "suite": discovered.suite,
        "run_leaf": discovered.run_leaf,
        "experiment_slug": experiment_slug,
        "predictor_name": (report or {}).get("predictor_name") if isinstance(report, dict) else None,
        "dataset_name": match.get("dataset_name"),
        "tier_size": tier_size,
        "n_samples": n_samples,
        "model_kind": match.get("model_kind") or (model.get("kind") if isinstance(model, dict) else None),
        "model_id": match.get("model_id") or params.get("model_id"),
        "thinking_level": match.get("thinking_level") or params.get("thinking_level"),
        "batch_size": match.get("batch_size") or params.get("batch_size"),
        "phase": phase,
        "accuracy": acc,
        "f1_macro": f1,
        "duration_seconds": _safe_float(meta.get("duration_seconds")) if isinstance(meta, dict) else None,
        "infer_time_s": _safe_float(extras.get("infer_time_s")) if isinstance(extras, dict) else None,
        "has_predictions": pred_src in {"csv", "json"},
        "predictions_source": pred_src,
        "saved_utc": meta.get("saved_utc") if isinstance(meta, dict) else _get(report, "created_at"),
        "format": fmt,
        "markers": ",".join(discovered.markers),
    }
    return RunRecord(row=row)
