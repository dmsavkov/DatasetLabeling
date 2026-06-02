from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.error_analysis.predictions_source import load_predictions_df
from src.error_analysis.metadata import (
    infer_campaign_and_suite,
    infer_series,
    match_key_from_config,
    unique_exp_id,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return _repo_root() / pp


_EXPECTED_FILES: tuple[str, ...] = ("predictions.csv", "report.json", "config.resolved.json")


def _looks_like_experiment_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    if (d / "predictions.csv").is_file():
        return True
    if (d / "full_predictions.json").is_file() and not (d / "full_predictions_skipped.txt").is_file():
        return True
    if (d / "report.json").is_file() or (d / "full_metadata.json").is_file():
        return True
    return False


def discover_experiments(paths: list[str], *, max_depth: int = 5) -> list[Path]:
    """
    Accept explicit (relative) path strings and return experiment directories.

    Rules:
    - If a path contains ``predictions.csv`` → single experiment dir.
    - Else walk subdirectories up to ``max_depth`` and collect every folder with
      ``predictions.csv`` (supports ``prompt_eng/<stamp>/<suite>/<exp>`` nesting).
    """
    found: list[Path] = []

    def walk(base: Path, depth: int) -> None:
        if _looks_like_experiment_dir(base):
            found.append(base.resolve())
            return
        if depth >= max_depth or not base.is_dir():
            return
        for child in sorted(base.iterdir()):
            if child.is_dir():
                walk(child, depth + 1)

    for p in paths:
        base = _as_path(p)
        if base.is_file():
            base = base.parent
        walk(base, 0)

    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in found:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _truncate_err(e: BaseException, *, max_len: int = 400) -> str:
    s = f"{type(e).__name__}: {e}"
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


@dataclass(frozen=True, slots=True)
class LoadedExperiment:
    exp_id: str
    path: Path
    predictions_df: pd.DataFrame | None
    report: dict[str, Any] | None
    config: dict[str, Any] | None
    warnings: list[str]
    errors: list[str]
    meta: dict[str, Any] = field(default_factory=dict)


def load_experiment(exp_dir: Path) -> LoadedExperiment:
    warnings: list[str] = []
    errors: list[str] = []

    preds: pd.DataFrame | None = None
    report: dict[str, Any] | None = None
    cfg: dict[str, Any] | None = None

    try:
        preds = load_predictions_df(exp_dir)
    except Exception as e:  # noqa: BLE001 - notebook-first robustness
        errors.append(f"Failed to load predictions at {exp_dir}: {_truncate_err(e)}")
        preds = None
    if preds is None:
        if (exp_dir / "predictions.csv").is_file() or (exp_dir / "full_predictions.json").is_file():
            warnings.append(f"Could not load predictions (missing or too large): {exp_dir}")
        else:
            warnings.append(f"No predictions.csv or full_predictions.json: {exp_dir}")

    report_path = exp_dir / "report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            errors.append(f"Failed to parse report.json at {report_path}: {_truncate_err(e)}")
    else:
        warnings.append(f"Missing report.json: {report_path}")

    cfg_path = exp_dir / "config.resolved.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            errors.append(f"Failed to parse config.resolved.json at {cfg_path}: {_truncate_err(e)}")
    else:
        warnings.append(f"Missing config.resolved.json: {cfg_path}")

    campaign, suite = infer_campaign_and_suite(exp_dir)
    match = match_key_from_config(cfg, report=report)
    meta: dict[str, Any] = {
        "series": infer_series(exp_dir),
        "campaign": campaign,
        "suite": suite,
        **match,
    }

    return LoadedExperiment(
        exp_id=unique_exp_id(exp_dir),
        path=exp_dir,
        predictions_df=preds,
        report=report,
        config=cfg,
        warnings=warnings,
        errors=errors,
        meta=meta,
    )


def load_many(paths: list[str], *, max_depth: int = 5) -> tuple[list[LoadedExperiment], pd.DataFrame]:
    exp_dirs = discover_experiments(paths, max_depth=max_depth)
    loaded = [load_experiment(p) for p in exp_dirs]

    rows: list[dict[str, object]] = []
    for e in loaded:
        missing = [w for w in e.warnings if w.startswith("Missing ")]
        rows.append(
            {
                "exp_id": e.exp_id,
                "path": str(e.path),
                "series": e.meta.get("series"),
                "campaign": e.meta.get("campaign"),
                "suite": e.meta.get("suite"),
                "dataset_name": e.meta.get("dataset_name"),
                "model_id": e.meta.get("model_id"),
                "model_kind": e.meta.get("model_kind"),
                "ok": len(e.errors) == 0,
                "has_predictions": e.predictions_df is not None,
                "has_report": e.report is not None,
                "has_config": e.config is not None,
                "n_rows_predictions": int(len(e.predictions_df)) if e.predictions_df is not None else None,
                "n_missing": len(missing),
                "n_warnings": len(e.warnings),
                "n_errors": len(e.errors),
                "warnings": "\n".join(e.warnings) if e.warnings else "",
                "errors": "\n".join(e.errors) if e.errors else "",
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["series", "campaign", "suite", "has_predictions", "exp_id"],
        ascending=[True, True, True, False, True],
        na_position="last",
    )
    return loaded, table
