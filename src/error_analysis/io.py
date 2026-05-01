from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _repo_root() -> Path:
    # src/error_analysis/io.py -> src/error_analysis -> src -> repo root
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
    for fn in _EXPECTED_FILES:
        if (d / fn).is_file():
            return True
    return False


def discover_experiments(paths: list[str]) -> list[Path]:
    """
    Accept explicit (relative) path strings and return experiment directories.

    Rules:
    - If a path points at a folder that contains predictions.csv -> treat as single experiment.
    - Else, scan direct subfolders and include those that look like experiments.
    - No deep recursion by default (predictable notebook behavior).
    """
    out: list[Path] = []
    for p in paths:
        base = _as_path(p)
        if base.is_file():
            base = base.parent

        if (base / "predictions.csv").is_file():
            out.append(base)
            continue

        if base.is_dir():
            for child in sorted(base.iterdir()):
                if _looks_like_experiment_dir(child):
                    out.append(child)
            continue

    # stable, de-duplicated order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
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


def load_experiment(exp_dir: Path) -> LoadedExperiment:
    warnings: list[str] = []
    errors: list[str] = []

    exp_id = exp_dir.name
    preds: pd.DataFrame | None = None
    report: dict[str, Any] | None = None
    cfg: dict[str, Any] | None = None

    pred_path = exp_dir / "predictions.csv"
    if pred_path.is_file():
        try:
            preds = pd.read_csv(pred_path)
        except Exception as e:  # noqa: BLE001 - notebook-first robustness
            errors.append(f"Failed to read predictions.csv at {pred_path}: {_truncate_err(e)}")
    else:
        warnings.append(f"Missing predictions.csv: {pred_path}")

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

    return LoadedExperiment(
        exp_id=exp_id,
        path=exp_dir,
        predictions_df=preds,
        report=report,
        config=cfg,
        warnings=warnings,
        errors=errors,
    )


def load_many(paths: list[str]) -> tuple[list[LoadedExperiment], pd.DataFrame]:
    exp_dirs = discover_experiments(paths)
    loaded = [load_experiment(p) for p in exp_dirs]

    rows: list[dict[str, object]] = []
    for e in loaded:
        missing = [w for w in e.warnings if w.startswith("Missing ")]
        rows.append(
            {
                "exp_id": e.exp_id,
                "path": str(e.path),
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
    table = pd.DataFrame(rows).sort_values(["ok", "has_predictions", "exp_id"], ascending=[False, False, True])
    return loaded, table

