# pyright: basic
"""Discover experiment run directories under ``results/`` (all layouts)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

_SCORING_MARKERS: tuple[str, ...] = (
    "report.json",
    "full_metadata.json",
    "metrics.json",
    "val_eval_macro_f1.json",
)
_EXTRA_MARKERS: tuple[str, ...] = ("run_manifest.json", "full_classification_report.json", "preliminary_metadata.json")
_CAMPAIGN_STAMP = re.compile(r"^\d{8}_\d{6}$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, (dict, list)) else None


def _has_scoring_artifact(run_dir: Path) -> bool:
    return any((run_dir / name).is_file() for name in _SCORING_MARKERS)


def _is_prompt_eng_suite_parent(run_dir: Path) -> bool:
    summary = _load_json(run_dir / "summary.json")
    if not isinstance(summary, dict):
        return False
    return isinstance(summary.get("planned"), list) or isinstance(summary.get("results"), list)


def _is_candidate_run_dir(run_dir: Path) -> bool:
    if not run_dir.is_dir():
        return False
    if run_dir.name in {"configs", "__pycache__"}:
        return False
    if _has_scoring_artifact(run_dir):
        return True
    if (run_dir / "run_manifest.json").is_file() and (
        (run_dir / "full_metadata.json").is_file()
        or (run_dir / "full_classification_report.json").is_file()
        or (run_dir / "preliminary_metadata.json").is_file()
    ):
        return True
    # Started runs with manifest only (no scores yet) still count as experiments.
    if (run_dir / "run_manifest.json").is_file():
        return True
    return False


def _prune_suite_parents(run_dirs: list[Path]) -> list[Path]:
    resolved = [d.resolve() for d in run_dirs]
    keep: list[Path] = []
    for d, dr in zip(run_dirs, resolved):
        has_child_run = any(
            other != dr and other.is_relative_to(dr) and _has_scoring_artifact(Path(other))
            for other in resolved
        )
        if has_child_run and _is_prompt_eng_suite_parent(d):
            continue
        keep.append(d)
    return keep


@dataclass(frozen=True, slots=True)
class DiscoveredRun:
    run_dir: Path
    rel_dir: str
    series: str
    campaign: str | None
    suite: str | None
    run_leaf: str
    markers: tuple[str, ...]


def _infer_path_parts(run_dir: Path, results_root: Path) -> tuple[str, str | None, str | None]:
    try:
        rel = run_dir.resolve().relative_to(results_root.resolve())
    except ValueError:
        rel = run_dir.resolve().relative_to(_repo_root())
    parts = list(rel.parts)
    series = parts[0] if parts else ""
    campaign: str | None = None
    suite: str | None = None
    if len(parts) >= 2 and _CAMPAIGN_STAMP.fullmatch(parts[1]):
        campaign = parts[1]
        if len(parts) >= 3:
            suite = parts[2]
    elif len(parts) >= 2 and _CAMPAIGN_STAMP.fullmatch(parts[-1]):
        campaign = parts[-1]
    return series, campaign, suite


def _markers_present(run_dir: Path) -> tuple[str, ...]:
    names = _SCORING_MARKERS + _EXTRA_MARKERS + ("predictions.csv", "full_predictions.json", "summary.json")
    return tuple(n for n in names if (run_dir / n).is_file())


def discover_all_runs(results_root: Path | None = None) -> list[DiscoveredRun]:
    root = (results_root or (_repo_root() / "results")).resolve()
    if not root.is_dir():
        logger.warning("Results root does not exist: {}", root)
        return []

    candidates: set[Path] = set()
    for pattern in _SCORING_MARKERS + ("run_manifest.json",):
        for path in root.rglob(pattern):
            candidates.add(path.parent.resolve())

    run_dirs = sorted(d for d in candidates if _is_candidate_run_dir(d))
    run_dirs = _prune_suite_parents(run_dirs)

    out: list[DiscoveredRun] = []
    for run_dir in run_dirs:
        series, campaign, suite = _infer_path_parts(run_dir, root)
        rel = str(run_dir.relative_to(root)).replace("\\", "/")
        out.append(
            DiscoveredRun(
                run_dir=run_dir,
                rel_dir=rel,
                series=series,
                campaign=campaign,
                suite=suite,
                run_leaf=run_dir.name,
                markers=_markers_present(run_dir),
            )
        )
    logger.info("Discovered {} run directories under {}", len(out), root)
    return out


def discover_reasoning_spectrum_rows(results_root: Path | None = None) -> list[dict[str, Any]]:
    """Flatten ``reasoning_spectrum/*/summary.json`` tier rows into leaderboard-shaped dicts."""
    root = (results_root or (_repo_root() / "results")).resolve()
    rs = root / "reasoning_spectrum"
    if not rs.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(rs.glob("*/summary.json")):
        payload = _load_json(summary_path)
        if not isinstance(payload, dict):
            continue
        parent = summary_path.parent
        rel = str(parent.relative_to(root)).replace("\\", "/")
        suite = str(payload.get("suite") or parent.name)
        for tier in payload.get("tiers") or []:
            if not isinstance(tier, dict):
                continue
            tier_n = tier.get("tier")
            tier_name = tier.get("name")
            rows.append(
                {
                    "run_key": f"{rel}/tier_{tier_n}_{tier_name}",
                    "run_dir": rel,
                    "series": "reasoning_spectrum",
                    "campaign": parent.name,
                    "suite": suite,
                    "run_leaf": f"tier_{tier_n}_{tier_name}",
                    "experiment_slug": "reasoning_spectrum",
                    "predictor_name": f"reasoning_spectrum:{tier_name}",
                    "dataset_name": suite.split("_")[0] if "_" in suite else suite,
                    "tier_size": payload.get("n_items"),
                    "model_id": tier.get("model_id"),
                    "model_kind": "reasoning_spectrum_tier",
                    "thinking_level": None,
                    "batch_size": payload.get("batch_size"),
                    "phase": "tier",
                    "accuracy": tier.get("top1_accuracy"),
                    "f1_macro": tier.get("gold_in_set_rate"),
                    "duration_seconds": tier.get("elapsed_s"),
                    "infer_time_s": tier.get("elapsed_s"),
                    "has_predictions": False,
                    "predictions_source": "none",
                    "saved_utc": None,
                    "format": "reasoning_spectrum_summary",
                }
            )
    return rows
