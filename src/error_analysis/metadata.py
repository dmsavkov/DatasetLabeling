# pyright: basic
"""Path and config metadata for experiment runs (evaluate_google_llm, prompt_eng, …)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_PROMPT_ENG_SUITES = frozenset({"multilabel_confusion_probe", "self_debate"})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def infer_series(exp_dir: Path) -> str:
    try:
        rel = exp_dir.resolve().relative_to(_repo_root() / "results")
        if rel.parts:
            return rel.parts[0]
    except ValueError:
        pass
    parts = {p.lower() for p in exp_dir.resolve().parts}
    if "prompt_eng" in parts:
        return "prompt_eng"
    if "evaluate_google_llm" in parts:
        return "evaluate_google_llm"
    return "other"


def infer_campaign_and_suite(exp_dir: Path) -> tuple[str | None, str | None]:
    """
    ``…/<series>/<campaign>/<suite?>/<exp>`` → campaign stamp + optional suite folder.
    """
    try:
        rel = exp_dir.resolve().relative_to(_repo_root())
    except ValueError:
        rel = exp_dir.resolve()

    parts = list(rel.parts)
    if len(parts) < 2:
        return None, None

    campaign: str | None = None
    suite: str | None = None
    # results/<series>/<campaign>/...
    if len(parts) >= 3 and re.fullmatch(r"\d{8}_\d{6}", parts[2]):
        campaign = parts[2]
        if len(parts) >= 4 and parts[3] in _PROMPT_ENG_SUITES:
            suite = parts[3]
    return campaign, suite


def match_key_from_config(config: dict[str, Any] | None, *, report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Stable join key: dataset + model + thinking + tier."""
    out: dict[str, Any] = {
        "dataset_name": None,
        "model_id": None,
        "thinking_level": None,
        "tier_size": None,
        "model_kind": None,
    }
    if report:
        out["dataset_name"] = report.get("dataset_name")
        out["tier_size"] = report.get("tier_size")
        out["predictor_name"] = report.get("predictor_name")

    if not config:
        return out

    model = config.get("model")
    if isinstance(model, dict):
        out["model_kind"] = model.get("kind")
        params = model.get("params")
        if isinstance(params, dict):
            out["model_id"] = params.get("model_id")
            out["thinking_level"] = params.get("thinking_level")
            out["batch_size"] = params.get("batch_size")

    if out["tier_size"] is None and isinstance(config.get("test_data"), str):
        m = re.search(r"tier_(\d+)", str(config["test_data"]))
        if m:
            out["tier_size"] = int(m.group(1))

    return out


def unique_exp_id(exp_dir: Path) -> str:
    """Leaf name for flat runs; ``suite/leaf`` for nested prompt_eng runs."""
    campaign, suite = infer_campaign_and_suite(exp_dir)
    if suite:
        return f"{suite}/{exp_dir.name}"
    return exp_dir.name
