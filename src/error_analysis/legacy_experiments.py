# pyright: basic
"""Discover and normalize legacy / non-harness experiment layouts under ``results/``.

Covers formats listed in ``.gitignore`` (mvp, hf_llms_comparison, prosocial, debug_accuracy, …).
Rows may duplicate harness runs when both exist; ``run_key`` namespaces keep them distinct.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.error_analysis.run_record import LEADERBOARD_COLUMNS

_STAMP = re.compile(r"(\d{8}_\d{6})")
_SUMMARY_JSON = re.compile(r"^(.+)_summary_(\d{8}_\d{6})\.json$")
_MVP_SUMMARY = re.compile(r"^mvp_(.+)_summary_(\d{8}_\d{6})\.json$")
_PROSOCIAL_EXP = re.compile(r"^(exp_.+)_summary_(\d{8}_\d{6})\.json$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_float(x: object) -> float | None:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _leaderboard_row(**kwargs: Any) -> dict[str, Any]:
    row = {c: None for c in LEADERBOARD_COLUMNS}
    row.update(kwargs)
    return row


def _resolve_local_path(results_root: Path, raw: str | None) -> Path | None:
    if not raw or not isinstance(raw, str):
        return None
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    # Artifacts often store old project roots; match by suffix under results/.
    norm = raw.replace("\\", "/")
    idx = norm.find("results/")
    if idx >= 0:
        candidate = results_root / norm[idx + len("results/") :]
        if candidate.is_file():
            return candidate.resolve()
    name = Path(norm).name
    hit = next(results_root.rglob(name), None)
    return hit.resolve() if hit is not None else None


def _pair_preds_for_summary(summary_path: Path, *, suffix: str = "preds") -> Path | None:
    name = summary_path.name
    m = _SUMMARY_JSON.match(name)
    if not m:
        return None
    prefix, stamp = m.group(1), m.group(2)
    parent = summary_path.parent
    for pattern in (
        f"{prefix}_{suffix}_*_{stamp}.csv",
        f"{prefix}_{suffix}_{stamp}.csv",
        f"{prefix}_preds_*_{stamp}.csv",
    ):
        hits = sorted(parent.glob(pattern))
        if hits:
            return hits[0]
    # mvp_v1 naming: mvp_dspy_preds_25test_20260421_082159.csv
    for p in parent.glob(f"{prefix.replace('_summary', '')}*.csv"):
        if stamp in p.name and "pred" in p.name.lower():
            return p
    for p in parent.glob(f"*{stamp}*.csv"):
        if "pred" in p.name.lower() and prefix.split("_")[0] in p.name:
            return p
    return None


@dataclass(frozen=True, slots=True)
class LegacyRunRecord:
    row: dict[str, Any]
    predictions_path: Path | None = None
    classification_report: dict[str, Any] | None = None


def _discover_mvp4(results_root: Path) -> list[LegacyRunRecord]:
    base = results_root / "mvp4_results"
    if not base.is_dir():
        return []
    out: list[LegacyRunRecord] = []
    payload_path = base / "mvp4_final_results.json"
    if not payload_path.is_file():
        payload_path = base / "mvp4_final_results.csv"
        if payload_path.is_file():
            df = pd.read_csv(payload_path)
            records = df.to_dict(orient="records")
        else:
            return []
    else:
        raw = json.loads(payload_path.read_text(encoding="utf-8"))
        records = raw if isinstance(raw, list) else []

    preds_by_stem = {p.stem: p for p in base.glob("preds_*.csv")}

    for rec in records:
        if not isinstance(rec, dict):
            continue
        dataset = str(rec.get("dataset") or "")
        model = str(rec.get("model") or "")
        train_n = rec.get("train_n")
        family = str(rec.get("model_family") or "ML")
        run_key = f"mvp4_results/{dataset}/{model}/train{train_n}"

        pred_path: Path | None = None
        for stem, path in preds_by_stem.items():
            if not stem.startswith("preds_"):
                continue
            body = stem[len("preds_") :]
            if dataset.replace("_", "") in body.replace("_", "") or dataset in body:
                if model.split("_")[0] in body or model in body:
                    if train_n is not None and f"n{int(train_n)}" in body:
                        pred_path = path
                        break
                    if train_n is None or family == "LLM":
                        if "n" not in body.split("_")[-1] or str(train_n) in body:
                            pred_path = path
        if pred_path is None and family == "LLM":
            for stem, path in preds_by_stem.items():
                if dataset in stem and model in stem:
                    pred_path = path
                    break

        out.append(
            LegacyRunRecord(
                row=_leaderboard_row(
                    run_key=run_key,
                    run_dir=f"mvp4_results/{dataset}",
                    series="mvp4_results",
                    campaign=None,
                    suite=dataset,
                    run_leaf=f"{model}_train{train_n}",
                    experiment_slug="mvp4",
                    predictor_name=model,
                    dataset_name=dataset,
                    tier_size=_safe_float(rec.get("test_n")),
                    n_samples=_safe_float(rec.get("test_n")),
                    model_kind=family,
                    model_id=model,
                    thinking_level=None,
                    batch_size=None,
                    phase="final",
                    accuracy=_safe_float(rec.get("accuracy")),
                    f1_macro=_safe_float(rec.get("macro_f1")),
                    duration_seconds=(_safe_float(rec.get("fit_seconds")) or 0)
                    + (_safe_float(rec.get("infer_seconds")) or 0)
                    or None,
                    infer_time_s=_safe_float(rec.get("infer_seconds")),
                    has_predictions=pred_path is not None,
                    predictions_source="legacy_csv" if pred_path else "none",
                    saved_utc=None,
                    format="mvp4_final_results",
                ),
                predictions_path=pred_path,
            )
        )
    return out


def _discover_mvp_v1(results_root: Path) -> list[LegacyRunRecord]:
    base = results_root / "mvp_v1"
    if not base.is_dir():
        return []
    out: list[LegacyRunRecord] = []
    for summary_path in sorted(base.glob("mvp_*_summary_*.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        m = _MVP_SUMMARY.match(summary_path.name)
        workflow = str(payload.get("workflow") or (m.group(1) if m else summary_path.stem))
        stamp = m.group(2) if m else summary_path.stem
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
        model_id = payload.get("model") or models.get("regular") or models.get("smart")
        pred_path = _pair_preds_for_summary(summary_path)
        if pred_path is None:
            art = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
            pred_path = _resolve_local_path(results_root, art.get("predictions_csv"))

        timing = payload.get("timing_seconds") if isinstance(payload.get("timing_seconds"), dict) else {}
        ss = payload.get("sample_sizes") if isinstance(payload.get("sample_sizes"), dict) else {}
        n_test = ss.get("test_eval") or ss.get("test") or ss.get("test_subset_25")

        cr = metrics.get("report") if isinstance(metrics.get("report"), dict) else None
        run_key = f"mvp_v1/{workflow}/{stamp}"
        out.append(
            LegacyRunRecord(
                row=_leaderboard_row(
                    run_key=run_key,
                    run_dir=f"mvp_v1/{workflow}",
                    series="mvp_v1",
                    campaign=stamp,
                    suite=workflow,
                    run_leaf=summary_path.stem,
                    experiment_slug=workflow,
                    predictor_name=workflow,
                    dataset_name="prosocial_dialogue",
                    tier_size=n_test,
                    n_samples=n_test,
                    model_kind="legacy_mvp",
                    model_id=str(model_id) if model_id else None,
                    thinking_level=None,
                    batch_size=None,
                    phase="final",
                    accuracy=_safe_float(metrics.get("accuracy")),
                    f1_macro=_safe_float(metrics.get("macro_f1")),
                    duration_seconds=_safe_float(timing.get("total")),
                    infer_time_s=_safe_float(timing.get("inference")),
                    has_predictions=pred_path is not None,
                    predictions_source="legacy_csv" if pred_path else "none",
                    saved_utc=None,
                    format="mvp_v1_summary",
                ),
                predictions_path=pred_path,
                classification_report=cr,
            )
        )
    return out


def _discover_prosocial(results_root: Path) -> list[LegacyRunRecord]:
    base = results_root / "prosocial_v3"
    if not base.is_dir():
        return []
    out: list[LegacyRunRecord] = []
    for summary_path in sorted(base.rglob("exp_*_summary_*.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        m = _PROSOCIAL_EXP.match(summary_path.name)
        exp_name = str(payload.get("workflow") or (m.group(1) if m else summary_path.stem))
        stamp = m.group(2) if m else ""
        rel = str(summary_path.parent.relative_to(results_root)).replace("\\", "/")
        pred_path = _pair_preds_for_summary(summary_path)
        if pred_path is None:
            art = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
            pred_path = _resolve_local_path(results_root, art.get("predictions_csv"))

        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        timing = extra.get("timing_seconds") if isinstance(extra.get("timing_seconds"), dict) else {}
        model_id = extra.get("prediction_model")

        run_key = f"{rel}/{exp_name}/{stamp}"
        out.append(
            LegacyRunRecord(
                row=_leaderboard_row(
                    run_key=run_key,
                    run_dir=rel,
                    series="prosocial_v3",
                    campaign=stamp,
                    suite=exp_name,
                    run_leaf=summary_path.stem,
                    experiment_slug=exp_name,
                    predictor_name=exp_name,
                    dataset_name="prosocial_dialogue",
                    tier_size=25,
                    n_samples=25,
                    model_kind="prosocial_v3",
                    model_id=str(model_id) if model_id else None,
                    thinking_level=None,
                    batch_size=extra.get("batch_size"),
                    phase="final",
                    accuracy=_safe_float(metrics.get("accuracy")),
                    f1_macro=_safe_float(metrics.get("macro_f1")),
                    duration_seconds=_safe_float(timing.get("inference")),
                    infer_time_s=_safe_float(timing.get("inference")),
                    has_predictions=pred_path is not None,
                    predictions_source="legacy_csv" if pred_path else "none",
                    saved_utc=None,
                    format="prosocial_v3_summary",
                ),
                predictions_path=pred_path,
            )
        )
    return out


def _discover_hf_llms(results_root: Path) -> list[LegacyRunRecord]:
    base = results_root / "hf_llms_comparison"
    if not base.is_dir():
        return []
    out: list[LegacyRunRecord] = []
    for summary_path in sorted(base.glob("summary_*.json")):
        stamp_m = _STAMP.search(summary_path.stem)
        stamp = stamp_m.group(1) if stamp_m else summary_path.stem.replace("summary_", "")
        metrics_path = base / f"metrics_{stamp}.csv"
        preds_path = base / f"predictions_long_{stamp}.csv"
        if not metrics_path.is_file():
            continue
        try:
            metrics_df = pd.read_csv(metrics_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read {}: {}", metrics_path, exc)
            continue

        for _, row in metrics_df.iterrows():
            dataset = str(row.get("dataset") or "")
            model = str(row.get("model") or "")
            if not dataset or not model or model == "major_vote":
                continue
            run_key = f"hf_llms_comparison/{stamp}/{dataset}/{model}"
            out.append(
                LegacyRunRecord(
                    row=_leaderboard_row(
                        run_key=run_key,
                        run_dir=f"hf_llms_comparison/{stamp}",
                        series="hf_llms_comparison",
                        campaign=stamp,
                        suite=dataset,
                        run_leaf=model,
                        experiment_slug="hf_llms_comparison",
                        predictor_name=model,
                        dataset_name=dataset,
                        tier_size=_safe_float(row.get("n_samples")),
                        n_samples=_safe_float(row.get("n_samples")),
                        model_kind="hf_openrouter",
                        model_id=model,
                        thinking_level=None,
                        batch_size=_safe_float(row.get("batch_size")),
                        phase="final",
                        accuracy=_safe_float(row.get("accuracy")),
                        f1_macro=_safe_float(row.get("macro_f1")),
                        duration_seconds=None,
                        infer_time_s=None,
                        has_predictions=preds_path.is_file(),
                        predictions_source="legacy_long_csv" if preds_path.is_file() else "none",
                        saved_utc=None,
                        format="hf_llms_metrics_csv",
                    ),
                    predictions_path=preds_path if preds_path.is_file() else None,
                )
            )
    return out


def _extract_debug_accuracy_metrics(payload: dict[str, Any]) -> tuple[float | None, float | None, int | None]:
    acc: float | None = None
    n: int | None = None
    for hyp in payload.get("hypotheses") or []:
        if not isinstance(hyp, dict):
            continue
        details = hyp.get("details") if isinstance(hyp.get("details"), dict) else {}
        hid = str(hyp.get("id") or "")
        for key in ("full_accuracy", "report_accuracy", "accuracy_raw_string", "accuracy_canonical_names"):
            v = _safe_float(details.get(key))
            if v is not None:
                acc = v
        if details.get("n") is not None:
            try:
                n = int(float(details["n"]))
            except (TypeError, ValueError):
                pass
        if acc is not None and hid.startswith("H4"):
            break
    return acc, acc, n


def _discover_debug_accuracy(results_root: Path) -> list[LegacyRunRecord]:
    base = results_root / "debug_accuracy"
    if not base.is_dir():
        return []
    out: list[LegacyRunRecord] = []
    for results_path in sorted(base.glob("*/results.json")):
        if not _STAMP.fullmatch(results_path.parent.name):
            continue
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        stamp = results_path.parent.name
        acc, _, n = _extract_debug_accuracy_metrics(payload)
        pred_csvs = list(results_path.parent.glob("live_*.csv"))
        pred_path = pred_csvs[0] if pred_csvs else None
        run_key = f"debug_accuracy/{stamp}"
        out.append(
            LegacyRunRecord(
                row=_leaderboard_row(
                    run_key=run_key,
                    run_dir=f"debug_accuracy/{stamp}",
                    series="debug_accuracy",
                    campaign=stamp,
                    suite=None,
                    run_leaf=stamp,
                    experiment_slug="debug_accuracy",
                    predictor_name="debug_ablation",
                    dataset_name="pubmed_20k_rct",
                    tier_size=n or 200,
                    n_samples=n,
                    model_kind="debug",
                    model_id=None,
                    thinking_level=None,
                    batch_size=None,
                    phase="analysis",
                    accuracy=acc,
                    f1_macro=acc,
                    duration_seconds=None,
                    infer_time_s=None,
                    has_predictions=pred_path is not None,
                    predictions_source="legacy_csv" if pred_path else "none",
                    saved_utc=payload.get("created_at"),
                    format="debug_accuracy_results",
                ),
                predictions_path=pred_path,
            )
        )
    return out


def discover_legacy_experiments(results_root: Path | None = None) -> list[LegacyRunRecord]:
    root = (results_root or (_repo_root() / "results")).resolve()
    records: list[LegacyRunRecord] = []
    for fn in (
        _discover_mvp4,
        _discover_mvp_v1,
        _discover_prosocial,
        _discover_hf_llms,
        _discover_debug_accuracy,
    ):
        try:
            batch = fn(root)
            records.extend(batch)
            logger.debug("{} legacy rows from {}", len(batch), fn.__name__)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Legacy discovery failed in {}: {}", fn.__name__, exc)
    logger.info("Discovered {} legacy experiment rows under {}", len(records), root)
    return records


def legacy_rows_for_leaderboard(records: list[LegacyRunRecord]) -> list[dict[str, Any]]:
    return [r.row for r in records]


def legacy_predictions_index(records: list[LegacyRunRecord]) -> dict[str, Path]:
    return {
        str(r.row["run_key"]): r.predictions_path
        for r in records
        if r.predictions_path is not None and r.row.get("run_key")
    }


def legacy_classification_reports(records: list[LegacyRunRecord]) -> dict[str, dict[str, Any]]:
    return {
        str(r.row["run_key"]): r.classification_report
        for r in records
        if r.classification_report is not None and r.row.get("run_key")
    }
