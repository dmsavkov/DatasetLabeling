# pyright: basic
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.datasets.io import load_processed_tier
from src.datasets.cards import default_card_path, read_card_json
from src.datasets.io import processed_root as processed_root_fn
from src.datasets.schema import SCHEMA
from src.models.interfaces import Prediction, Predictor

from .artifacts import ensure_dir, predictions_path, report_path
from .label_compare import correctness_series, normalize_label_scalar
from .metrics import (
    compute_performance_excluding_confused,
    compute_performance_metrics,
    probs_if_accessible,
    summarize_usage,
)
from .reports import new_report, write_report_json


@dataclass(frozen=True, slots=True)
class EvalResult:
    predictions_df: pd.DataFrame
    report: dict[str, Any]
    output_dir: Path


def _pred_to_row(pred: Prediction) -> dict[str, Any]:
    usage = pred.usage
    row: dict[str, Any] = {
        "pred_label": pred.pred_label,
        "confidence": pred.confidence,
        "reason": pred.reason,
        "probs": pred.probs,
        "in_tokens": getattr(usage, "in_tokens", None) if usage is not None else None,
        "out_tokens": getattr(usage, "out_tokens", None) if usage is not None else None,
        "raw": pred.raw,
    }
    raw = pred.raw
    if isinstance(raw, dict):
        if "is_confusing" in raw:
            row["is_confusing"] = bool(raw.get("is_confusing"))
        if "pred_labels" in raw:
            row["pred_labels"] = raw.get("pred_labels")
        if "n_pred_labels" in raw:
            row["n_pred_labels"] = raw.get("n_pred_labels")
        for key in ("a_label", "a_conf", "a_reasoning", "b_label", "b_conf", "b_reasoning"):
            if key in raw:
                row[key] = raw.get(key)
        committee = raw.get("committee")
        if isinstance(committee, dict):
            members = committee.get("members")
            if isinstance(members, list):
                for idx, m in enumerate(members, start=1):
                    if not isinstance(m, dict):
                        continue
                    row[f"pred_label_member_{idx}"] = m.get("pred_label")
                    row[f"member_name_{idx}"] = m.get("name")
            if "majority" in committee:
                row["pred_label_majority"] = committee.get("majority")
    return row


def _attach_eval_columns(out_df: pd.DataFrame, *, dataset_name: str) -> pd.DataFrame:
    """Canonical correctness + string pred labels for stable CSV reload."""
    result = out_df.copy()
    if "pred_label" in result.columns:
        result["pred_label"] = result["pred_label"].map(normalize_label_scalar)
    result["correct"] = correctness_series(
        result,
        dataset_name=dataset_name,
        true_col=SCHEMA.true_label,
        pred_col="pred_label",
    )
    return result


def evaluate_predictor_on_tier(
    predictor: Predictor,
    *,
    dataset_name: str,
    split_name: str = "test",
    tier_size: int,
    output_dir: Path,
    processed_root: Path | None = None,
) -> EvalResult:
    df = load_processed_tier(dataset_name=dataset_name, split_name=split_name, tier_size=int(tier_size), root=processed_root)
    texts = df[SCHEMA.text].astype(str).tolist()
    # Prefer stable label-space from the dataset card, so tiny tiers (e.g. 20)
    # don't accidentally hide rare labels from the model.
    pr = processed_root_fn(processed_root)
    card_path = default_card_path(processed_root_dir=pr, dataset_name=dataset_name)
    if card_path.exists():
        allowed_labels = list(read_card_json(card_path).labels)
    else:
        allowed_labels = sorted(df[SCHEMA.true_label].astype(str).unique().tolist())

    start = time.perf_counter()
    preds = predictor.predict(texts, allowed_labels=allowed_labels)
    infer_s = time.perf_counter() - start

    if len(preds) != len(df):
        raise ValueError(f"Predictor returned {len(preds)} predictions for {len(df)} rows")

    pred_rows = [_pred_to_row(p) for p in preds]
    out_df = _attach_eval_columns(
        pd.concat([df.reset_index(drop=True), pd.DataFrame(pred_rows)], axis=1),
        dataset_name=dataset_name,
    )

    eval_df = out_df.rename(columns={SCHEMA.true_label: "true_label"})
    if "is_confusing" in eval_df.columns:
        perf_bundle = compute_performance_excluding_confused(eval_df, dataset_name=dataset_name)
        perf = perf_bundle.metrics
        confusion_stats = perf_bundle.confusion_stats
    else:
        perf = compute_performance_metrics(eval_df, dataset_name=dataset_name)
        confusion_stats = None

    extras: dict[str, Any] = {
        "infer_time_s": float(infer_s),
    }
    if confusion_stats is not None:
        extras["confusion_stats"] = {
            "n_total": confusion_stats.n_total,
            "n_scored": confusion_stats.n_scored,
            "n_confusing": confusion_stats.n_confusing,
            "confusing_rate": confusion_stats.confusing_rate,
            "n_zero_labels": confusion_stats.n_zero_labels,
            "n_multi_labels": confusion_stats.n_multi_labels,
        }
    pextra = probs_if_accessible(out_df)
    if pextra:
        extras["probs_if_accessible"] = pextra
    uextra = summarize_usage(out_df)
    if uextra:
        extras["usage"] = uextra

    rep = new_report(
        dataset_name=dataset_name,
        split_name=split_name,
        tier_size=int(tier_size),
        predictor_name=predictor.name,
        metrics=perf,
        extras=extras,
    )

    out_dir = ensure_dir(Path(output_dir))
    out_df.to_csv(predictions_path(out_dir), index=False)
    write_report_json(report_path(out_dir), rep)

    return EvalResult(predictions_df=out_df, report=rep.to_dict(), output_dir=out_dir)


async def aevaluate_predictor_on_df(
    predictor: Any,
    *,
    df: pd.DataFrame,
    allowed_labels: list[str],
    dataset_name: str,
    split_name: str,
    tier_size: int,
    output_dir: Path,
) -> EvalResult:
    """
    Async evaluation for predictors exposing `async def apredict(...)`.
    """

    texts = df[SCHEMA.text].astype(str).tolist()

    start = time.perf_counter()
    apredict_fn = getattr(predictor, "apredict", None)
    if apredict_fn is None:
        raise TypeError(f"Predictor {type(predictor)!r} has no apredict(); use evaluate_predictor_on_df for sync models")
    preds = await apredict_fn(texts, allowed_labels=allowed_labels)
    infer_s = time.perf_counter() - start

    if len(preds) != len(df):
        raise ValueError(f"Predictor returned {len(preds)} predictions for {len(df)} rows")

    pred_rows = [_pred_to_row(p) for p in preds]
    out_df = _attach_eval_columns(
        pd.concat([df.reset_index(drop=True), pd.DataFrame(pred_rows)], axis=1),
        dataset_name=dataset_name,
    )

    eval_df = out_df.rename(columns={SCHEMA.true_label: "true_label"})
    if "is_confusing" in eval_df.columns:
        perf_bundle = compute_performance_excluding_confused(eval_df, dataset_name=dataset_name)
        perf = perf_bundle.metrics
        confusion_stats = perf_bundle.confusion_stats
    else:
        perf = compute_performance_metrics(eval_df, dataset_name=dataset_name)
        confusion_stats = None

    extras: dict[str, Any] = {
        "infer_time_s": float(infer_s),
    }
    if confusion_stats is not None:
        extras["confusion_stats"] = {
            "n_total": confusion_stats.n_total,
            "n_scored": confusion_stats.n_scored,
            "n_confusing": confusion_stats.n_confusing,
            "confusing_rate": confusion_stats.confusing_rate,
            "n_zero_labels": confusion_stats.n_zero_labels,
            "n_multi_labels": confusion_stats.n_multi_labels,
        }
    pextra = probs_if_accessible(out_df)
    if pextra:
        extras["probs_if_accessible"] = pextra
    uextra = summarize_usage(out_df)
    if uextra:
        extras["usage"] = uextra

    rep = new_report(
        dataset_name=dataset_name,
        split_name=split_name,
        tier_size=int(tier_size),
        predictor_name=predictor.name,
        metrics=perf,
        extras=extras,
    )

    out_dir = ensure_dir(Path(output_dir))
    out_df.to_csv(predictions_path(out_dir), index=False)
    write_report_json(report_path(out_dir), rep)

    return EvalResult(predictions_df=out_df, report=rep.to_dict(), output_dir=out_dir)


def evaluate_predictor_on_df(
    predictor: Predictor,
    *,
    df: pd.DataFrame,
    allowed_labels: list[str],
    dataset_name: str,
    split_name: str,
    tier_size: int,
    output_dir: Path,
) -> EvalResult:
    texts = df[SCHEMA.text].astype(str).tolist()

    start = time.perf_counter()
    preds = predictor.predict(texts, allowed_labels=allowed_labels)
    infer_s = time.perf_counter() - start

    if len(preds) != len(df):
        raise ValueError(f"Predictor returned {len(preds)} predictions for {len(df)} rows")

    pred_rows = [_pred_to_row(p) for p in preds]
    out_df = _attach_eval_columns(
        pd.concat([df.reset_index(drop=True), pd.DataFrame(pred_rows)], axis=1),
        dataset_name=dataset_name,
    )

    eval_df = out_df.rename(columns={SCHEMA.true_label: "true_label"})
    if "is_confusing" in eval_df.columns:
        perf_bundle = compute_performance_excluding_confused(eval_df, dataset_name=dataset_name)
        perf = perf_bundle.metrics
        confusion_stats = perf_bundle.confusion_stats
    else:
        perf = compute_performance_metrics(eval_df, dataset_name=dataset_name)
        confusion_stats = None

    extras: dict[str, Any] = {
        "infer_time_s": float(infer_s),
    }
    if confusion_stats is not None:
        extras["confusion_stats"] = {
            "n_total": confusion_stats.n_total,
            "n_scored": confusion_stats.n_scored,
            "n_confusing": confusion_stats.n_confusing,
            "confusing_rate": confusion_stats.confusing_rate,
            "n_zero_labels": confusion_stats.n_zero_labels,
            "n_multi_labels": confusion_stats.n_multi_labels,
        }
    pextra = probs_if_accessible(out_df)
    if pextra:
        extras["probs_if_accessible"] = pextra
    uextra = summarize_usage(out_df)
    if uextra:
        extras["usage"] = uextra

    rep = new_report(
        dataset_name=dataset_name,
        split_name=split_name,
        tier_size=int(tier_size),
        predictor_name=predictor.name,
        metrics=perf,
        extras=extras,
    )

    out_dir = ensure_dir(Path(output_dir))
    out_df.to_csv(predictions_path(out_dir), index=False)
    write_report_json(report_path(out_dir), rep)

    return EvalResult(predictions_df=out_df, report=rep.to_dict(), output_dir=out_dir)

