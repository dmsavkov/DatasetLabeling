from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from openai import AsyncOpenAI, OpenAI

from prosocial.constants import ExperimentRun, UNKNOWN_DISTANCE_PENALTY
from prosocial.inference import run_inference
from src.data import collapse_series_to_three_labels, evaluate_adjusted_distance, evaluate_predictions, now_stamp, save_json


def get_available_model_ids(client: OpenAI) -> list[str]:
    try:
        response = client.models.list()
        raw_ids = [str(item.id) for item in getattr(response, "data", [])]
        normalized: set[str] = set(raw_ids)
        for model_id in raw_ids:
            if model_id.startswith("models/"):
                normalized.add(model_id.split("/", 1)[1])
        return sorted(normalized)
    except Exception:
        return []


def summarize_experiment(
    *,
    name: str,
    results_df: pd.DataFrame,
    labels: list[str],
    summary_path: Path,
    predictions_path: Path,
    extra: dict[str, Any],
) -> dict[str, Any]:
    y_true = pd.Series(results_df["true_label"], dtype="string")
    y_pred = pd.Series(results_df["pred_label"], dtype="string")

    valid_mask = y_pred != "parse_error"
    if valid_mask.any():
        core_metrics = evaluate_predictions(
            pd.Series(y_true[valid_mask], dtype="string"),
            pd.Series(y_pred[valid_mask], dtype="string"),
            labels,
        )
    else:
        core_metrics = {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "report": {}}

    adjusted_distance = evaluate_adjusted_distance(y_true, y_pred, unknown_penalty=UNKNOWN_DISTANCE_PENALTY)
    payload = {
        "workflow": name,
        "metrics": {
            "accuracy": float(core_metrics["accuracy"]),
            "macro_f1": float(core_metrics["macro_f1"]),
            "weighted_f1": float(core_metrics["weighted_f1"]),
            "adjusted_distance": float(adjusted_distance),
            "parse_error_count": int((y_pred == "parse_error").sum()),
        },
        "artifacts": {"predictions_csv": str(predictions_path)},
        "extra": extra,
    }
    save_json(summary_path, payload)
    return payload


def write_comparison_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# Prosocial V3 Experiment Comparison",
        "",
        "| Workflow | Accuracy | Macro F1 | Weighted F1 | Adjusted Distance | Parse Errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + f"{row['workflow']} | {row['accuracy']:.3f} | {row['macro_f1']:.3f} | "
            + f"{row['weighted_f1']:.3f} | {row['adjusted_distance']:.3f} | {row['parse_error_count']} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_model_experiment(
    *,
    run: ExperimentRun,
    test25_df: pd.DataFrame,
    labels: list[str],
    async_client: AsyncOpenAI,
    results_dir: Path,
    max_concurrency: int,
    retrieval_map: dict[int, list[dict[str, Any]]] | None,
    static_fewshots: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    t0 = perf_counter()
    results = asyncio.run(
        run_inference(
            test25_df,
            client=async_client,
            labels=labels,
            optimized_prompt=run.optimized_prompt,
            prediction_model=run.prediction_model,
            assertion_text=run.assertion_text,
            static_fewshots=static_fewshots,
            batch_size=run.batch_size,
            max_concurrency=max_concurrency,
            retrieval_map=retrieval_map if run.include_dynamic_retrieval else None,
            enable_statement_extraction=run.enable_statement_extraction,
            moe_experts=run.moe_experts,
        )
    )
    infer_seconds = perf_counter() - t0

    results_df = pd.DataFrame(results)
    stamp = now_stamp()
    pred_path = results_dir / f"{run.name}_preds_test25_{stamp}.csv"
    summary_path = results_dir / f"{run.name}_summary_{stamp}.json"
    results_df.to_csv(pred_path, index=False)

    summary = summarize_experiment(
        name=run.name,
        results_df=results_df,
        labels=labels,
        summary_path=summary_path,
        predictions_path=pred_path,
        extra={
            "prediction_model": run.prediction_model,
            "batch_size": run.batch_size,
            "statement_extraction": run.enable_statement_extraction,
            "dynamic_retrieval": run.include_dynamic_retrieval,
            "moe_experts": run.moe_experts or [],
            "timing_seconds": {"inference": float(infer_seconds)},
        },
    )
    return results_df, summary, pred_path, summary_path


def run_collapse_experiment(
    *,
    name: str,
    base_results_df: pd.DataFrame,
    summary_base: dict[str, Any],
    results_dir: Path,
) -> dict[str, Any]:
    y_true_3 = collapse_series_to_three_labels(pd.Series(base_results_df["true_label"], dtype="string"))
    y_pred_3 = collapse_series_to_three_labels(pd.Series(base_results_df["pred_label"], dtype="string"))

    labels_3 = ["casual", "needs_caution", "needs_intervention"]
    valid_mask = y_pred_3 != "parse_error"
    if valid_mask.any():
        metrics = evaluate_predictions(
            pd.Series(y_true_3[valid_mask], dtype="string"),
            pd.Series(y_pred_3[valid_mask], dtype="string"),
            labels_3,
        )
    else:
        metrics = {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "report": {}}

    stamp = now_stamp()
    collapsed_pred_path = results_dir / f"{name}_preds_test25_{stamp}.csv"
    collapsed_summary_path = results_dir / f"{name}_summary_{stamp}.json"

    pd.DataFrame(
        {
            "source_index": base_results_df["source_index"],
            "true_label_3class": y_true_3,
            "pred_label_3class": y_pred_3,
        }
    ).to_csv(collapsed_pred_path, index=False)

    payload = {
        "workflow": name,
        "based_on": summary_base.get("workflow", "unknown"),
        "metrics": {
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
            "parse_error_count": int((y_pred_3 == "parse_error").sum()),
        },
        "artifacts": {"predictions_csv": str(collapsed_pred_path)},
    }
    save_json(collapsed_summary_path, payload)
    return payload
