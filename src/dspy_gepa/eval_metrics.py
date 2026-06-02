# pyright: basic
"""Evaluation helpers (macro-F1 on sentence-level predictions)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def sentence_level_metrics(
    y_true: list[str],
    y_pred: list[str],
    *,
    labels: list[str],
) -> dict[str, Any]:
    if not y_true:
        return {"accuracy": 0.0, "f1_macro": 0.0, "n_samples": 0}
    f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    report_d = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    report_t = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "n_samples": int(len(y_true)),
        "classification_report": report_d,
        "classification_report_text": report_t,
    }


def flatten_batch_predictions(
    *,
    gold_batches: list[list[str]],
    pred_batches: list[list[str]],
) -> tuple[list[str], list[str]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    for g_batch, p_batch in zip(gold_batches, pred_batches, strict=True):
        y_true.extend(g_batch)
        y_pred.extend(p_batch)
    return y_true, y_pred
