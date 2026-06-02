# pyright: basic
"""Backward-compatible PubMed aliases (prefer ``batch_classifier`` for new code)."""

from __future__ import annotations

from src.dspy_gepa.batch_classifier import (
    BatchTextClassifier,
    batch_metric_factory,
    examples_from_batches,
    parse_predicted_labels as _parse_core,
)
from src.dspy_gepa.labels import PUBMED_LABEL_NAMES, labels_for_dataset, normalize_pubmed_label

PubMedBatchClassifier = BatchTextClassifier


def parse_predicted_labels(
    raw: object,
    *,
    batch_size: int,
    allowed_labels: list[str] | None = None,
) -> list[str]:
    allowed = allowed_labels or list(PUBMED_LABEL_NAMES)
    return _parse_core(
        raw,
        batch_size=batch_size,
        allowed_labels=allowed,
        dataset_name="pubmed_20k_rct",
    )


__all__ = [
    "PubMedBatchClassifier",
    "batch_metric_factory",
    "examples_from_batches",
    "parse_predicted_labels",
    "labels_for_dataset",
    "normalize_pubmed_label",
]
