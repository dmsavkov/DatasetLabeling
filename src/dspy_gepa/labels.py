# pyright: basic
"""Dataset-aware label helpers for DSPy GEPA runs."""

from __future__ import annotations

from src.data_selection.label_utils import (
    PUBMED_ID_TO_NAME,
    PUBMED_LABEL_NAMES,
    PUBMED_NAME_TO_ID,
    canonicalizer_for_dataset,
    normalize_pubmed_label,
    prompt_labels_for_dataset,
)


def normalize_label_for_dataset(value: object, *, dataset_name: str) -> str:
    """Canonical prompt/metric label string for a dataset."""
    canon = canonicalizer_for_dataset(dataset_name)
    if value is None:
        return ""
    return canon(value)


def labels_for_dataset(*, dataset_name: str, label_ids: list[str]) -> list[str]:
    """Strings shown in allowed_labels / sklearn reports."""
    return prompt_labels_for_dataset(dataset_name=dataset_name, label_ids=label_ids)


__all__ = [
    "PUBMED_LABEL_NAMES",
    "PUBMED_ID_TO_NAME",
    "PUBMED_NAME_TO_ID",
    "normalize_pubmed_label",
    "normalize_label_for_dataset",
    "labels_for_dataset",
]
