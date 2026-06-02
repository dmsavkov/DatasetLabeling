# pyright: basic
from __future__ import annotations

import pytest

from src.data_selection.gepa_pipeline_params import load_and_suggest, validate_or_raise
from src.data_selection.label_utils import load_dataset_context


def test_suggest_binary_dataset() -> None:
    ctx = load_dataset_context("implicit_hate")
    _, sug = load_and_suggest("implicit_hate")
    validate_or_raise(
        ctx,
        prediction_size=sug.prediction_size,
        batch_size=sug.batch_size,
        n_centroids_per_label=sug.n_centroids_per_label,
    )


def test_pubmed_defaults_validate() -> None:
    ctx = load_dataset_context("pubmed_20k_rct")
    _, sug = load_and_suggest("pubmed_20k_rct")
    n_centroids, per = validate_or_raise(
        ctx,
        prediction_size=sug.prediction_size,
        batch_size=sug.batch_size,
        n_centroids_per_label=sug.n_centroids_per_label,
    )
    assert n_centroids == 100
    assert per == 5


def test_bad_params_raise_with_hint() -> None:
    ctx = load_dataset_context("implicit_hate")
    with pytest.raises(ValueError, match="Suggested"):
        validate_or_raise(
            ctx,
            prediction_size=500,
            batch_size=10,
            n_centroids_per_label=20,
        )
