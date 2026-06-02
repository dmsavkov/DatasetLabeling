# pyright: basic
"""Default / validated parameters for the huge-prediction → GEPA pipeline per dataset."""

from __future__ import annotations

from dataclasses import dataclass

from src.data_selection.label_utils import DatasetContext, load_dataset_context


@dataclass(frozen=True, slots=True)
class HugePredictionDefaults:
    prediction_size: int
    batch_size: int
    n_centroids_per_label: int
    pool_size: int
    note: str


def suggest_huge_prediction_defaults(ctx: DatasetContext) -> HugePredictionDefaults:
    """
    Pick prediction_size / n_centroids_per_label so divisibility checks pass.

    Pubmed (5 labels): 20×5=100 centroids, 500 predictions → 5 samples/centroid.
    Binary (2 labels): 10×2=20 centroids, 400 predictions → 20 samples/centroid.
    Banking (10 labels): 10×10=100 centroids, 400 predictions → 4 samples/centroid.
    """
    n = ctx.n_labels
    batch_size = 10
    if n <= 2:
        n_centroids_per_label = 10
        prediction_size = 400
        note = "2-class: smaller centroid grid so 400 % (10×n_labels) == 0"
    elif n <= 5:
        n_centroids_per_label = 20
        prediction_size = 500
        note = "5-class pubmed-style grid"
    else:
        n_centroids_per_label = 10
        prediction_size = 400
        note = f"{n}-class: 10 centroids/label × {n} labels"
    pool_size = 5000
    return HugePredictionDefaults(
        prediction_size=prediction_size,
        batch_size=batch_size,
        n_centroids_per_label=n_centroids_per_label,
        pool_size=pool_size,
        note=note,
    )


def validate_or_raise(
    ctx: DatasetContext,
    *,
    prediction_size: int,
    batch_size: int,
    n_centroids_per_label: int,
) -> tuple[int, int]:
    """Run DatasetContext validation; on failure, attach suggested defaults to the error."""
    try:
        return ctx.validate_prediction_params(
            prediction_size=prediction_size,
            batch_size=batch_size,
            n_centroids_per_label=n_centroids_per_label,
        )
    except ValueError as exc:
        sug = suggest_huge_prediction_defaults(ctx)
        raise ValueError(
            f"{exc}. Suggested for {ctx.dataset_name!r} ({ctx.n_labels} labels): "
            f"prediction_size={sug.prediction_size}, batch_size={sug.batch_size}, "
            f"n_centroids_per_label={sug.n_centroids_per_label} ({sug.note})"
        ) from exc


def load_and_suggest(dataset_name: str) -> tuple[DatasetContext, HugePredictionDefaults]:
    ctx = load_dataset_context(dataset_name)
    return ctx, suggest_huge_prediction_defaults(ctx)
