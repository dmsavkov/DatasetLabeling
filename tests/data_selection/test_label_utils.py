# pyright: basic
from __future__ import annotations

from src.data_selection.label_utils import DatasetContext, per_label_quotas, prompt_labels_for_dataset


def test_per_label_quotas_even() -> None:
    q = per_label_quotas(10, ["a", "b"])
    assert sum(q.values()) == 10
    assert q == {"a": 5, "b": 5}


def test_per_label_quotas_remainder() -> None:
    q = per_label_quotas(11, ["a", "b", "c"])
    assert sum(q.values()) == 11
    assert set(q.values()) == {3, 4}


def test_pubmed_labels_match() -> None:
    ctx = DatasetContext(
        dataset_name="pubmed_20k_rct",
        label_ids=["0", "1", "2", "3", "4"],
        prompt_labels=prompt_labels_for_dataset(
            dataset_name="pubmed_20k_rct",
            label_ids=["0", "1", "2", "3", "4"],
        ),
        prompt_id="baseline_v1",
    )
    assert ctx.labels_match("2", "methods")
    assert ctx.canonicalize("2") == "methods"


def test_dataset_context_validate_params() -> None:
    ctx = DatasetContext(
        dataset_name="tweet_eval_irony",
        label_ids=["0", "1"],
        prompt_labels=["0", "1"],
        prompt_id="baseline_v1",
    )
    n_centroids, spc = ctx.validate_prediction_params(
        prediction_size=100,
        batch_size=5,
        n_centroids_per_label=10,
    )
    assert n_centroids == 20
    assert spc == 5
