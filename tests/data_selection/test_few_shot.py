# pyright: basic
from __future__ import annotations

import pandas as pd

from src.data_selection.few_shot import few_shot_examples_from_df
from src.data_selection.label_utils import DatasetContext, prompt_labels_for_dataset
from src.datasets.schema import SCHEMA


def test_few_shot_uses_prompt_labels_for_pubmed() -> None:
    ctx = DatasetContext(
        dataset_name="pubmed_20k_rct",
        label_ids=["0", "1", "2", "3", "4"],
        prompt_labels=prompt_labels_for_dataset(
            dataset_name="pubmed_20k_rct",
            label_ids=["0", "1", "2", "3", "4"],
        ),
        prompt_id="baseline_v1",
    )
    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "dataset_name": "pubmed_20k_rct",
                "text": "We enrolled patients.",
                "true_label": "2",
            }
        ]
    )
    examples = few_shot_examples_from_df(df, ctx)
    assert examples == [("We enrolled patients.", "methods")]


def test_few_shot_excludes_prediction_pool_ids() -> None:
    ctx = DatasetContext(
        dataset_name="tweet_eval_irony",
        label_ids=["0", "1"],
        prompt_labels=["0", "1"],
        prompt_id="baseline_v1",
    )
    df = pd.DataFrame(
        [
            {"sample_id": "a", "dataset_name": "t", "text": "t1", "true_label": "0"},
            {"sample_id": "b", "dataset_name": "t", "text": "t2", "true_label": "1"},
        ]
    )
    out = few_shot_examples_from_df(df, ctx, exclude_sample_ids=frozenset({"a"}))
    assert len(out) == 1
    assert out[0][0] == "t2"
