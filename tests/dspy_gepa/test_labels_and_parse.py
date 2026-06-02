# pyright: basic
from __future__ import annotations

from src.dspy_gepa.batch_classifier import parse_predicted_labels
from src.dspy_gepa.labels import labels_for_dataset, normalize_label_for_dataset


def test_labels_for_implicit_hate() -> None:
    ids = ["HS", "Non-HS"]
    names = labels_for_dataset(dataset_name="implicit_hate", label_ids=ids)
    assert "hs" in names or "HS" in names


def test_parse_implicit_hate_json() -> None:
    raw = '["HS", "Non-HS"]'
    out = parse_predicted_labels(
        raw,
        batch_size=2,
        allowed_labels=["hs", "non-hs"],
        dataset_name="implicit_hate",
    )
    assert len(out) == 2
    assert out[0] in {"hs", "HS"}
    assert out[1] in {"non-hs", "Non-HS"}


def test_pubmed_id_and_name_same_canonical() -> None:
    a = normalize_label_for_dataset("2", dataset_name="pubmed_20k_rct")
    b = normalize_label_for_dataset("methods", dataset_name="pubmed_20k_rct")
    assert a == b == "methods"
