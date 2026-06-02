from __future__ import annotations

from src.eval.label_compare import labels_equal_for_metrics, normalize_label_scalar


def test_normalize_label_scalar_strips_float_suffix() -> None:
    assert normalize_label_scalar(4.0) == "4"
    assert normalize_label_scalar("4.0") == "4"


def test_pubmed_label_zero_not_treated_as_missing() -> None:
    from src.data_selection.label_utils import normalize_pubmed_label

    assert normalize_pubmed_label(0) == "background"
    assert normalize_pubmed_label("0") == "background"


def test_pubmed_labels_match_id_name_and_float() -> None:
    assert labels_equal_for_metrics("4", "4.0", dataset_name="pubmed_20k_rct")
    assert labels_equal_for_metrics(4, "conclusions", dataset_name="pubmed_20k_rct")
    assert labels_equal_for_metrics("1", "objective", dataset_name="pubmed_20k_rct")
    assert not labels_equal_for_metrics("3", "1", dataset_name="pubmed_20k_rct")
