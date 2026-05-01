from src.datasets.builders.banking10 import select_top_labels


def test_select_top_labels_is_deterministic_with_ties() -> None:
    counts = {"b": 5, "a": 5, "c": 10}
    # c first (10), then a before b due to lex tie-break
    assert select_top_labels(counts, 3) == ["c", "a", "b"]


def test_select_top_labels_respects_k() -> None:
    counts = {"x": 3, "y": 2, "z": 1}
    assert select_top_labels(counts, 2) == ["x", "y"]

