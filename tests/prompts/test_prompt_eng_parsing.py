# pyright: basic
from __future__ import annotations

import pytest

from src.prompts.parsing import (
    confusion_from_debate,
    confusion_from_label_lists,
    extract_multilabel_json_text,
    multilabel_confusion_kind,
    parse_multilabel_batch,
    parse_self_debate_batch,
    strip_thought_blocks,
)

ALLOWED = ["methods", "results", "conclusions"]


def test_strip_thought_blocks() -> None:
    raw = '<thought>internal</thought>\n[{"id": "1", "labels": ["methods"]}]'
    assert "internal" not in strip_thought_blocks(raw)
    assert "[{" in strip_thought_blocks(raw)


def test_parse_multilabel_batch_with_thought() -> None:
    text = (
        "<thought>thinking</thought>\n"
        '[{"id": "a", "labels": ["Methods"]}, {"id": "b", "labels": []}]'
    )
    out = parse_multilabel_batch(text, allowed_labels=ALLOWED)
    assert out["a"] == ["methods"]
    assert out["b"] == []


def test_confusion_from_label_lists() -> None:
    assert confusion_from_label_lists([]) == (True, None)
    assert confusion_from_label_lists(["methods"]) == (False, "methods")
    assert confusion_from_label_lists(["methods", "results"]) == (True, None)


def test_multilabel_confusion_kind() -> None:
    assert multilabel_confusion_kind([]) == "none"
    assert multilabel_confusion_kind(["methods"]) == "single"
    assert multilabel_confusion_kind(["methods", "results"]) == "multi"


def test_extract_multilabel_json_after_external_cot() -> None:
    raw = (
        "Logical Process: item 0 looks like results.\n"
        'Final Answer: [{"id": "0", "labels": ["results"]}, {"id": "1", "labels": []}]\n'
    )
    text = extract_multilabel_json_text(raw)
    out = parse_multilabel_batch(text, allowed_labels=ALLOWED)
    assert out["0"] == ["results"]
    assert out["1"] == []


def test_extract_multilabel_json_last_array_wins() -> None:
    raw = (
        "Here is a draft: [{\"id\": \"x\", \"labels\": [\"bad\"]}] — ignore.\n"
        'Final Answer: [{"id": "a", "labels": ["methods"]}]'
    )
    out = parse_multilabel_batch(raw, allowed_labels=ALLOWED)
    assert out["a"] == ["methods"]


def test_parse_self_debate_batch() -> None:
    text = '[{"id": "x", "label": "results", "confidence": 1.5, "reasoning": "ok"}]'
    out = parse_self_debate_batch(text, allowed_labels=ALLOWED)
    assert out["x"]["label"] == "results"
    assert out["x"]["confidence"] == 1.0


def test_confusion_from_debate() -> None:
    assert confusion_from_debate("methods", "methods") is False
    assert confusion_from_debate("methods", "results") is True
    assert confusion_from_debate(None, "results") is True
    assert confusion_from_debate("error", "results") is True
