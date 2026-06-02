# pyright: basic
"""DSPy batched text classifier (dataset-agnostic)."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

import dspy

from src.dspy_gepa.batching import TextBatch, batch_to_numbered_text
from src.dspy_gepa.labels import labels_for_dataset, normalize_label_for_dataset

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _make_batch_signature(*, allowed_labels: list[str], dataset_name: str) -> type[dspy.Signature]:
    labels_blob = json.dumps(allowed_labels)
    doc = (
        f"Classify numbered input lines into labels for dataset {dataset_name!r}. "
        f"Each line is one item. Allowed labels: {labels_blob}."
    )

    class BatchTextSignature(dspy.Signature):  # type: ignore[misc]
        __doc__ = doc
        input_texts = dspy.InputField(desc="Numbered list of texts (one per line).")
        predicted_labels = dspy.OutputField(
            desc=(
                f"Strict JSON array of label strings, length must match input line count. "
                f"Each element must be one of: {labels_blob}."
            )
        )

    return BatchTextSignature


class BatchTextClassifier(dspy.Module):
    """Chain-of-thought batch classifier; compile/save/load like other DSPy modules."""

    def __init__(self, *, dataset_name: str, allowed_labels: list[str]) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.allowed_labels = list(allowed_labels)
        sig = _make_batch_signature(allowed_labels=self.allowed_labels, dataset_name=dataset_name)
        self.predictor = dspy.ChainOfThought(sig)

    def forward(self, input_texts: str, **kwargs: Any) -> dspy.Prediction:
        return self.predictor(input_texts=input_texts)


def parse_predicted_labels(
    raw: object,
    *,
    batch_size: int,
    allowed_labels: list[str],
    dataset_name: str,
) -> list[str]:
    allowed_set = {normalize_label_for_dataset(x, dataset_name=dataset_name) for x in allowed_labels}

    if isinstance(raw, list):
        labels = [normalize_label_for_dataset(x, dataset_name=dataset_name) for x in raw]
        if len(labels) == batch_size:
            return [lab if lab in allowed_set else "error" for lab in labels]

    text = str(raw if raw is not None else "")
    match = _JSON_ARRAY_RE.search(text)
    blob = match.group(0) if match else text
    data: object = None
    try:
        data = json.loads(blob)
    except Exception:
        try:
            data = ast.literal_eval(blob)
        except Exception:
            data = None

    if isinstance(data, list):
        labels = [normalize_label_for_dataset(x, dataset_name=dataset_name) for x in data]
    else:
        labels = ["error"] * batch_size

    if len(labels) != batch_size:
        labels = (labels + ["error"] * batch_size)[:batch_size]
    return [lab if lab in allowed_set else "error" for lab in labels]


def examples_from_batches(
    batches: list[TextBatch],
    *,
    allowed_labels: list[str],
) -> list[dspy.Example]:
    examples: list[dspy.Example] = []
    for batch in batches:
        if batch.size == 0:
            continue
        ex = dspy.Example(
            input_texts=batch_to_numbered_text(batch),
            target_labels=[r.label_name for r in batch.rows],
            sample_ids=[r.sample_id for r in batch.rows],
            batch_id=int(batch.batch_id),
        ).with_inputs("input_texts")
        examples.append(ex)
    del allowed_labels
    return examples


def batch_metric_factory(
    *,
    batch_size: int,
    allowed_labels: list[str],
    dataset_name: str,
):
    """DSPy metric: mean per-sentence accuracy within the batch (0..1)."""

    def metric(
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace: object | None = None,
        pred_name: str | None = None,
        pred_trace: object | None = None,
    ) -> dspy.Prediction:
        del trace, pred_name, pred_trace
        gold_labels = [
            normalize_label_for_dataset(x, dataset_name=dataset_name) for x in gold.target_labels
        ]
        pred_labels = parse_predicted_labels(
            getattr(pred, "predicted_labels", None),
            batch_size=batch_size,
            allowed_labels=allowed_labels,
            dataset_name=dataset_name,
        )
        correct = sum(1 for g, p in zip(gold_labels, pred_labels, strict=True) if g == p)
        score = correct / float(batch_size)
        if score >= 1.0:
            feedback = "Perfect batch."
        else:
            errs = [
                f"Pos {i + 1}: expected {gold_labels[i]}, got {pred_labels[i]}"
                for i in range(batch_size)
                if gold_labels[i] != pred_labels[i]
            ]
            feedback = f"Partial ({correct}/{batch_size}). " + "; ".join(errs)
        return dspy.Prediction(score=float(score), feedback=feedback)

    return metric


def classifier_for_dataset(*, dataset_name: str, label_ids: list[str]) -> BatchTextClassifier:
    allowed = labels_for_dataset(dataset_name=dataset_name, label_ids=label_ids)
    return BatchTextClassifier(dataset_name=dataset_name, allowed_labels=allowed)
