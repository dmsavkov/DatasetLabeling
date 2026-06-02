# pyright: basic
"""Prompt templates for prompt-eng experiments (multi-label probe, self-debate)."""

from __future__ import annotations

import json

from .baseline import BatchItem


def build_multilabel_confusion_messages(
    *,
    allowed_labels: list[str],
    items: list[BatchItem],
    few_shot: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    if not allowed_labels:
        raise ValueError("allowed_labels must be non-empty")
    if not items:
        raise ValueError("items must be non-empty")

    labels_str = ", ".join(json.dumps(l) for l in allowed_labels)
    few = few_shot or []
    few_block = ""
    if few:
        lines = []
        for t, lab in few:
            lines.append(f"- text: {json.dumps(t)}\n  labels: {json.dumps([lab])}")
        few_block = "\n\nFew-shot examples (single label each):\n" + "\n".join(lines) + "\n"

    user_payload = {
        "allowed_labels": allowed_labels,
        "items": [{"id": it.id, "text": it.text} for it in items],
        "output_schema": [{"id": "string", "labels": ["string", "..."]}],
    }

    system = (
        "You are a strict JSON-only text classifier.\n"
        "For each item, return ALL labels from allowed_labels that genuinely apply to the text.\n"
        "If exactly one label applies, return a list with that one label.\n"
        "If none apply, return an empty list [].\n"
        "If you are uncertain between multiple labels, you may return multiple labels.\n"
        "Return ONLY a JSON array: [{\"id\": \"...\", \"labels\": [\"...\", ...]}, ...].\n"
        "No markdown. No extra keys."
    )

    user = (
        f"Allowed labels: [{labels_str}]\n"
        f"{few_block}\n"
        "Classify all items. Output JSON only.\n"
        + json.dumps(user_payload, ensure_ascii=True)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_self_debate_pass_a_messages(
    *,
    allowed_labels: list[str],
    items: list[BatchItem],
    few_shot: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    labels_str = ", ".join(json.dumps(l) for l in allowed_labels)
    few = few_shot or []
    few_block = ""
    if few:
        ex = "\n".join(
            [f"- text: {json.dumps(t)}\n  label: {json.dumps(l)}" for t, l in few]
        )
        few_block = f"\n\nFew-shot examples:\n{ex}\n"

    user_payload = {
        "allowed_labels": allowed_labels,
        "items": [{"id": it.id, "text": it.text} for it in items],
        "output_schema": [{"id": "string", "label": "string", "confidence": "number", "reasoning": "string"}],
    }

    system = (
        "You are a careful single-label classifier.\n"
        f"Allowed labels: [{labels_str}].\n"
        "Return ONLY a JSON array with one object per item:\n"
        '[{"id": "...", "label": "...", "confidence": 0.0-1.0, "reasoning": "brief"}]\n'
        "No markdown."
    )
    user = (
        f"{few_block}\n"
        "Classify each item (Pass A — initial answer).\n"
        + json.dumps(user_payload, ensure_ascii=True)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_self_debate_pass_b_messages(
    *,
    allowed_labels: list[str],
    items: list[BatchItem],
    pass_a_by_id: dict[str, dict[str, object]],
) -> list[dict[str, str]]:
    labels_str = ", ".join(json.dumps(l) for l in allowed_labels)
    critique_rows = []
    for it in items:
        a = pass_a_by_id.get(it.id, {})
        critique_rows.append(
            {
                "id": it.id,
                "text": it.text,
                "answer_a": {
                    "label": a.get("label"),
                    "confidence": a.get("confidence"),
                    "reasoning": a.get("reasoning"),
                },
            }
        )

    system = (
        "You are the same classifier, now in critique mode.\n"
        f"Allowed labels: [{labels_str}].\n"
        "Each Answer A may contain a flaw or overconfidence. Find flaws and revise if needed.\n"
        "Return ONLY a JSON array:\n"
        '[{"id": "...", "label": "...", "confidence": 0.0-1.0, "reasoning": "brief"}]\n'
        "No markdown."
    )
    user = (
        "Revise each Answer A (Pass B). Items:\n"
        + json.dumps(critique_rows, ensure_ascii=True)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
