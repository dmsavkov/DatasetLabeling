from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


DEFAULT_BATCH_SIZE = 5


@dataclass(frozen=True, slots=True)
class BatchItem:
    id: str
    text: str


def build_llm_batch_messages(
    *,
    allowed_labels: list[str],
    items: list[BatchItem],
    few_shot: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """
    Returns OpenAI-style chat messages for batch classification.
    Expects strict JSON output: a list of {id, label}.
    """

    if not allowed_labels:
        raise ValueError("allowed_labels must be non-empty")
    if not items:
        raise ValueError("items must be non-empty")

    labels_str = ", ".join([json.dumps(l) for l in allowed_labels])
    few = few_shot or []
    few_shot_block = ""
    if few:
        examples = "\n".join([f"- text: {json.dumps(t)}\n  label: {json.dumps(l)}" for t, l in few])
        few_shot_block = f"\n\nExamples:\n{examples}\n"

    user_payload = {
        "allowed_labels": allowed_labels,
        "items": [{"id": it.id, "text": it.text} for it in items],
        "output_schema": [{"id": "string", "label": "string"}],
    }

    system = (
        "You are a strict JSON-only classifier.\n"
        "Choose exactly one label from allowed_labels for each item.\n"
        "Return ONLY a JSON array of objects: [{\"id\":..., \"label\":...}, ...].\n"
        "No markdown. No explanations. No extra keys."
    )

    user = (
        f"Allowed labels: [{labels_str}]\n"
        f"{few_shot_block}\n"
        "Classify the following items. Output JSON only.\n"
        + json.dumps(user_payload, ensure_ascii=True)
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def strip_markdown_fences(text: str) -> str:
    t = text.strip()
    # ```json ... ``` or ``` ... ```
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def extract_json_array(text: str) -> str:
    t = strip_markdown_fences(text)
    m = _JSON_ARRAY_RE.search(t)
    if not m:
        raise ValueError("Could not find JSON array in model output")
    return m.group(0)


def normalize_label(label: str, allowed_labels: list[str]) -> str | None:
    if label in allowed_labels:
        return label
    norm = str(label).strip()
    # simple normalization: whitespace + lower
    allowed_norm = {a.strip().lower(): a for a in allowed_labels}
    hit = allowed_norm.get(norm.lower())
    return hit


def parse_batch_predictions(
    text: str,
    *,
    allowed_labels: list[str],
) -> dict[str, str | None]:
    """
    Returns mapping item_id -> normalized_label (or None if invalid).
    """

    def _try_parse_payload(obj: Any) -> dict[str, str | None]:
        payload = obj
        if isinstance(payload, dict):
            # Common wrappers that still contain a list of {id,label} rows.
            for key in ("predictions", "items", "data", "results"):
                if key in payload and isinstance(payload[key], list):
                    payload = payload[key]
                    break

            # Mapping form: { "<id>": "<label>", ... }
            if isinstance(payload, dict):
                if all(isinstance(k, str) for k in payload.keys()) and all(
                    isinstance(v, (str, int, float)) or v is None for v in payload.values()
                ):
                    rows = []
                    for k, v in payload.items():
                        rows.append({"id": k, "label": v})
                    payload = rows

        if not isinstance(payload, list):
            raise ValueError("Could not parse batch predictions into a list")

        out: dict[str, str | None] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            item_id = row.get("id")
            lab = row.get("label", row.get("pred_label", row.get("prediction")))
            if item_id is None:
                continue
            out[str(item_id)] = normalize_label(str(lab), allowed_labels) if lab is not None else None
        return out

    # 1) Fast path: JSON array inside text.
    try:
        arr_text = extract_json_array(text)
        payload = json.loads(arr_text)
        return _try_parse_payload(payload)
    except ValueError:
        pass

    # 2) Fallback: try to extract the first JSON object/array blob.
    # This is deliberately permissive to handle models that don't follow the
    # “array-only” contract (e.g. missing system prompt).
    t = strip_markdown_fences(text)
    any_re = re.compile(r"(\[[\s\S]*\]|\{[\s\S]*\})")
    m = any_re.search(t)
    if not m:
        raise ValueError("Could not find JSON payload in model output")

    payload = json.loads(m.group(1))
    return _try_parse_payload(payload)

