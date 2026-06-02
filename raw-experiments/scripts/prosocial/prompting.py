from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI

from prosocial.constants import FLASH_MODEL
from src.data import normalize_label


def extract_first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found")

    depth = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError("No complete JSON object found")
    return json.loads(text[start : end + 1])


def normalize_model_label(value: Any) -> str:
    return normalize_label(value)


def is_unanimous_annotations(values: Any) -> tuple[bool, str | None]:
    if not isinstance(values, list) or not values:
        return False, None
    normalized = [normalize_model_label(v) for v in values]
    first = normalized[0]
    return all(v == first for v in normalized), first


def format_reason_text(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return " | ".join(str(v).strip() for v in values if str(v).strip())


def build_optimizer_examples(
    df: pd.DataFrame,
    *,
    count: int,
    labels: list[str],
    seed: int,
    include_reasons: bool,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    per_label = max(1, count // max(1, len(labels)))
    selected_parts: list[pd.DataFrame] = []

    for label in labels:
        part = df[df["safety_label"] == label]
        if len(part) == 0:
            continue
        idx = rng.choice(part.index.to_numpy(), size=min(per_label, len(part)), replace=False)
        selected_parts.append(part.loc[idx])

    selected_df = pd.concat(selected_parts, axis=0) if selected_parts else df.head(0)
    if len(selected_df) < count:
        remaining = df.loc[~df.index.isin(selected_df.index)]
        n_extra = min(count - len(selected_df), len(remaining))
        if n_extra > 0:
            idx = rng.choice(remaining.index.to_numpy(), size=n_extra, replace=False)
            selected_df = pd.concat([selected_df, remaining.loc[idx]], axis=0)

    examples: list[dict[str, Any]] = []
    for row in selected_df.reset_index(drop=True).to_dict(orient="records"):
        ex = {
            "source_index": int(row["source_index"]),
            "context": str(row["context"]),
            "label": str(row["safety_label"]),
        }
        if include_reasons:
            ex["annotation_reasons"] = format_reason_text(row.get("safety_annotation_reasons", []))
            ex["annotations_raw"] = row.get("safety_annotations", [])
        examples.append(ex)
    return examples


def build_prompt_optimizer_request(
    *,
    examples: list[dict[str, Any]],
    labels: list[str],
    rubric: str,
    assertions: str,
    objective: str,
) -> str:
    return (
        "You are optimizing a high-performance safety-labeling prompt for Gemma.\n"
        f"Objective: {objective}\n"
        f"Allowed labels: {labels}\n\n"
        "Use this rubric exactly as the safety policy:\n"
        f"{rubric}\n\n"
        "Your prompt must:\n"
        "1) include concise decision rules,\n"
        "2) use few-shot examples,\n"
        "3) preserve strict JSON output behavior.\n\n"
        "Required output assertions:\n"
        f"{assertions}\n\n"
        "Return strictly one JSON object with schema:\n"
        "{\"optimized_prompt\": \"string\"}\n\n"
        "Training examples:\n"
        f"{json.dumps(examples, ensure_ascii=True)}"
    )


def optimize_prompt_with_flash(
    client: OpenAI,
    *,
    examples: list[dict[str, Any]],
    labels: list[str],
    rubric: str,
    assertions: str,
    objective: str,
) -> str:
    response = client.chat.completions.create(
        model=FLASH_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert prompt engineer for safety classification."},
            {
                "role": "user",
                "content": build_prompt_optimizer_request(
                    examples=examples,
                    labels=labels,
                    rubric=rubric,
                    assertions=assertions,
                    objective=objective,
                ),
            },
        ],
        temperature=0.2,
    )
    payload = extract_first_json_object(response.choices[0].message.content or "")
    value = str(payload.get("optimized_prompt", "")).strip()
    if not value:
        raise ValueError("Prompt optimizer did not return optimized_prompt")
    return value


def optimize_rubric_with_flash(
    client: OpenAI,
    *,
    examples: list[dict[str, Any]],
    labels: list[str],
    current_rubric: str,
) -> str:
    user_prompt = (
        "You are improving a safety-classification rubric for better class boundary calibration.\n"
        f"Allowed labels: {labels}\n\n"
        "Current rubric:\n"
        f"{current_rubric}\n\n"
        "Revise the rubric for clearer boundaries and tie-break rules while preserving label semantics.\n"
        "Return exactly one JSON object:\n"
        "{\"optimized_rubric\": \"string\"}\n\n"
        "Calibration examples:\n"
        f"{json.dumps(examples, ensure_ascii=True)}"
    )
    response = client.chat.completions.create(
        model=FLASH_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in rubric calibration for safety classification."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    payload = extract_first_json_object(response.choices[0].message.content or "")
    value = str(payload.get("optimized_rubric", "")).strip()
    if not value:
        raise ValueError("Rubric optimizer did not return optimized_rubric")
    return value


def build_prediction_prompt(
    *,
    optimized_prompt: str,
    labels: list[str],
    context: str,
    assertion_text: str,
    fewshot_examples: list[dict[str, Any]] | None = None,
    extracted_statements: list[str] | None = None,
) -> str:
    parts: list[str] = [optimized_prompt.strip(), f"Allowed labels: {labels}"]
    if assertion_text.strip():
        parts.append("Assertions:\n" + assertion_text.strip())

    if fewshot_examples:
        lines = [f"- Context: {str(ex['context'])}\n  Label: {str(ex['label'])}" for ex in fewshot_examples]
        parts.append("Few-shot examples:\n" + "\n".join(lines))

    if extracted_statements:
        statements = [s.strip() for s in extracted_statements if s.strip()]
        if statements:
            parts.append("Extracted risk statements:\n" + "\n".join(f"- {s}" for s in statements))

    parts.append("Return ONLY one JSON object: {\"label\": \"<one_allowed_label>\"}.")
    parts.append("Do not include markdown or explanation.")
    parts.append("Context:\n" + context)
    return "\n\n".join(parts)


def parse_prediction_label(raw_text: str, labels: list[str]) -> str:
    try:
        payload = extract_first_json_object(raw_text)
        value = normalize_model_label(payload.get("label", ""))
    except Exception:
        return "parse_error"
    return value if value in labels else "parse_error"
