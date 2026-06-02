# pyright: basic
"""JSON parsing helpers for prompt-eng LLM experiments."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from src.prompts.baseline import BatchItem, extract_json_array, normalize_label, strip_markdown_fences

_THOUGHT_BLOCK_RE = re.compile(r"<thought>[\s\S]*?</thought>", re.IGNORECASE)
_FINAL_ANSWER_MARKER_RE = re.compile(r"final\s+answer\s*:?\s*", re.IGNORECASE)


def strip_thought_blocks(text: str) -> str:
    """Remove Gemma-style thought wrappers before JSON extraction."""
    t = str(text or "")
    t = _THOUGHT_BLOCK_RE.sub("", t)
    return t.strip()


def _score_json_array_candidate(payload: Any, raw: str) -> int:
    """Prefer batch arrays [{id, labels}, ...] over incidental [] fragments."""
    if not isinstance(payload, list):
        return -1
    score = len(raw)
    if not payload:
        return score
    if all(isinstance(row, dict) for row in payload):
        score += 5000 + len(payload) * 200
        if any("id" in row and ("labels" in row or "label" in row) for row in payload):
            score += 5000
    return score


def _find_best_balanced_json_array(text: str) -> str | None:
    """Best top-level JSON array in text (largest batch-shaped array, not inner [])."""
    candidates: list[str] = []
    s = text
    for i, ch in enumerate(s):
        if ch != "[":
            continue
        depth = 0
        in_str = False
        escape = False
        for j in range(i, len(s)):
            c = s[j]
            if in_str:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
                continue
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    candidates.append(s[i : j + 1])
                    break
    best_raw: str | None = None
    best_score = -1
    for cand in candidates:
        try:
            payload = json.loads(cand)
        except json.JSONDecodeError:
            continue
        sc = _score_json_array_candidate(payload, cand)
        if sc > best_score:
            best_score = sc
            best_raw = cand
    return best_raw


def extract_multilabel_json_text(raw: str) -> str:
    """
    Pull the batch JSON array from model output that may include CoT / prose prefixes.
    Prefers content after 'Final Answer:', then the last balanced [...] array.
    """
    cleaned = strip_thought_blocks(strip_markdown_fences(str(raw or "")))
    marker = _FINAL_ANSWER_MARKER_RE.search(cleaned)
    if marker:
        cleaned = cleaned[marker.end() :].strip()
    balanced = _find_best_balanced_json_array(cleaned)
    if balanced is not None:
        return balanced
    try:
        return extract_json_array(cleaned)
    except ValueError:
        any_re = re.compile(r"(\[[\s\S]*\])")
        matches = list(any_re.finditer(cleaned))
        if matches:
            return matches[-1].group(1)
        raise ValueError("Could not find JSON array in model output") from None


def _load_json_payload(text: str) -> Any:
    return json.loads(extract_multilabel_json_text(text))


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("predictions", "items", "data", "results"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError("Expected JSON array of per-item objects")
    return [r for r in payload if isinstance(r, dict)]


def parse_multilabel_batch(
    text: str,
    *,
    allowed_labels: list[str],
    label_normalizer: Callable[[str], str] | None = None,
) -> dict[str, list[str]]:
    """
    Parse `[{"id": "...", "labels": ["a", "b"]}, ...]`.
    Returns item_id -> normalized label list (may be empty).
    """

    norm = label_normalizer or (lambda x: str(x).strip().lower())
    allowed_norm = {norm(a): a for a in allowed_labels}
    rows = _rows_from_payload(_load_json_payload(text))
    out: dict[str, list[str]] = {}
    for row in rows:
        item_id = row.get("id")
        if item_id is None:
            continue
        raw_labels = row.get("labels", row.get("label", []))
        labels: list[str] = []
        if isinstance(raw_labels, str):
            raw_labels = [raw_labels]
        if isinstance(raw_labels, list):
            for lab in raw_labels:
                if lab is None:
                    continue
                n = norm(str(lab))
                if n in allowed_norm:
                    labels.append(allowed_norm[n])
        out[str(item_id)] = labels
    return out


def clamp01(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def parse_self_debate_batch(
    text: str,
    *,
    allowed_labels: list[str],
    label_normalizer: Callable[[str], str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Parse `[{"id": "...", "label": "...", "confidence": 0.9, "reasoning": "..."}, ...]`.
    """

    norm = label_normalizer or (lambda x: str(x).strip().lower())
    allowed_norm = {norm(a): a for a in allowed_labels}
    rows = _rows_from_payload(_load_json_payload(text))
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get("id")
        if item_id is None:
            continue
        lab_raw = row.get("label", row.get("pred_label"))
        lab: str | None = None
        if lab_raw is not None:
            n = norm(str(lab_raw))
            lab = allowed_norm.get(n)
        out[str(item_id)] = {
            "label": lab,
            "confidence": clamp01(row.get("confidence", 0.0)),
            "reasoning": str(row.get("reasoning", row.get("reason", ""))).strip(),
        }
    return out


def multilabel_confusion_kind(pred_labels: list[str]) -> str:
    """``none`` = 0 labels; ``single`` = 1 label; ``multi`` = 2+ labels (ambiguous)."""
    n = len(pred_labels)
    if n == 0:
        return "none"
    if n == 1:
        return "single"
    return "multi"


def confusion_from_label_lists(pred_labels: list[str]) -> tuple[bool, str | None]:
    """
    Single-label gold path: confused when model returns != 1 label.
    Returns (is_confusing, pred_label for scoring).
    """
    if len(pred_labels) != 1:
        return True, None
    return False, pred_labels[0]


def confusion_from_debate(a_label: str | None, b_label: str | None) -> bool:
    if a_label is None or b_label is None:
        return True
    if a_label == "error" or b_label == "error":
        return True
    return a_label != b_label
