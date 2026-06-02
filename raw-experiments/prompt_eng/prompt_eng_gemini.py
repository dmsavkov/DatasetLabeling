"""
Minimal google.genai helpers for prompt_eng scripts (no OpenAI-compat, no DSPy).

Uses GOOGLE_API_KEY from prompt_eng_common. Retries: generate_with_retries.

Some models reject system_instruction; on recoverable API errors we retry once with the system
text merged into the user message (single block).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions, HttpRetryOptions
from loguru import logger

import prompt_eng_common as pec

_HTTP_RETRY_INITIAL_DELAY_S = 2.0
_HTTP_RETRY_STATUS_CODES: tuple[int, ...] = (429, 500, 503, 504)


def get_genai_client(*, http_retry_attempts: int = 5) -> genai.Client:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required for google.genai.")
    attempts = max(1, int(http_retry_attempts))
    retry_options = HttpRetryOptions(
        attempts=attempts,
        initial_delay=_HTTP_RETRY_INITIAL_DELAY_S,
        http_status_codes=list(_HTTP_RETRY_STATUS_CODES),
    )
    http_options = HttpOptions(retry_options=retry_options)
    return genai.Client(api_key=pec.GOOGLE_API_KEY, http_options=http_options)


def merge_system_user(system_instruction: str, user_text: str) -> str:
    """Single user turn: instructions first, then task."""
    return (
        "### Instructions (follow before answering)\n"
        + (system_instruction or "").strip()
        + "\n\n### Task\n"
        + (user_text or "").strip()
    )


def _should_fallback_merge(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = (
        "json",
        "system",
        "system_instruction",
        "invalid argument",
        "400",
        "not supported",
        "unsupported",
        "response_schema",
        "response_mime",
        "mime",
    )
    return any(x in msg for x in needles)


def _parts_text(resp: Any) -> str:
    chunks: list[str] = []
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        return ""
    c0 = cands[0]
    content = getattr(c0, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return ""
    for p in parts:
        if getattr(p, "thought", False):
            continue
        tx = getattr(p, "text", None) or ""
        if tx:
            chunks.append(str(tx))
    return "\n".join(chunks).strip()


def _raw_generate(
    client: genai.Client,
    *,
    model: str,
    contents: str,
    system_instruction: str | None,
    temperature: float,
    top_p: float | None,
    max_output_tokens: int | None,
) -> str:
    kwargs: dict[str, Any] = {"temperature": float(temperature)}
    if top_p is not None:
        kwargs["top_p"] = float(top_p)
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = int(max_output_tokens)
    cfg = GenerateContentConfig(**kwargs)
    cfg_dump = cfg.model_dump(exclude_none=True)
    if system_instruction:
        cfg_dump["system_instruction"] = system_instruction
    cfg2 = GenerateContentConfig.model_validate(cfg_dump)
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=cfg2,
    )
    return _parts_text(resp)


def generate_content_text(
    client: genai.Client,
    *,
    model: str,
    user_text: str,
    system_instruction: str | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
    max_output_tokens: int | None = None,
) -> str:
    """
    Synchronous generate_content. Uses system_instruction when provided; on recoverable
    errors merges system into the user message and retries once.
    """
    if not system_instruction:
        return _raw_generate(
            client,
            model=model,
            contents=user_text,
            system_instruction=None,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )

    try:
        return _raw_generate(
            client,
            model=model,
            contents=user_text,
            system_instruction=system_instruction,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
    except Exception as e:
        if _should_fallback_merge(e):
            logger.warning("genai merged fallback after error: {}", e)
            merged = merge_system_user(system_instruction, user_text)
            return _raw_generate(
                client,
                model=model,
                contents=merged,
                system_instruction=None,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            )
        raise


def generate_with_retries(
    client: genai.Client,
    *,
    model: str,
    user_text: str,
    system_instruction: str | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
    max_output_tokens: int | None = None,
    max_retries: int,
    label: str = "",
) -> str:
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.perf_counter()
            out = generate_content_text(
                client,
                model=model,
                user_text=user_text,
                system_instruction=system_instruction,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            )
            logger.debug("{} ok in {:.2f}s (chars={})", label or model, time.perf_counter() - t0, len(out))
            return out
        except Exception as e:
            last = e
            logger.warning("{} attempt {}/{} failed: {}", label or model, attempt, max_retries, e)
            time.sleep(min(8.0, 0.5 * attempt))
    raise RuntimeError(f"genai failed after {max_retries} attempts: {last}")


def extract_json_object(raw: str) -> dict[str, Any] | None:
    raw = (raw or "").strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    blob = m.group(0)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def extract_json_list(raw: str) -> list[Any] | None:
    m = re.search(r"\[[\s\S]*\]", raw or "")
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def labels_csv() -> str:
    return ", ".join(pec.VALID_LABELS)


# --- Copy-paste valid JSON one-liners for prompts (must parse with json.loads) ---

EXAMPLE_TRACE_JSON = json.dumps(
    {
        "reasoning": "- Step one\n- Step two\n- Step three\n- Step four",
        "label": "methods",
        "confidence": 75,
    },
    ensure_ascii=False,
)

EXAMPLE_QUESTIONS_JSON = json.dumps(
    {"questions": ["Does the sentence describe a protocol?", "Does it report measured outcomes?", "Who was studied?"]},
    ensure_ascii=False,
)

EXAMPLE_BLIND_ANSWERS_JSON = json.dumps(
    {
        "answers": [
            {"question": "Does the sentence describe a protocol?", "answer": "Yes, it describes recruitment."},
            {"question": "Does it report measured outcomes?", "answer": "not stated"},
        ]
    },
    ensure_ascii=False,
)

EXAMPLE_FINAL_JUDGE_JSON = json.dumps(
    {"final_label": "methods", "synthesis": "Blind answers support the draft."},
    ensure_ascii=False,
)

EXAMPLE_DINCO_PASS1_JSON = json.dumps(
    {
        "multilabel": ["methods", "results"],
        "primary_label": "methods",
        "critical_reasoning": "Compare each role...",
        "claims": [
            {"id": "c0", "label": "methods", "text": "This sentence primarily describes the study protocol."},
            {"id": "c1", "label": None, "text": "None of the standard labels fit this sentence."},
        ],
    },
    ensure_ascii=False,
)

EXAMPLE_DINCO_SCORES_JSON = json.dumps({"scores": {"c0": 82, "c1": 15}}, ensure_ascii=False)

EXAMPLE_PASS1_TOPTWO_JSON = json.dumps(
    {
        "multilabel": ["results"],
        "top_two": [{"label": "results", "probability": 0.62}, {"label": "methods", "probability": 0.21}],
        "notes": "Empirical wording.",
    },
    ensure_ascii=False,
)

EXAMPLE_CRITIC_JSON = json.dumps(
    {"critique": "Short attack on the hypothesis.", "recommended_label": "results", "confidence": 70},
    ensure_ascii=False,
)

EXAMPLE_RESOLVER_JSON = json.dumps(
    {"final_label": "results", "resolution_notes": "Two critics agree."},
    ensure_ascii=False,
)

EXAMPLE_TOP2_SINGLE_JSON = json.dumps(
    {
        "critique_of_label_a": "Why label A may fail.",
        "critique_of_label_b": "Why label B may fail.",
        "alternatives_considered": "Other labels briefly.",
        "final_label": "results",
    },
    ensure_ascii=False,
)
