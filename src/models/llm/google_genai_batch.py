# pyright: basic
"""
Batch LLM predictor using the official `google.genai` SDK (Gemini / Gemma on AI Studio).

Uses `Client(...).aio.models.generate_content` for async I/O. Captures optional thinking/reasoning
parts and usage_metadata (prompt vs candidates vs thoughts tokens).

`out_tokens` on Usage = candidates_token_count + thoughts_token_count (when present), so billing-style
totals include reasoning overhead; see `raw["usage_metadata"]` for the breakdown.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Literal

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig, ThinkingLevel
from loguru import logger

from src.models.clients.dispatch import _google_genai_api_key
from src.models.clients.registry import model_supports_system_prompt
from src.models.interfaces import Prediction, Usage, split_call_usage_across_rows
from src.models.llm.openai_compat_chat_batch import OpenAICompatChatBatchPredictor
from src.prompts.baseline import BatchItem, DEFAULT_BATCH_SIZE, parse_batch_predictions
from src.prompts.registry import get_prompt

ThinkingLevelStr = Literal["off", "low", "high"]

_THINKING_MAP: dict[ThinkingLevelStr, ThinkingLevel | None] = {
    "off": None,
    "low": ThinkingLevel.LOW,
    "high": ThinkingLevel.HIGH,
}

# SDK HTTP-layer retries (rate limits / transient server errors).
_HTTP_RETRY_INITIAL_DELAY_S = 2.0
_HTTP_RETRY_STATUS_CODES: tuple[int, ...] = (429, 500, 503, 504)


def _make_google_genai_client(*, api_key: str, http_retry_attempts: int) -> genai.Client:
    attempts = max(1, int(http_retry_attempts))
    retry_options = genai.types.HttpRetryOptions(
        attempts=attempts,
        initial_delay=_HTTP_RETRY_INITIAL_DELAY_S,
        http_status_codes=list(_HTTP_RETRY_STATUS_CODES),
    )
    http_options = genai.types.HttpOptions(retry_options=retry_options)
    return genai.Client(api_key=api_key, http_options=http_options)


def _predictions_from_genai_response(
    items: list[BatchItem],
    *,
    resp: Any,
    allowed_labels: list[str],
    attempt: int,
    with_thinking: bool,
    elapsed_s: float,
) -> dict[str, Prediction]:
    answer_text, thought_text, part_debug = _parts_from_response(resp)
    usage, raw_um = _usage_from_response(resp)
    per_row_usage = split_call_usage_across_rows(usage, len(items))
    mapping = parse_batch_predictions(answer_text, allowed_labels=allowed_labels)
    reason = thought_text if thought_text else None
    raw_base: dict[str, Any] = {
        "attempt": attempt,
        "with_thinking": with_thinking,
        "answer_text_preview": answer_text[:2000],
        "thought_text_preview": thought_text[:2000] if thought_text else None,
        "parts": part_debug,
        "usage_metadata": raw_um,
        "response_meta": _response_meta(resp),
        "elapsed_s": float(elapsed_s),
    }
    return {
        it.id: Prediction(
            pred_label=mapping.get(it.id),
            confidence=None,
            reason=reason,
            probs=None,
            usage=per_row_usage[idx],
            raw=raw_base,
        )
        for idx, it in enumerate(items)
    }


def _predictions_all_failed(items: list[BatchItem], *, error: Exception | None) -> dict[str, Prediction]:
    msg = str(error) if error else "unknown"
    raw_err = {"error": msg}
    return {
        it.id: Prediction(
            pred_label=None,
            confidence=None,
            reason=None,
            probs=None,
            usage=Usage(None, None),
            raw=raw_err,
        )
        for it in items
    }


@dataclass(frozen=True, slots=True)
class GoogleGenaiBatchParams:
    model_id: str
    prompt_id: str = "baseline_v1"
    few_shot: list[tuple[str, str]] | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    max_concurrency: int = 5
    temperature: float = 0.0
    max_tokens: int | None = None
    retries: int = 20
    thinking_level: ThinkingLevelStr = "off"
    include_thoughts: bool = False


def _messages_to_contents_and_system(
    msgs: list[dict[str, str]],
) -> tuple[str, str | None]:
    systems = [str(m["content"]) for m in msgs if m.get("role") == "system" and m.get("content")]
    users = [str(m["content"]) for m in msgs if m.get("role") == "user" and m.get("content")]
    contents = "\n\n".join(users) if users else ""
    system_instruction = "\n\n".join(systems) if systems else None
    return contents, system_instruction


def _usage_from_response(resp: Any) -> tuple[Usage, dict[str, Any]]:
    um = getattr(resp, "usage_metadata", None)
    raw_um: dict[str, Any] = {}
    if um is not None and hasattr(um, "model_dump"):
        raw_um = um.model_dump(mode="json")

    prompt = getattr(um, "prompt_token_count", None) if um is not None else None
    cand = getattr(um, "candidates_token_count", None) if um is not None else None
    thoughts = getattr(um, "thoughts_token_count", None) if um is not None else None

    in_t = int(prompt) if prompt is not None else None
    out_base = int(cand) if cand is not None else None
    out_th = int(thoughts) if thoughts is not None else None
    if out_base is not None or out_th is not None:
        out_tokens = (out_base or 0) + (out_th or 0)
    else:
        out_tokens = None

    return Usage(in_tokens=in_t, out_tokens=out_tokens), raw_um


def _parts_from_response(resp: Any) -> tuple[str, str, list[dict[str, Any]]]:
    """Returns (answer_text, thought_text_joined, part_debug_list)."""
    thought_chunks: list[str] = []
    text_chunks: list[str] = []
    part_debug: list[dict[str, Any]] = []

    cands = getattr(resp, "candidates", None) or []
    if not cands:
        return "", "", []
    c0 = cands[0]
    content = getattr(c0, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return "", "", []

    for p in parts:
        tx = getattr(p, "text", None) or ""
        is_thought = bool(getattr(p, "thought", False))
        sig = getattr(p, "thought_signature", None)
        entry: dict[str, Any] = {"thought": is_thought, "has_text": bool(tx)}
        if sig is not None:
            entry["thought_signature_len"] = len(str(sig))
        part_debug.append(entry)
        if is_thought:
            thought_chunks.append(str(tx))
        else:
            text_chunks.append(str(tx))

    return "\n".join(text_chunks).strip(), "\n".join(thought_chunks).strip(), part_debug


def _response_meta(resp: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "model_version": getattr(resp, "model_version", None),
        "response_id": getattr(resp, "response_id", None),
    }
    pf = getattr(resp, "prompt_feedback", None)
    if pf is not None and hasattr(pf, "model_dump"):
        meta["prompt_feedback"] = pf.model_dump(mode="json")
    return meta


class GoogleGenaiBatchPredictor:
    def __init__(
        self,
        *,
        params: GoogleGenaiBatchParams,
        client: genai.Client | None = None,
        name: str | None = None,
    ) -> None:
        self._params = params
        self._client = client or _make_google_genai_client(
            api_key=_google_genai_api_key(),
            http_retry_attempts=int(params.retries),
        )
        self._name = name or f"google_genai_batch:{params.model_id}"

    @property
    def name(self) -> str:
        return self._name

    def _build_generate_config(self, *, with_thinking: bool) -> GenerateContentConfig:
        kwargs: dict[str, Any] = {"temperature": float(self._params.temperature)}
        if self._params.max_tokens is not None:
            kwargs["max_output_tokens"] = int(self._params.max_tokens)
        if with_thinking:
            level = _THINKING_MAP.get(self._params.thinking_level)
            if level is not None:
                kwargs["thinking_config"] = ThinkingConfig(
                    thinking_level=level,
                    include_thoughts=bool(self._params.include_thoughts),
                )
        return GenerateContentConfig(**kwargs)

    async def _agenerate_one(
        self,
        *,
        contents: str,
        system_instruction: str | None,
        with_thinking: bool,
    ) -> Any:
        cfg_base = self._build_generate_config(with_thinking=with_thinking)
        cfg_dump = cfg_base.model_dump(exclude_none=True)
        if system_instruction:
            cfg_dump["system_instruction"] = system_instruction
        cfg = GenerateContentConfig.model_validate(cfg_dump)

        return await self._client.aio.models.generate_content(
            model=self._params.model_id,
            contents=contents,
            config=cfg,
        )

    async def _apredict_one_batch(self, items: list[BatchItem], *, allowed_labels: list[str]) -> dict[str, Prediction]:
        prompt = get_prompt(self._params.prompt_id)
        msgs = prompt.build_messages(allowed_labels=allowed_labels, items=items, few_shot=self._params.few_shot)
        if not model_supports_system_prompt(self._params.model_id):
            msgs = OpenAICompatChatBatchPredictor._merge_system_into_first_user(msgs)
        contents, system_instruction = _messages_to_contents_and_system(msgs)

        last_error: Exception | None = None
        # Try thinking first when enabled; on failure, same attempt retries without thinking.
        thinking_modes: list[bool] = [True, False] if self._params.thinking_level != "off" else [False]
        max_attempts = int(self._params.retries) + 1

        for attempt in range(1, max_attempts + 1):
            for with_thinking in thinking_modes:
                try:
                    t0 = time.perf_counter()
                    resp = await self._agenerate_one(
                        contents=contents,
                        system_instruction=system_instruction,
                        with_thinking=with_thinking,
                    )
                    elapsed_s = time.perf_counter() - t0
                    return _predictions_from_genai_response(
                        items,
                        resp=resp,
                        allowed_labels=allowed_labels,
                        attempt=attempt,
                        with_thinking=with_thinking,
                        elapsed_s=elapsed_s,
                    )
                except Exception as exc:
                    last_error = exc
                    if with_thinking and self._params.thinking_level != "off":
                        logger.warning(
                            "google.genai batch failed with thinking (model_id={}, attempt={}): {} — retrying without thinking",
                            self._params.model_id,
                            attempt,
                            repr(exc),
                        )
                        continue
                    logger.warning(
                        "google.genai batch failed (model_id={}, attempt={}/{}, batch_size={}): {}",
                        self._params.model_id,
                        attempt,
                        max_attempts,
                        len(items),
                        repr(exc),
                    )
                    break

        return _predictions_all_failed(items, error=last_error)

    async def apredict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
        if not texts:
            return []
        if not allowed_labels:
            raise ValueError("allowed_labels must be non-empty")

        bs = max(1, int(self._params.batch_size))
        mc = max(1, int(self._params.max_concurrency))
        items = [BatchItem(id=str(i), text=t) for i, t in enumerate(texts)]

        batch_list: list[tuple[int, list[BatchItem]]] = []
        for idx, start in enumerate(range(0, len(items), bs)):
            batch_list.append((idx, items[start : start + bs]))

        sem = asyncio.Semaphore(mc)

        async def run_batch(batch_idx: int, batch: list[BatchItem]) -> tuple[int, dict[str, Prediction]]:
            async with sem:
                preds_by_id = await self._apredict_one_batch(batch, allowed_labels=allowed_labels)
            return batch_idx, preds_by_id

        results = await asyncio.gather(*(run_batch(i, b) for i, b in batch_list))
        results.sort(key=lambda x: x[0])

        out: list[Prediction] = [Prediction(pred_label=None) for _ in texts]
        for _, preds_by_id in results:
            for bid, pred in preds_by_id.items():
                out[int(bid)] = pred
        return out

    def predict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.apredict(texts, allowed_labels=allowed_labels))
        raise RuntimeError(
            "GoogleGenaiBatchPredictor.predict() cannot be used inside a running event loop; use await apredict()"
        )
