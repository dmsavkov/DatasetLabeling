# pyright: basic
"""Shared async batching for google.genai predictors."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
from loguru import logger

from src.models.clients.dispatch import _google_genai_api_key
from src.models.clients.registry import model_supports_system_prompt
from src.models.interfaces import Prediction, Usage, split_call_usage_across_rows
from src.models.llm.google_genai_batch import (
    ThinkingLevelStr,
    _THINKING_MAP,
    _make_google_genai_client,
    _messages_to_contents_and_system,
    _parts_from_response,
    _response_meta,
    _usage_from_response,
)
from src.models.llm.openai_compat_chat_batch import OpenAICompatChatBatchPredictor
from src.prompts.baseline import BatchItem

MessageBuilder = Callable[
    [list[str], list[BatchItem], list[tuple[str, str]] | None],
    list[dict[str, str]],
]
BatchParseFn = Callable[[str, list[str], list[BatchItem]], dict[str, Prediction]]


@dataclass(frozen=True, slots=True)
class GenaiEngineParams:
    model_id: str
    batch_size: int = 5
    max_concurrency: int = 5
    temperature: float = 0.0
    max_tokens: int | None = None
    retries: int = 20
    thinking_level: ThinkingLevelStr = "off"
    include_thoughts: bool = False


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


class GenaiBatchEngine:
    def __init__(self, *, params: GenaiEngineParams, client: genai.Client | None = None) -> None:
        self._params = params
        self._client = client or _make_google_genai_client(
            api_key=_google_genai_api_key(),
            http_retry_attempts=int(params.retries),
        )

    def _build_generate_config(self, *, temperature: float, with_thinking: bool) -> GenerateContentConfig:
        kwargs: dict[str, Any] = {"temperature": float(temperature)}
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

    async def _agenerate(self, *, msgs: list[dict[str, str]], temperature: float, with_thinking: bool) -> Any:
        if not model_supports_system_prompt(self._params.model_id):
            msgs = OpenAICompatChatBatchPredictor._merge_system_into_first_user(msgs)
        contents, system_instruction = _messages_to_contents_and_system(msgs)
        cfg_dump = self._build_generate_config(temperature=temperature, with_thinking=with_thinking).model_dump(
            exclude_none=True
        )
        if system_instruction:
            cfg_dump["system_instruction"] = system_instruction
        cfg = GenerateContentConfig.model_validate(cfg_dump)
        return await self._client.aio.models.generate_content(
            model=self._params.model_id,
            contents=contents,
            config=cfg,
        )

    async def _call_batch(
        self,
        items: list[BatchItem],
        *,
        allowed_labels: list[str],
        build_messages: MessageBuilder,
        parse_response: BatchParseFn,
        few_shot: list[tuple[str, str]] | None,
        temperature: float,
    ) -> dict[str, Prediction]:
        msgs = build_messages(allowed_labels, items, few_shot)
        last_error: Exception | None = None
        thinking_modes: list[bool] = [True, False] if self._params.thinking_level != "off" else [False]
        max_attempts = int(self._params.retries) + 1

        for attempt in range(1, max_attempts + 1):
            for with_thinking in thinking_modes:
                try:
                    t0 = time.perf_counter()
                    resp = await self._agenerate(msgs=msgs, temperature=temperature, with_thinking=with_thinking)
                    elapsed_s = time.perf_counter() - t0
                    answer_text, thought_text, part_debug = _parts_from_response(resp)
                    usage, raw_um = _usage_from_response(resp)
                    per_row_usage = split_call_usage_across_rows(usage, len(items))
                    preds_by_id = parse_response(answer_text, allowed_labels, items)
                    out: dict[str, Prediction] = {}
                    for idx, it in enumerate(items):
                        base = preds_by_id.get(it.id)
                        if base is None:
                            base = Prediction(pred_label=None, usage=per_row_usage[idx], raw={"error": "missing_id"})
                        raw = dict(base.raw) if isinstance(base.raw, dict) else {}
                        raw.update(
                            {
                                "attempt": attempt,
                                "with_thinking": with_thinking,
                                "elapsed_s": float(elapsed_s),
                                "thought_text_preview": thought_text[:2000] if thought_text else None,
                                "parts": part_debug,
                                "usage_metadata": raw_um,
                                "response_meta": _response_meta(resp),
                            }
                        )
                        out[it.id] = Prediction(
                            pred_label=base.pred_label,
                            confidence=base.confidence,
                            reason=base.reason or (thought_text if thought_text else None),
                            probs=base.probs,
                            usage=per_row_usage[idx],
                            raw=raw,
                        )
                    return out
                except Exception as exc:
                    last_error = exc
                    if with_thinking and self._params.thinking_level != "off":
                        logger.warning(
                            "genai batch failed with thinking (model_id={}, attempt={}): {} — retrying without",
                            self._params.model_id,
                            attempt,
                            repr(exc),
                        )
                        continue
                    logger.warning(
                        "genai batch failed (model_id={}, attempt={}/{}): {}",
                        self._params.model_id,
                        attempt,
                        max_attempts,
                        repr(exc),
                    )
                    break
        return _predictions_all_failed(items, error=last_error)

    async def predict_items(
        self,
        items: list[BatchItem],
        *,
        allowed_labels: list[str],
        build_messages: MessageBuilder,
        parse_response: BatchParseFn,
        few_shot: list[tuple[str, str]] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Prediction]:
        temp = float(self._params.temperature if temperature is None else temperature)
        return await self._call_batch(
            items,
            allowed_labels=allowed_labels,
            build_messages=build_messages,
            parse_response=parse_response,
            few_shot=few_shot,
            temperature=temp,
        )

    async def apredict_custom(
        self,
        texts: list[str],
        *,
        allowed_labels: list[str],
        build_messages: MessageBuilder,
        parse_response: BatchParseFn,
        few_shot: list[tuple[str, str]] | None = None,
        temperature: float | None = None,
    ) -> list[Prediction]:
        if not texts:
            return []
        if not allowed_labels:
            raise ValueError("allowed_labels must be non-empty")

        temp = float(self._params.temperature if temperature is None else temperature)
        bs = max(1, int(self._params.batch_size))
        mc = max(1, int(self._params.max_concurrency))
        items = [BatchItem(id=str(i), text=t) for i, t in enumerate(texts)]

        batch_list: list[tuple[int, list[BatchItem]]] = []
        for idx, start in enumerate(range(0, len(items), bs)):
            batch_list.append((idx, items[start : start + bs]))

        sem = asyncio.Semaphore(mc)

        async def run_batch(batch_idx: int, batch: list[BatchItem]) -> tuple[int, dict[str, Prediction]]:
            async with sem:
                preds = await self._call_batch(
                    batch,
                    allowed_labels=allowed_labels,
                    build_messages=build_messages,
                    parse_response=parse_response,
                    few_shot=few_shot,
                    temperature=temp,
                )
            return batch_idx, preds

        results = await asyncio.gather(*(run_batch(i, b) for i, b in batch_list))
        results.sort(key=lambda x: x[0])

        out: list[Prediction] = [Prediction(pred_label=None) for _ in texts]
        for _, preds_by_id in results:
            for bid, pred in preds_by_id.items():
                out[int(bid)] = pred
        return out
