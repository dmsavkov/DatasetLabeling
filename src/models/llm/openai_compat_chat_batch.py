# pyright: basic
"""
OpenAI-compatible chat batch predictor.

Despite the historical "Google" naming in this repo, this predictor is provider-neutral:
it works with any OpenAI-compatible endpoint (Google GenAI OpenAI-compat, HF Inference
OpenAI-compat, etc.). Provider selection happens in `src.models.clients`.

Primary API is async (`apredict`). Sync `predict` only runs when no event loop is
active (uses `asyncio.run` once). Do not call `predict` from inside async code.

Concurrency: texts are chunked into batches of `batch_size`; up to `max_concurrency`
batches run in parallel (HTTP), controlled by `asyncio.Semaphore`. Results preserve
input row order.

Retries: failures are retried at **batch** granularity. After retries are exhausted,
every row in that batch gets `Prediction(pred_label=None, raw={...})`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from src.models.interfaces import Prediction, Usage, split_call_usage_across_rows
from src.models.clients.registry import model_supports_system_prompt
from src.prompts.baseline import BatchItem, DEFAULT_BATCH_SIZE, parse_batch_predictions
from src.prompts.registry import get_prompt


@dataclass(frozen=True, slots=True)
class OpenAICompatChatBatchParams:
    model_id: str
    prompt_id: str = "baseline_v1"
    few_shot: list[tuple[str, str]] | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    max_concurrency: int = 5
    temperature: float = 0.0
    max_tokens: int | None = None
    retries: int = 20


class OpenAICompatChatBatchPredictor:
    def __init__(self, client: AsyncOpenAI, *, params: OpenAICompatChatBatchParams, name: str | None = None) -> None:
        self._client = client
        self._params = params
        self._name = name or f"openai_compat_chat_batch:{params.model_id}"

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _merge_system_into_first_user(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
        systems = [m for m in msgs if m.get("role") == "system" and m.get("content")]
        if not systems:
            return msgs
        users = [m for m in msgs if m.get("role") == "user" and m.get("content")]
        if not users:
            return msgs

        system_text = "\n\n".join([str(m["content"]) for m in systems])
        # Mutate local copy only: avoid side-effects on the caller's list.
        out = [dict(m) for m in msgs]
        first_user_idx = next(i for i, m in enumerate(out) if m.get("role") == "user" and m.get("content"))
        out[first_user_idx]["content"] = system_text + "\n\n" + str(out[first_user_idx]["content"])
        out = [m for m in out if m.get("role") != "system"]
        return out

    async def _apredict_one_batch(self, items: list[BatchItem], *, allowed_labels: list[str]) -> dict[str, Prediction]:
        prompt = get_prompt(self._params.prompt_id)
        msgs = prompt.build_messages(allowed_labels=allowed_labels, items=items, few_shot=self._params.few_shot)
        if not model_supports_system_prompt(self._params.model_id):
            msgs = self._merge_system_into_first_user(msgs)

        last_error: Exception | None = None
        for attempt in range(1, int(self._params.retries) + 2):
            try:
                start = time.perf_counter()
                kwargs: dict[str, Any] = {
                    "model": self._params.model_id,
                    "messages": msgs,
                    "temperature": float(self._params.temperature),
                }
                if self._params.max_tokens is not None:
                    kwargs["max_tokens"] = int(self._params.max_tokens)
                resp = await self._client.chat.completions.create(**kwargs)
                _ = time.perf_counter() - start

                content = ""
                if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                    content = str(resp.choices[0].message.content)

                mapping = parse_batch_predictions(content, allowed_labels=allowed_labels)

                usage = getattr(resp, "usage", None)
                in_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
                out_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
                call_usage = Usage(in_tokens=in_tokens, out_tokens=out_tokens)
                per_row_usage = split_call_usage_across_rows(call_usage, len(items))

                preds: dict[str, Prediction] = {}
                for idx, it in enumerate(items):
                    label = mapping.get(it.id)
                    preds[it.id] = Prediction(
                        pred_label=label,
                        confidence=None,
                        reason=None,
                        probs=None,
                        usage=per_row_usage[idx],
                        raw={"attempt": attempt, "content": content[:2000]},
                    )
                return preds
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM batch failed (model_id={}, attempt={}/{}, batch_size={}): {}",
                    self._params.model_id,
                    attempt,
                    int(self._params.retries) + 1,
                    len(items),
                    repr(exc),
                )
                if attempt >= int(self._params.retries) + 1:
                    break

        preds = {}
        for it in items:
            preds[it.id] = Prediction(
                pred_label=None,
                confidence=None,
                reason=None,
                probs=None,
                usage=Usage(in_tokens=None, out_tokens=None),
                raw={"error": str(last_error) if last_error else "unknown"},
            )
        return preds

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
            "OpenAICompatChatBatchPredictor.predict() cannot be used inside a running event loop; use await apredict()"
        )

