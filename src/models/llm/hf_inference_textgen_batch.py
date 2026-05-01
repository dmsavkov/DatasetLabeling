# pyright: basic
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from huggingface_hub import AsyncInferenceClient
from loguru import logger

from src.models.interfaces import Prediction, Usage
from src.prompts.baseline import BatchItem, DEFAULT_BATCH_SIZE, parse_batch_predictions
from src.prompts.registry import get_prompt
from src.utils.retry import BackoffPolicy, async_retry


@dataclass(frozen=True, slots=True)
class HFInferenceTextGenBatchParams:
    model_id: str
    prompt_id: str = "baseline_v1"
    few_shot: list[tuple[str, str]] | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    max_concurrency: int = 5
    temperature: float = 0.0
    max_new_tokens: int | None = None
    retries: int = 20


class HFInferenceTextGenBatchPredictor:
    """
    HF Inference batch predictor.

    Primary path is `AsyncInferenceClient.chat.completions.create` ("conversational"),
    matching `raw-experiments/hf_llms_comparison.ipynb`.

    Fallback path is `text_generation` for providers/models exposing only that task.
    """

    def __init__(self, client: AsyncInferenceClient, *, params: HFInferenceTextGenBatchParams, name: str | None = None) -> None:
        self._client = client
        self._params = params
        self._name = name or f"hf_inference_textgen:{params.model_id}"

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _messages_to_prompt(msgs: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for m in msgs:
            role = str(m.get("role", "")).strip()
            content = str(m.get("content", "")).strip()
            if not content:
                continue
            if role:
                parts.append(f"{role.upper()}:\n{content}")
            else:
                parts.append(content)
        return "\n\n".join(parts).strip()

    async def _apredict_one_batch(self, items: list[BatchItem], *, allowed_labels: list[str]) -> dict[str, Prediction]:
        prompt = get_prompt(self._params.prompt_id)
        msgs = prompt.build_messages(allowed_labels=allowed_labels, items=items, few_shot=self._params.few_shot)
        text_prompt = self._messages_to_prompt(msgs)

        backoff = BackoffPolicy(base_delay_s=0.8, multiplier=1.7, max_delay_s=30.0, jitter_s=0.5)

        @async_retry(retries=int(self._params.retries), backoff=backoff, log_prefix=f"hf:{self._params.model_id}")
        async def _call_once() -> str:
            # 1) Prefer conversational chat completion (matches notebook behavior).
            chat_kwargs: dict[str, Any] = {
                "model": self._params.model_id,
                "messages": msgs,
                "temperature": float(self._params.temperature),
            }
            if self._params.max_new_tokens is not None:
                chat_kwargs["max_tokens"] = int(self._params.max_new_tokens)
            try:
                resp = await self._client.chat.completions.create(**chat_kwargs)  # type: ignore[attr-defined]
                if getattr(resp, "choices", None):
                    return str(resp.choices[0].message.content or "")
            except Exception:
                # 2) Fallback to text-generation task.
                gen_kwargs: dict[str, Any] = {
                    "temperature": float(self._params.temperature),
                    "do_sample": bool(float(self._params.temperature) > 0.0),
                    "return_full_text": False,
                }
                if self._params.max_new_tokens is not None:
                    gen_kwargs["max_new_tokens"] = int(self._params.max_new_tokens)
                return str(await self._client.text_generation(text_prompt, **gen_kwargs))
            return ""

        try:
            start = time.perf_counter()
            content = await _call_once()
            _ = time.perf_counter() - start
            mapping = parse_batch_predictions(str(content), allowed_labels=allowed_labels)
            preds: dict[str, Prediction] = {}
            for it in items:
                label = mapping.get(it.id)
                preds[it.id] = Prediction(
                    pred_label=label,
                    confidence=None,
                    reason=None,
                    probs=None,
                    usage=Usage(in_tokens=None, out_tokens=None),
                    raw={"content": str(content)[:2000]},
                )
            return preds
        except Exception as exc:
            logger.warning("HF batch permanently failed (model_id={}, batch_size={}): {}", self._params.model_id, len(items), repr(exc))
            return {
                it.id: Prediction(
                    pred_label=None,
                    confidence=None,
                    reason=None,
                    probs=None,
                    usage=Usage(None, None),
                    raw={"error": str(exc)},
                )
                for it in items
            }

    async def apredict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
        if not texts:
            return []
        if not allowed_labels:
            raise ValueError("allowed_labels must be non-empty")

        bs = int(self._params.batch_size)
        if bs <= 0:
            raise ValueError("batch_size must be positive")

        batches: list[list[BatchItem]] = []
        for start in range(0, len(texts), bs):
            chunk = texts[start : start + bs]
            items = [BatchItem(id=str(start + i), text=str(t)) for i, t in enumerate(chunk)]
            batches.append(items)

        sem = asyncio.Semaphore(int(self._params.max_concurrency))
        out: list[Prediction | None] = [None] * len(texts)

        async def run_batch(batch_items: list[BatchItem]) -> None:
            async with sem:
                preds = await self._apredict_one_batch(batch_items, allowed_labels=allowed_labels)
            for it in batch_items:
                idx = int(it.id)
                out[idx] = preds[it.id]

        await asyncio.gather(*[run_batch(b) for b in batches])
        return [p if p is not None else Prediction(None, None, None, None, Usage(None, None), {"error": "missing"}) for p in out]

