# pyright: basic
"""Self-debate predictor: Pass A then Pass B; label disagreement => confusing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

from src.models.interfaces import Prediction
from src.models.llm.genai_batch_engine import GenaiBatchEngine, GenaiEngineParams
from src.models.llm.google_genai_batch import ThinkingLevelStr
from src.prompts.baseline import BatchItem
from src.prompts.parsing import confusion_from_debate, parse_self_debate_batch
from src.prompts.prompt_eng import build_self_debate_pass_a_messages, build_self_debate_pass_b_messages


@dataclass(frozen=True, slots=True)
class SelfDebateBatchParams:
    model_id: str
    batch_size: int = 3
    max_concurrency: int = 5
    temperature_a: float = 0.0
    pass_b_temperature: float = 0.5
    max_tokens: int | None = None
    retries: int = 20
    thinking_level: ThinkingLevelStr = "off"
    include_thoughts: bool = False


class SelfDebateBatchPredictor:
    def __init__(
        self,
        *,
        params: SelfDebateBatchParams,
        few_shot: list[tuple[str, str]] | None = None,
        label_normalizer: Callable[[str], str] | None = None,
        name: str | None = None,
    ) -> None:
        self._params = params
        self._few_shot = few_shot
        self._label_normalizer = label_normalizer
        self._engine = GenaiBatchEngine(
            params=GenaiEngineParams(
                model_id=params.model_id,
                batch_size=params.batch_size,
                max_concurrency=params.max_concurrency,
                temperature=params.temperature_a,
                max_tokens=params.max_tokens,
                retries=params.retries,
                thinking_level=params.thinking_level,
                include_thoughts=params.include_thoughts,
            )
        )
        self._name = name or f"self_debate:{params.model_id}"

    @property
    def name(self) -> str:
        return self._name

    def _parse_pass_response(
        self, answer_text: str, allowed_labels: list[str], items: list[BatchItem]
    ) -> dict[str, Prediction]:
        parsed = parse_self_debate_batch(
            answer_text,
            allowed_labels=allowed_labels,
            label_normalizer=self._label_normalizer,
        )
        out: dict[str, Prediction] = {}
        for it in items:
            row = parsed.get(it.id, {})
            out[it.id] = Prediction(
                pred_label=row.get("label") if isinstance(row.get("label"), str) else None,
                confidence=float(row.get("confidence", 0.0)),
                reason=str(row.get("reasoning", "")),
                raw=row,
            )
        return out

    async def apredict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
        if not texts:
            return []
        items = [BatchItem(id=str(i), text=t) for i, t in enumerate(texts)]
        bs = max(1, int(self._params.batch_size))
        mc = max(1, int(self._params.max_concurrency))
        sem = asyncio.Semaphore(mc)
        out: list[Prediction | None] = [None] * len(texts)

        async def process_batch(batch: list[BatchItem]) -> None:
            async with sem:
                pass_a_preds = await self._engine.predict_items(
                    batch,
                    allowed_labels=allowed_labels,
                    build_messages=lambda al, it, fs: build_self_debate_pass_a_messages(
                        allowed_labels=al, items=it, few_shot=fs
                    ),
                    parse_response=self._parse_pass_response,
                    few_shot=self._few_shot,
                    temperature=float(self._params.temperature_a),
                )
                pass_a_by_id: dict[str, dict[str, object]] = {}
                for it in batch:
                    p = pass_a_preds[it.id]
                    pass_a_by_id[it.id] = {
                        "label": p.pred_label,
                        "confidence": p.confidence,
                        "reasoning": p.reason,
                    }

                pass_b_preds = await self._engine.predict_items(
                    batch,
                    allowed_labels=allowed_labels,
                    build_messages=lambda al, it, _fs: build_self_debate_pass_b_messages(
                        allowed_labels=al, items=it, pass_a_by_id=pass_a_by_id
                    ),
                    parse_response=self._parse_pass_response,
                    few_shot=None,
                    temperature=float(self._params.pass_b_temperature),
                )

                for it in batch:
                    idx = int(it.id)
                    a = pass_a_preds[it.id]
                    b = pass_b_preds[it.id]
                    a_lab = a.pred_label
                    b_lab = b.pred_label
                    confusing = confusion_from_debate(a_lab, b_lab)
                    out[idx] = Prediction(
                        pred_label=None if confusing else b_lab,
                        confidence=b.confidence,
                        reason=b.reason,
                        usage=b.usage or a.usage,
                        raw={
                            "is_confusing": confusing,
                            "a_label": a_lab,
                            "a_conf": a.confidence,
                            "a_reasoning": a.reason,
                            "b_label": b_lab,
                            "b_conf": b.confidence,
                            "b_reasoning": b.reason,
                        },
                    )

        batches = [items[i : i + bs] for i in range(0, len(items), bs)]
        await asyncio.gather(*(process_batch(b) for b in batches))
        return [p if p is not None else Prediction(pred_label=None) for p in out]

    def predict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.apredict(texts, allowed_labels=allowed_labels))
        raise RuntimeError("Use await apredict() inside async context")
