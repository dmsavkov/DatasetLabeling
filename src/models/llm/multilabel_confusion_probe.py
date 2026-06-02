# pyright: basic
"""Multi-label probe: multiple predicted labels => confusion on single-label gold."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

from src.models.interfaces import Prediction
from src.models.llm.genai_batch_engine import GenaiBatchEngine, GenaiEngineParams
from src.models.llm.google_genai_batch import ThinkingLevelStr
from src.prompts.baseline import BatchItem
from src.prompts.parsing import confusion_from_label_lists, parse_multilabel_batch
from src.prompts.prompt_eng import build_multilabel_confusion_messages


@dataclass(frozen=True, slots=True)
class MultilabelConfusionProbeParams:
    model_id: str
    batch_size: int = 3
    max_concurrency: int = 7
    temperature: float = 0.0
    max_tokens: int | None = None
    retries: int = 20
    thinking_level: ThinkingLevelStr = "off"
    include_thoughts: bool = False


class MultilabelConfusionProbePredictor:
    def __init__(
        self,
        *,
        params: MultilabelConfusionProbeParams,
        few_shot: list[tuple[str, str]] | None = None,
        label_normalizer: Callable[[str], str] | None = None,
        name: str | None = None,
    ) -> None:
        self._few_shot = few_shot
        self._label_normalizer = label_normalizer
        self._engine = GenaiBatchEngine(
            params=GenaiEngineParams(
                model_id=params.model_id,
                batch_size=params.batch_size,
                max_concurrency=params.max_concurrency,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                retries=params.retries,
                thinking_level=params.thinking_level,
                include_thoughts=params.include_thoughts,
            )
        )
        self._name = name or f"multilabel_confusion_probe:{params.model_id}"

    @property
    def name(self) -> str:
        return self._name

    def _parse_response(self, answer_text: str, allowed_labels: list[str], items: list[BatchItem]) -> dict[str, Prediction]:
        mapping = parse_multilabel_batch(
            answer_text,
            allowed_labels=allowed_labels,
            label_normalizer=self._label_normalizer,
        )
        out: dict[str, Prediction] = {}
        for it in items:
            labels = mapping.get(it.id, [])
            is_confusing, pred_label = confusion_from_label_lists(labels)
            out[it.id] = Prediction(
                pred_label=pred_label,
                confidence=None,
                reason=None,
                probs=None,
                usage=None,
                raw={
                    "pred_labels": labels,
                    "is_confusing": is_confusing,
                    "n_pred_labels": len(labels),
                },
            )
        return out

    def _build_messages(self, allowed_labels: list[str], items: list[BatchItem], few_shot: list[tuple[str, str]] | None) -> list[dict[str, str]]:
        return build_multilabel_confusion_messages(
            allowed_labels=allowed_labels,
            items=items,
            few_shot=few_shot,
        )

    async def apredict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
        return await self._engine.apredict_custom(
            texts,
            allowed_labels=allowed_labels,
            build_messages=lambda al, it, fs: self._build_messages(al, it, fs),
            parse_response=self._parse_response,
            few_shot=self._few_shot,
        )

    def predict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.apredict(texts, allowed_labels=allowed_labels))
        raise RuntimeError("Use await apredict() inside async context")
