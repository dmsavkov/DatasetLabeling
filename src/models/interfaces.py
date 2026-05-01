from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class Usage:
    in_tokens: int | None = None
    out_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Prediction:
    pred_label: str | None
    confidence: float | None = None
    reason: str | None = None
    probs: Mapping[str, float] | None = None
    usage: Usage | None = None
    raw: Any | None = None


class Predictor(Protocol):
    """
    Minimal predictor contract for v1 evaluation.
    Implementations can be classical ML models or LLM-backed judges.
    """

    @property
    def name(self) -> str: ...

    def predict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]: ...
