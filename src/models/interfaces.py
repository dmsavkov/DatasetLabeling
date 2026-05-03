from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class Usage:
    in_tokens: int | None = None
    out_tokens: int | None = None


def split_call_usage_across_rows(usage: Usage, n: int) -> list[Usage]:
    """
    One batched API call returns token counts for the whole request/response.
    Eval summaries sum per-row ``in_tokens`` / ``out_tokens``; without splitting,
    stamping the call totals on every row multiplies totals by ``n``.

    Returns ``n`` usages whose integer sums match the call (remainder goes to the
    first rows). This is a reporting convention, not a tokenizer-accurate split.
    """
    if n <= 0:
        return []

    def split_field(v: int | None) -> list[int | None]:
        if v is None:
            return [None] * n
        iv = int(v)
        base, rem = divmod(iv, n)
        return [base + (1 if i < rem else 0) for i in range(n)]

    ins = split_field(usage.in_tokens)
    outs = split_field(usage.out_tokens)
    return [Usage(in_tokens=ins[i], out_tokens=outs[i]) for i in range(n)]


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
