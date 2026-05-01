# pyright: basic
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from src.models.interfaces import Prediction


@dataclass(frozen=True, slots=True)
class CommitteeMember:
    name: str
    predictor: Any  # expects `.apredict(texts, allowed_labels=...)`


def _majority_vote(labels: list[str | None]) -> str | None:
    counts: dict[str, int] = {}
    for lab in labels:
        if lab is None:
            continue
        counts[str(lab)] = counts.get(str(lab), 0) + 1
    if not counts:
        return None
    # deterministic: highest count, then label lexicographically
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return items[0][0]


class CommitteePredictor:
    """
    Async committee predictor.

    Returns `Prediction.pred_label` as the majority vote. Per-member labels are
    stored in `Prediction.raw["committee"]` so the harness can flatten them.
    """

    def __init__(self, members: list[CommitteeMember], *, name: str = "committee") -> None:
        if len(members) < 2:
            raise ValueError("CommitteePredictor requires at least 2 members")
        self._members = list(members)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def apredict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
        if not texts:
            return []
        if not allowed_labels:
            raise ValueError("allowed_labels must be non-empty")

        async def run_member(m: CommitteeMember) -> list[Prediction]:
            ap = getattr(m.predictor, "apredict", None)
            if ap is None:
                raise TypeError(f"Member predictor {m.name!r} has no apredict()")
            return await ap(texts, allowed_labels=allowed_labels)

        member_preds: list[list[Prediction]] = await asyncio.gather(*(run_member(m) for m in self._members))
        if any(len(mp) != len(texts) for mp in member_preds):
            raise ValueError("Committee member returned wrong number of predictions")

        out: list[Prediction] = []
        for i in range(len(texts)):
            per_member = [{"name": self._members[j].name, "pred_label": member_preds[j][i].pred_label} for j in range(len(self._members))]
            maj = _majority_vote([x["pred_label"] for x in per_member])
            out.append(
                Prediction(
                    pred_label=maj,
                    raw={"committee": {"members": per_member, "majority": maj}},
                )
            )
        return out

