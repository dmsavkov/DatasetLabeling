from __future__ import annotations

import asyncio

import pytest

from src.models.ensemble.committee import CommitteeMember, CommitteePredictor
from src.models.interfaces import Prediction


def test_committee_majority_vote_and_raw_shape() -> None:
    class M:
        def __init__(self, label: str | None) -> None:
            self._label = label

        async def apredict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
            return [Prediction(pred_label=self._label) for _ in texts]

    committee = CommitteePredictor(
        [
            CommitteeMember(name="a", predictor=M("x")),
            CommitteeMember(name="b", predictor=M("x")),
            CommitteeMember(name="c", predictor=M("y")),
        ],
        name="cmt",
    )

    async def go() -> list[Prediction]:
        return await committee.apredict(["t1", "t2"], allowed_labels=["x", "y"])

    out = asyncio.run(go())
    assert [p.pred_label for p in out] == ["x", "x"]
    assert isinstance(out[0].raw, dict)
    c = out[0].raw["committee"]
    assert c["majority"] == "x"
    assert len(c["members"]) == 3


def test_committee_requires_multiple_members() -> None:
    with pytest.raises(ValueError):
        _ = CommitteePredictor([CommitteeMember(name="a", predictor=object())])

