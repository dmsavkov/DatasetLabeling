# pyright: basic
from __future__ import annotations

from src.dspy_gepa.batching import SentenceRow, build_label_balanced_batches


def test_batches_avoid_single_label_when_possible() -> None:
    rows = []
    for lab in ("methods", "results", "objective"):
        for i in range(6):
            rows.append(SentenceRow(sample_id=f"{lab}_{i}", text=f"t {lab} {i}", label_key=lab))
    batches = build_label_balanced_batches(rows, batch_size=5, seed=0)
    assert batches
    for b in batches:
        assert b.size == 5
        assert len({r.label_name for r in b.rows}) >= 2
