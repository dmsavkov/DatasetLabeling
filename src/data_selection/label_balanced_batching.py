# pyright: basic
"""Label-aware batch construction for LLM and DSPy runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.datasets.schema import SCHEMA


@dataclass(frozen=True, slots=True)
class SentenceRow:
    sample_id: str
    text: str
    label_key: str

    @property
    def label_name(self) -> str:
        """Alias for DSPy callers that used ``label_name``."""
        return self.label_key


@dataclass(frozen=True, slots=True)
class TextBatch:
    batch_id: int
    rows: tuple[SentenceRow, ...]

    @property
    def size(self) -> int:
        return len(self.rows)


def dataframe_to_sentence_rows(
    df: pd.DataFrame,
    *,
    label_key_fn: Callable[[object], str],
) -> list[SentenceRow]:
    out: list[SentenceRow] = []
    for _, row in df.iterrows():
        out.append(
            SentenceRow(
                sample_id=str(row[SCHEMA.sample_id]),
                text=str(row[SCHEMA.text]),
                label_key=label_key_fn(row[SCHEMA.true_label]),
            )
        )
    return out


def build_label_balanced_batches(
    rows: list[SentenceRow],
    *,
    batch_size: int,
    seed: int,
    drop_incomplete: bool = True,
) -> list[TextBatch]:
    """
    Build full batches using round-robin dequeues per label key.
    Batch order is shuffled after construction.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not rows:
        return []

    rng = np.random.default_rng(int(seed))
    by_label: dict[str, list[SentenceRow]] = {}
    for row in rows:
        by_label.setdefault(row.label_key, []).append(row)

    label_keys = list(by_label.keys())
    rng.shuffle(label_keys)
    for lab in label_keys:
        rng.shuffle(by_label[lab])

    queues: dict[str, list[SentenceRow]] = {lab: list(items) for lab, items in by_label.items()}
    batches: list[list[SentenceRow]] = []

    while any(queues.values()):
        batch: list[SentenceRow] = []
        while len(batch) < batch_size:
            progressed = False
            for lab in label_keys:
                if not queues[lab]:
                    continue
                batch.append(queues[lab].pop(0))
                progressed = True
                if len(batch) >= batch_size:
                    break
            if not progressed:
                break
        if len(batch) == batch_size:
            batches.append(batch)
        elif batch and not drop_incomplete:
            batches.append(batch)
        else:
            break

    order = np.arange(len(batches))
    rng.shuffle(order)
    return [TextBatch(batch_id=int(i), rows=tuple(batches[int(j)])) for i, j in enumerate(order)]


def batch_to_numbered_text(batch: TextBatch) -> str:
    return "\n".join(f"{i + 1}. {row.text}" for i, row in enumerate(batch.rows))


def batches_to_manifest(batches: list[TextBatch], *, split: str) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for b in batches:
        manifest.append(
            {
                "split": split,
                "batch_id": b.batch_id,
                "input_texts_numbered": batch_to_numbered_text(b),
                "target_labels": [r.label_key for r in b.rows],
                "sample_ids": [r.sample_id for r in b.rows],
                "sentences": [
                    {
                        "position": i + 1,
                        "sample_id": r.sample_id,
                        "text": r.text,
                        "gold": r.label_key,
                    }
                    for i, r in enumerate(b.rows)
                ],
            }
        )
    return manifest
