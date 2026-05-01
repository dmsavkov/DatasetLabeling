# pyright: basic
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .builders.banking10 import build_banking10
from .builders.implicit_hate import build_implicit_hate
from .builders.pubmed_20k_rct import build_pubmed_20k_rct
from .builders.tweet_eval_irony import build_tweet_eval_irony


@dataclass(frozen=True, slots=True)
class DatasetBuilder:
    dataset_name: str
    build_fn: Callable[..., dict[str, Any]]


DATASET_BUILDERS: list[DatasetBuilder] = [
    DatasetBuilder(dataset_name="banking-10", build_fn=build_banking10),
    DatasetBuilder(dataset_name="tweet_eval_irony", build_fn=build_tweet_eval_irony),
    DatasetBuilder(dataset_name="implicit_hate", build_fn=build_implicit_hate),
    DatasetBuilder(dataset_name="pubmed_20k_rct", build_fn=build_pubmed_20k_rct),
]


def build_all_splits(
    *,
    seed: int = 42,
    test_tiers: tuple[int, ...] = (20, 200, 5000),
    train_seed_tiers: tuple[int, ...] = (10, 100),
    root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """
    v1 orchestrator: build processed tiers for all registered benchmark datasets.
    """

    results: dict[str, dict[str, Any]] = {}
    for b in DATASET_BUILDERS:
        results[b.dataset_name] = b.build_fn(
            seed=seed,
            test_tiers=test_tiers,
            train_seed_tiers=train_seed_tiers,
            root=root,
        )
    return results

