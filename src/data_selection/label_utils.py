# pyright: basic
"""Dataset card context, per-dataset label canonicalization, and stratification quotas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.datasets.cards import default_card_path, read_card_json
from src.datasets.io import processed_root

PUBMED_LABEL_NAMES: tuple[str, ...] = (
    "background",
    "objective",
    "methods",
    "results",
    "conclusions",
)

PUBMED_ID_TO_NAME: dict[str, str] = {str(i): name for i, name in enumerate(PUBMED_LABEL_NAMES)}
PUBMED_NAME_TO_ID: dict[str, str] = {name: sid for sid, name in PUBMED_ID_TO_NAME.items()}

DEFAULT_PROMPT_BY_DATASET: dict[str, str] = {
    "pubmed_20k_rct": "baseline_v1",
    "banking-10": "baseline_v1",
    "tweet_eval_irony": "baseline_v1",
    "implicit_hate": "baseline_v1",
}

Canonicalizer = Callable[[str], str]


def normalize_pubmed_label(value: object) -> str:
    # Do not use ``value or ""`` — label id 0 (background) is falsy in Python.
    raw = "" if value is None else str(value).strip().lower()
    if raw in PUBMED_NAME_TO_ID:
        return raw
    if raw in PUBMED_ID_TO_NAME:
        return PUBMED_ID_TO_NAME[raw]
    aliases = {
        "conclusion": "conclusions",
        "concl": "conclusions",
        "method": "methods",
        "result": "results",
        "objective(s)": "objective",
    }
    return aliases.get(raw, raw)


def _passthrough_canonicalize(value: object) -> str:
    return "" if value is None else str(value).strip().lower()


def canonicalizer_for_dataset(dataset_name: str) -> Canonicalizer:
    if dataset_name == "pubmed_20k_rct":
        return normalize_pubmed_label
    return _passthrough_canonicalize


def prompt_labels_for_dataset(*, dataset_name: str, label_ids: list[str]) -> list[str]:
    """Strings shown in LLM allowed_labels."""
    canon = canonicalizer_for_dataset(dataset_name)
    out: list[str] = []
    seen: set[str] = set()
    for lid in label_ids:
        name = canon(lid)
        if dataset_name == "pubmed_20k_rct" and name not in PUBMED_LABEL_NAMES:
            continue
        if name and name not in seen:
            out.append(name)
            seen.add(name)
    if out:
        return out
    if dataset_name == "pubmed_20k_rct":
        return list(PUBMED_LABEL_NAMES)
    return [canon(lid) for lid in label_ids]


def default_prompt_id(dataset_name: str) -> str:
    return DEFAULT_PROMPT_BY_DATASET.get(dataset_name, "baseline_v1")


def per_label_quotas(n_total: int, label_ids: list[str]) -> dict[str, int]:
    if n_total <= 0 or not label_ids:
        return {}
    per_label = n_total // len(label_ids)
    remainder = n_total % len(label_ids)
    return {lab: per_label + (1 if i < remainder else 0) for i, lab in enumerate(label_ids)}


@dataclass(frozen=True, slots=True)
class DatasetContext:
    dataset_name: str
    label_ids: list[str]
    prompt_labels: list[str]
    prompt_id: str
    processed_root_path: Path | None = None

    @property
    def n_labels(self) -> int:
        return len(self.label_ids)

    def canonicalize(self, value: object) -> str:
        return canonicalizer_for_dataset(self.dataset_name)(value)

    def prompt_label_for_id(self, label_id: str) -> str:
        return self.canonicalize(label_id)

    def labels_match(self, gold_label_id: object, pred_raw: object) -> bool:
        if pred_raw is None:
            return False
        return self.canonicalize(gold_label_id) == self.canonicalize(pred_raw)

    def validate_prediction_params(
        self,
        *,
        prediction_size: int,
        batch_size: int,
        n_centroids_per_label: int,
    ) -> tuple[int, int]:
        """Returns (n_centroids, samples_per_centroid). Raises ValueError on bad params."""
        if self.n_labels < 1:
            raise ValueError("dataset has no labels")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if prediction_size % batch_size != 0:
            raise ValueError(
                f"prediction_size ({prediction_size}) must be divisible by batch_size ({batch_size})"
            )
        n_centroids = int(n_centroids_per_label) * self.n_labels
        if prediction_size % n_centroids != 0:
            raise ValueError(
                f"prediction_size ({prediction_size}) must be divisible by "
                f"n_centroids ({n_centroids} = {n_centroids_per_label} per label × {self.n_labels} labels)"
            )
        return n_centroids, prediction_size // n_centroids


def load_dataset_context(
    dataset_name: str,
    *,
    prompt_id: str | None = None,
    processed_root_path: Path | None = None,
) -> DatasetContext:
    pr = processed_root(processed_root_path)
    card_path = default_card_path(processed_root_dir=pr, dataset_name=dataset_name)
    if not card_path.exists():
        raise FileNotFoundError(f"Missing dataset card: {card_path}")
    card = read_card_json(card_path)
    label_ids = [str(x) for x in card.labels]
    pid = prompt_id if prompt_id is not None else default_prompt_id(dataset_name)
    return DatasetContext(
        dataset_name=dataset_name,
        label_ids=label_ids,
        prompt_labels=prompt_labels_for_dataset(dataset_name=dataset_name, label_ids=label_ids),
        prompt_id=pid,
        processed_root_path=processed_root_path,
    )
