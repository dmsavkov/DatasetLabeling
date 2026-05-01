from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import SCHEMA, validate_processed_samples_df


@dataclass(frozen=True, slots=True)
class DatasetCard:
    dataset_name: str
    description: str
    origin: dict[str, Any]
    labels: list[str]
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "description": self.description,
            "origin": self.origin,
            "labels": self.labels,
            "sample_count": self.sample_count,
        }


def make_card_from_df(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    description: str,
    origin: dict[str, Any],
) -> DatasetCard:
    validate_processed_samples_df(df)
    labels = sorted(df[SCHEMA.true_label].astype(str).unique().tolist())
    return DatasetCard(
        dataset_name=dataset_name,
        description=description,
        origin=origin,
        labels=labels,
        sample_count=int(len(df)),
    )


def write_card_json(path: Path, card: DatasetCard) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")


def read_card_json(path: Path) -> DatasetCard:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DatasetCard(
        dataset_name=str(payload["dataset_name"]),
        description=str(payload.get("description", "")),
        origin=dict(payload.get("origin", {})),
        labels=[str(x) for x in payload.get("labels", [])],
        sample_count=int(payload.get("sample_count", 0)),
    )


def default_card_path(*, processed_root_dir: Path, dataset_name: str) -> Path:
    return Path(processed_root_dir) / dataset_name / "dataset_card.json"

