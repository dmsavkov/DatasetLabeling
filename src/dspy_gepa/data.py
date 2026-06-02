# pyright: basic
"""Load GEPA optimizer parquet pools produced by ``build_gepa_optimizer_sets``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.datasets.cards import default_card_path, read_card_json
from src.datasets.io import processed_root
from src.datasets.schema import SCHEMA, validate_processed_samples_df


@dataclass(frozen=True, slots=True)
class GepaOptimizerSets:
    root_dir: Path
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    manifest: dict[str, object]


def resolve_gepa_sets_dir(*, dataset_name: str, explicit: Path | None, repo_root: Path) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(f"GEPA sets directory not found: {path}")
        return path
    base = repo_root / "data" / "gepa_optimizer_sets" / dataset_name
    if not base.is_dir():
        raise FileNotFoundError(
            f"No GEPA sets under {base}. Run scripts/build_gepa_optimizer_sets.py first."
        )
    children = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not children:
        raise FileNotFoundError(f"No timestamped runs under {base}")
    return children[-1]


def load_gepa_optimizer_sets(path: Path) -> GepaOptimizerSets:
    root = Path(path)
    train_p = root / "gepa_train.parquet"
    val_p = root / "gepa_val.parquet"
    manifest_p = root / "manifest.json"
    for p in (train_p, val_p):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}")
    train_df = pd.read_parquet(train_p)
    val_df = pd.read_parquet(val_p)
    validate_processed_samples_df(train_df)
    validate_processed_samples_df(val_df)
    manifest: dict[str, object] = {}
    if manifest_p.is_file():
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    return GepaOptimizerSets(root_dir=root, train_df=train_df, val_df=val_df, manifest=manifest)


def card_label_ids(dataset_name: str, *, repo_root: Path) -> list[str]:
    card_path = default_card_path(processed_root_dir=processed_root(repo_root), dataset_name=dataset_name)
    if not card_path.is_file():
        raise FileNotFoundError(card_path)
    return [str(x) for x in read_card_json(card_path).labels]
