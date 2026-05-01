from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import SCHEMA, ensure_unique_sample_ids, validate_processed_samples_df


@dataclass(frozen=True, slots=True)
class ProcessedPaths:
    root_dir: Path

    def dataset_dir(self, dataset_name: str) -> Path:
        return self.root_dir / dataset_name

    def split_dir(self, dataset_name: str, split_name: str) -> Path:
        return self.dataset_dir(dataset_name) / split_name

    def tier_dir(self, dataset_name: str, split_name: str, tier_size: int) -> Path:
        return self.split_dir(dataset_name, split_name) / f"tier_{tier_size}"

    def artifact_dir(self, dataset_name: str, split_name: str, artifact: str) -> Path:
        """
        Generic artifact directory under a split namespace.

        Examples:
        - split_name="test", artifact="tier_200"
        - split_name="train_seed", artifact="tier_10"
        - split_name="train_pool_remaining", artifact="full"
        """

        return self.split_dir(dataset_name, split_name) / artifact


def processed_root(root: Path | None = None) -> Path:
    """
    Returns the canonical `data/processed` directory.

    Accepts either:
    - `root=None` or a repo root (we append `data/processed`)
    - a path that already points to `.../data/processed` (we return it as-is)
    """

    base = Path.cwd() if root is None else Path(root)
    # If the caller already passed `.../data/processed`, don't append again.
    if base.name == "processed" and base.parent.name == "data":
        return base
    return base / "data" / "processed"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_of_ids(sample_ids: list[str]) -> str:
    payload = "\n".join(sample_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_processed_tier(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    split_name: str,
    tier_size: int,
    seed: int,
    builder: str,
    origin: dict[str, Any],
    extra_manifest: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Path:
    """
    Persist a processed tier as Parquet + a `manifest.json`.
    Returns the tier directory path.
    """

    validate_processed_samples_df(df)
    ensure_unique_sample_ids(df, id_col=SCHEMA.sample_id)

    out_root = processed_root(root)
    paths = ProcessedPaths(out_root)
    tier_dir = paths.tier_dir(dataset_name, split_name, int(tier_size))
    tier_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = tier_dir / "samples.parquet"
    df.to_parquet(parquet_path, index=False)

    sample_ids = df[SCHEMA.sample_id].astype(str).tolist()
    manifest: dict[str, Any] = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "tier_size_requested": int(tier_size),
        "tier_size_actual": int(len(df)),
        "seed": int(seed),
        "builder": builder,
        "origin": origin,
        "schema": {
            "required_columns": [SCHEMA.sample_id, SCHEMA.dataset_name, SCHEMA.text, SCHEMA.true_label],
            "optional_columns": [SCHEMA.meta_json],
        },
        "sample_id_sha256": _sha256_of_ids(sample_ids),
        "created_at": _utc_iso(),
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    (tier_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return tier_dir


def save_processed_artifact(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    split_name: str,
    artifact: str,
    seed: int,
    builder: str,
    origin: dict[str, Any],
    extra_manifest: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Path:
    """
    Persist a processed artifact as Parquet + a `manifest.json`.
    Use this for non-tier artifacts like `full`.
    """

    validate_processed_samples_df(df)
    ensure_unique_sample_ids(df, id_col=SCHEMA.sample_id)

    out_root = processed_root(root)
    paths = ProcessedPaths(out_root)
    out_dir = paths.artifact_dir(dataset_name, split_name, artifact)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "samples.parquet"
    df.to_parquet(parquet_path, index=False)

    sample_ids = df[SCHEMA.sample_id].astype(str).tolist()
    manifest: dict[str, Any] = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "artifact": artifact,
        "rows": int(len(df)),
        "seed": int(seed),
        "builder": builder,
        "origin": origin,
        "schema": {
            "required_columns": [SCHEMA.sample_id, SCHEMA.dataset_name, SCHEMA.text, SCHEMA.true_label],
            "optional_columns": [SCHEMA.meta_json],
        },
        "sample_id_sha256": _sha256_of_ids(sample_ids),
        "created_at": _utc_iso(),
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return out_dir


def load_processed_tier(
    *,
    dataset_name: str,
    split_name: str,
    tier_size: int,
    root: Path | None = None,
) -> pd.DataFrame:
    out_root = processed_root(root)
    paths = ProcessedPaths(out_root)
    tier_dir = paths.tier_dir(dataset_name, split_name, int(tier_size))
    parquet_path = tier_dir / "samples.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Processed parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    validate_processed_samples_df(df)
    return df


def load_processed_artifact(
    *,
    dataset_name: str,
    split_name: str,
    artifact: str,
    root: Path | None = None,
) -> pd.DataFrame:
    out_root = processed_root(root)
    paths = ProcessedPaths(out_root)
    out_dir = paths.artifact_dir(dataset_name, split_name, artifact)
    parquet_path = out_dir / "samples.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Processed parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    validate_processed_samples_df(df)
    return df


def load_manifest(
    *,
    dataset_name: str,
    split_name: str,
    tier_size: int,
    root: Path | None = None,
) -> dict[str, Any]:
    out_root = processed_root(root)
    paths = ProcessedPaths(out_root)
    tier_dir = paths.tier_dir(dataset_name, split_name, int(tier_size))
    manifest_path = tier_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_artifact_manifest(
    *,
    dataset_name: str,
    split_name: str,
    artifact: str,
    root: Path | None = None,
) -> dict[str, Any]:
    out_root = processed_root(root)
    paths = ProcessedPaths(out_root)
    out_dir = paths.artifact_dir(dataset_name, split_name, artifact)
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))
