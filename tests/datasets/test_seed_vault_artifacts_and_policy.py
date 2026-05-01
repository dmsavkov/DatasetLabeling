from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.datasets.io import (
    load_artifact_manifest,
    load_manifest,
    load_processed_artifact,
    load_processed_tier,
    processed_root,
    save_processed_artifact,
    save_processed_tier,
)
from src.datasets.schema import SCHEMA
from src.datasets.seed_vault import build_seed_vault


def _make_processed_df(*, dataset_name: str, split_prefix: str, n: int) -> pd.DataFrame:
    labels = ["a", "b", "c", "d"]
    return pd.DataFrame(
        {
            SCHEMA.sample_id: [f"{split_prefix}_{i}" for i in range(n)],
            SCHEMA.dataset_name: [dataset_name] * n,
            SCHEMA.text: [f"t{i}" for i in range(n)],
            SCHEMA.true_label: [labels[i % len(labels)] for i in range(n)],
            SCHEMA.meta_json: ["{}"] * n,
        }
    )


def _ids(df: pd.DataFrame) -> set[str]:
    return set(df[SCHEMA.sample_id].astype(str).tolist())


def test_seed_vault_build_is_disjoint_and_policy_works(tmp_path: Path) -> None:
    train = _make_processed_df(dataset_name="toy", split_prefix="train", n=250)
    test = _make_processed_df(dataset_name="toy", split_prefix="test", n=120)

    sv = build_seed_vault(
        train_source=train,
        test_source=test,
        seed=42,
        train_seed_tiers=(10, 100),
        test_tiers=(20,),
        on_oversize_train_seed="error",
        on_oversize_test_5000="clip",
    )

    assert _ids(sv.train_seed_tiers[10]).isdisjoint(_ids(sv.train_seed_tiers[100]))
    assert _ids(sv.train_seed_tiers[10]).isdisjoint(_ids(sv.train_pool_remaining))
    assert _ids(sv.train_seed_tiers[100]).isdisjoint(_ids(sv.train_pool_remaining))

    assert len(sv.test_tiers[20]) == 20


def test_oversize_policy_clip_vs_error_unit(tmp_path: Path) -> None:
    train = _make_processed_df(dataset_name="toy", split_prefix="train", n=50)
    test = _make_processed_df(dataset_name="toy", split_prefix="test", n=100)

    sv = build_seed_vault(
        train_source=train,
        test_source=test,
        seed=1,
        train_seed_tiers=(10, 20),
        test_tiers=(5000,),
        on_oversize_train_seed="error",
        on_oversize_test_5000="clip",
    )
    assert sv.statuses["test_5000"] == "clipped"
    assert len(sv.test_tiers[5000]) == 100

    with pytest.raises(ValueError):
        _ = build_seed_vault(
            train_source=train,
            test_source=test,
            seed=2,
            train_seed_tiers=(10, 20),
            test_tiers=(5000,),
            on_oversize_train_seed="error",
            on_oversize_test_5000="error",
        )


def test_seed_vault_artifact_io_roundtrip(tmp_path: Path) -> None:
    train = _make_processed_df(dataset_name="toy", split_prefix="train", n=200)
    test = _make_processed_df(dataset_name="toy", split_prefix="test", n=120)

    sv = build_seed_vault(
        train_source=train,
        test_source=test,
        seed=42,
        train_seed_tiers=(10, 100),
        test_tiers=(20,),
        on_oversize_train_seed="error",
        on_oversize_test_5000="clip",
    )

    pr = processed_root(tmp_path)
    _ = save_processed_tier(
        sv.train_seed_tiers[10],
        dataset_name="toy",
        split_name="train_seed",
        tier_size=10,
        seed=42,
        builder="test",
        origin={"x": 1},
        extra_manifest={"seed_vault": {"kind": "train_seed", "tier": 10, "status": sv.statuses["train_seed_10"]}},
        root=tmp_path,
    )
    _ = save_processed_artifact(
        sv.train_pool_remaining,
        dataset_name="toy",
        split_name="train_pool_remaining",
        artifact="full",
        seed=42,
        builder="test",
        origin={"x": 1},
        extra_manifest={"seed_vault": {"kind": "train_pool_remaining"}},
        root=tmp_path,
    )

    loaded_seed = load_processed_tier(dataset_name="toy", split_name="train_seed", tier_size=10, root=pr)
    loaded_rem = load_processed_artifact(dataset_name="toy", split_name="train_pool_remaining", artifact="full", root=pr)
    assert len(loaded_seed) == 10
    assert len(loaded_rem) == len(sv.train_pool_remaining)

    man_seed = load_manifest(dataset_name="toy", split_name="train_seed", tier_size=10, root=pr)
    assert (man_seed.get("seed_vault") or {}).get("status") == sv.statuses["train_seed_10"]
    man_rem = load_artifact_manifest(dataset_name="toy", split_name="train_pool_remaining", artifact="full", root=pr)
    assert man_rem["artifact"] == "full"

