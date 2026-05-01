# pyright: basic
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..cards import DatasetCard, write_card_json
from ..io import processed_root, save_processed_artifact, save_processed_tier
from ..schema import SCHEMA, ensure_string_labels, stable_sort_for_determinism
from ..seed_vault import SeedVaultBuild, build_seed_vault
from ..splitter import stratified_take


def pick_canonical_test_split_name(ds: Any) -> str:
    if "test" in ds:
        return "test"
    if "validation" in ds:
        return "validation"
    raise RuntimeError("HF dataset has no predefined test/validation split; refusing to create evaluation pool implicitly.")


def add_orig_cols(df: pd.DataFrame, *, split_name: str) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out["__orig_split"] = split_name
    out["__orig_row"] = out.index.astype(int)
    return out


def to_processed_df(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    text_col: str,
    label_col: str,
    hf_dataset_id: str,
    hf_config: str | None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            SCHEMA.sample_id: [
                f"{s}_{int(i)}" for s, i in zip(df["__orig_split"].tolist(), df["__orig_row"].tolist())
            ],
            SCHEMA.dataset_name: [dataset_name] * len(df),
            SCHEMA.text: df[text_col].astype(str).tolist(),
            SCHEMA.true_label: df[label_col].astype(str).tolist(),
            SCHEMA.meta_json: [
                json.dumps(
                    {
                        "hf_dataset": hf_dataset_id,
                        "hf_config": hf_config,
                        "orig_split": s,
                        "orig_row": int(i),
                        "label_col": label_col,
                    },
                    ensure_ascii=True,
                )
                for s, i in zip(df["__orig_split"].tolist(), df["__orig_row"].tolist())
            ],
        }
    )
    out = ensure_string_labels(out, label_col=SCHEMA.true_label)
    out = stable_sort_for_determinism(out, id_col=SCHEMA.sample_id)
    return out


def maybe_stratified_test_pool(
    processed_test_source: pd.DataFrame,
    *,
    test_pool_frac: float,
    seed: int,
) -> pd.DataFrame:
    if float(test_pool_frac) < 1.0:
        test_pool_n = max(1, int(round(len(processed_test_source) * float(test_pool_frac))))
        return stratified_take(processed_test_source, test_pool_n, SCHEMA.true_label, seed=seed)
    return processed_test_source


def write_default_card(
    *,
    dataset_name: str,
    description: str,
    origin: dict[str, Any],
    processed_train_source: pd.DataFrame,
    processed_test_source: pd.DataFrame,
    root: Path | None,
) -> None:
    out_root = processed_root(root)
    dataset_dir = out_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(processed_train_source[SCHEMA.true_label].astype(str).unique().tolist())
    card = DatasetCard(
        dataset_name=dataset_name,
        description=description,
        origin=origin,
        labels=labels,
        sample_count=int(len(processed_train_source) + len(processed_test_source)),
    )
    write_card_json(dataset_dir / "dataset_card.json", card)


def build_and_persist_seed_vault(
    *,
    dataset_name: str,
    processed_train_source: pd.DataFrame,
    test_pool: pd.DataFrame,
    test_tiers: tuple[int, ...],
    train_seed_tiers: tuple[int, ...],
    seed: int,
    builder_name: str,
    origin: dict[str, Any],
    on_oversize_test_5000: str,
    on_oversize_train_seed: str,
    root: Path | None,
) -> tuple[SeedVaultBuild, dict[str, str], dict[int, str]]:
    sv = build_seed_vault(
        train_source=processed_train_source,
        test_source=test_pool,
        seed=seed,
        train_seed_tiers=tuple(int(x) for x in train_seed_tiers),
        test_tiers=tuple(int(x) for x in test_tiers),
        on_oversize_train_seed=on_oversize_train_seed,
        on_oversize_test_5000=on_oversize_test_5000,
    )

    train_written: dict[str, str] = {}
    for tier_n in sorted(sv.train_seed_tiers.keys()):
        stat_key = f"train_seed_{tier_n}"
        train_written[stat_key] = str(
            save_processed_tier(
                sv.train_seed_tiers[tier_n],
                dataset_name=dataset_name,
                split_name="train_seed",
                tier_size=int(tier_n),
                seed=seed,
                builder=builder_name,
                origin=origin,
                extra_manifest={"seed_vault": {"kind": "train_seed", "tier": int(tier_n), "status": sv.statuses[stat_key]}},
                root=root,
            )
        )
    train_written["train_pool_remaining"] = str(
        save_processed_artifact(
            sv.train_pool_remaining,
            dataset_name=dataset_name,
            split_name="train_pool_remaining",
            artifact="full",
            seed=seed,
            builder=builder_name,
            origin=origin,
            extra_manifest={"seed_vault": {"kind": "train_pool_remaining"}},
            root=root,
        )
    )

    test_written: dict[int, str] = {}
    for tier in test_tiers:
        tier_df = sv.test_tiers[int(tier)]
        status = sv.statuses[f"test_{int(tier)}"]
        tier_path = save_processed_tier(
            tier_df,
            dataset_name=dataset_name,
            split_name="test",
            tier_size=int(tier),
            seed=seed,
            builder=builder_name,
            origin=origin,
            extra_manifest={"seed_vault": {"kind": "test", "tier": int(tier), "status": status}},
            root=root,
        )
        test_written[int(tier)] = str(tier_path)

    return sv, train_written, test_written

