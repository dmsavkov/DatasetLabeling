from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .schema import SCHEMA
from .splitter import stratified_take


@dataclass(frozen=True, slots=True)
class SeedVaultBuild:
    """Train seeds keyed by tier size (e.g. 10, 100); disjoint sequential draws from train_source."""

    train_seed_tiers: dict[int, pd.DataFrame]
    train_pool_remaining: pd.DataFrame
    test_tiers: dict[int, pd.DataFrame]
    statuses: dict[str, str]


def take_with_policy(
    df: pd.DataFrame,
    n: int,
    *,
    label_col: str,
    seed: int,
    policy: str,
) -> tuple[pd.DataFrame, str]:
    if n <= len(df):
        return stratified_take(df, n, label_col, seed=seed), "ok"
    if policy == "clip":
        return df.copy(), "clipped"
    raise ValueError(f"Requested n={n} > available={len(df)} (policy={policy})")


def build_seed_vault(
    *,
    train_source: pd.DataFrame,
    test_source: pd.DataFrame,
    seed: int,
    train_seed_tiers: tuple[int, ...] = (10, 100, 5000),
    test_tiers: tuple[int, ...] = (20, 200, 5000),
    on_oversize_train_seed: str = "error",
    on_oversize_test_5000: str = "clip",
) -> SeedVaultBuild:
    """
    Pure helper: create Seed Vault artifacts from normalized processed dataframes.

    Train tiers are applied in ascending order by tier size (unique). Each tier is a
    stratified draw from the remaining train rows after prior tiers (disjoint).
    """

    statuses: dict[str, str] = {}

    train_out: dict[int, pd.DataFrame] = {}
    remaining = train_source.reset_index(drop=True).copy()

    tier_sizes = sorted({int(x) for x in train_seed_tiers})
    for tier_n in tier_sizes:
        policy = "clip" if int(tier_n) == 5000 else on_oversize_train_seed
        df_t, st = take_with_policy(
            remaining,
            tier_n,
            label_col=SCHEMA.true_label,
            seed=seed,
            policy=policy,
        )
        train_out[tier_n] = df_t
        statuses[f"train_seed_{tier_n}"] = st

        taken_ids = set(df_t[SCHEMA.sample_id].astype(str).tolist())
        remaining = remaining[~remaining[SCHEMA.sample_id].astype(str).isin(list(taken_ids))].reset_index(drop=True)

    pool_remaining = remaining

    out_test_tiers: dict[int, pd.DataFrame] = {}
    for t in test_tiers:
        policy = on_oversize_test_5000 if int(t) == 5000 else "error"
        df_t, st = take_with_policy(test_source, int(t), label_col=SCHEMA.true_label, seed=seed, policy=policy)
        out_test_tiers[int(t)] = df_t
        statuses[f"test_{int(t)}"] = st

    return SeedVaultBuild(
        train_seed_tiers=train_out,
        train_pool_remaining=pool_remaining,
        test_tiers=out_test_tiers,
        statuses=statuses,
    )
