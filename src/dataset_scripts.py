from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_prosocial_dialog


def load_prosocial_dialog_bundle(root: Path | None = None, *, include_all_features: bool = False) -> dict[str, Any]:
    return load_prosocial_dialog(root=root, include_all_features=include_all_features)


def select_rows_by_source_index(df: pd.DataFrame, source_indices: list[int]) -> pd.DataFrame:
    index_to_pos = {int(v): i for i, v in enumerate(source_indices)}
    selected = df[df["source_index"].isin(source_indices)].copy()
    missing = sorted(set(int(v) for v in source_indices) - set(int(v) for v in selected["source_index"].tolist()))
    if missing:
        raise ValueError(f"Missing source_index rows: {missing}")
    selected["_sort_idx"] = [index_to_pos[int(v)] for v in selected["source_index"].tolist()]
    selected = selected.sort_values("_sort_idx").drop(columns=["_sort_idx"]).reset_index(drop=True)
    return selected


def make_dspy_sample_splits(
    test_df: pd.DataFrame,
    *,
    seed: int = 42,
    sample_size: int = 50,
    train_size: int = 25,
) -> dict[str, pd.DataFrame]:
    sample_df, _ = train_test_split(
        test_df,
        train_size=sample_size,
        random_state=seed,
        stratify=test_df["safety_label"],
    )
    sample_df = sample_df.reset_index(drop=True)

    try:
        dspy_train_df, dspy_test_df = train_test_split(
            sample_df,
            train_size=train_size,
            random_state=seed,
            stratify=sample_df["safety_label"],
        )
    except ValueError:
        dspy_train_df, dspy_test_df = train_test_split(
            sample_df,
            train_size=train_size,
            random_state=seed,
            stratify=None,
        )

    return {
        "sample_df": sample_df,
        "dspy_train_df": dspy_train_df.reset_index(drop=True),
        "dspy_test_df": dspy_test_df.reset_index(drop=True),
    }