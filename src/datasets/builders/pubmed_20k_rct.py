# pyright: basic
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pandas as pd

hf_datasets = importlib.import_module("datasets")

from ._utils import (
    add_orig_cols,
    build_and_persist_seed_vault,
    maybe_stratified_test_pool,
    pick_canonical_test_split_name,
    to_processed_df,
    write_default_card,
)


DATASET_NAME = "pubmed_20k_rct"
HF_DATASET_ID = "pietrolesci/pubmed-20k-rct"


def _pick_text_col(df: pd.DataFrame) -> str:
    for c in ("text", "sentence"):
        if c in df.columns:
            return c
    # fallback: first object/string column that isn't the label
    for c in df.columns:
        if c.lower() in {"label", "labels"}:
            continue
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            return c
    raise RuntimeError(f"{DATASET_NAME}: could not find a text column in {list(df.columns)}")


def _pick_label_col(df: pd.DataFrame) -> str:
    for c in ("label", "labels", "target"):
        if c in df.columns:
            return c
    raise RuntimeError(f"{DATASET_NAME}: could not find a label column in {list(df.columns)}")


def build_pubmed_20k_rct(
    *,
    seed: int = 42,
    test_pool_frac: float = 1.0,
    test_tiers: tuple[int, ...] = (20, 200, 5000),
    train_seed_tiers: tuple[int, ...] = (10, 100, 5000),
    on_oversize_test_5000: str = "clip",
    on_oversize_train_seed: str = "error",
    root: Path | None = None,
) -> dict[str, Any]:
    """
    Build PubMed 20k RCT processed tiers under `data/processed/`.

    - Loads HF `pietrolesci/pubmed-20k-rct`
    - Uses canonical splits (train vs test/validation)
    - Each row is a sentence classified into {background, objective, method, result, conclusion} (typical)
    """

    if not (0.0 < float(test_pool_frac) <= 1.0):
        raise ValueError("test_pool_frac must be in (0, 1]")

    load_dataset = getattr(hf_datasets, "load_dataset")
    ds = load_dataset(HF_DATASET_ID)

    test_split_name = pick_canonical_test_split_name(ds)

    train_df = add_orig_cols(ds["train"].to_pandas(), split_name="train")
    test_df = add_orig_cols(ds[test_split_name].to_pandas(), split_name=test_split_name)

    text_col = _pick_text_col(train_df)
    label_col = _pick_label_col(train_df)

    origin = {"hf_dataset": HF_DATASET_ID}
    builder_name = "build_pubmed_20k_rct"

    processed_train_source = to_processed_df(
        train_df,
        dataset_name=DATASET_NAME,
        text_col=text_col,
        label_col=label_col,
        hf_dataset_id=HF_DATASET_ID,
        hf_config=None,
    )
    processed_test_source = to_processed_df(
        test_df,
        dataset_name=DATASET_NAME,
        text_col=text_col,
        label_col=label_col,
        hf_dataset_id=HF_DATASET_ID,
        hf_config=None,
    )
    test_pool = maybe_stratified_test_pool(processed_test_source, test_pool_frac=test_pool_frac, seed=seed)

    write_default_card(
        dataset_name=DATASET_NAME,
        description="PubMed 20k RCT sentence role classification. Using HF canonical splits.",
        origin=origin,
        processed_train_source=processed_train_source,
        processed_test_source=processed_test_source,
        root=root,
    )

    _, train_written, test_written = build_and_persist_seed_vault(
        dataset_name=DATASET_NAME,
        processed_train_source=processed_train_source,
        test_pool=test_pool,
        test_tiers=test_tiers,
        train_seed_tiers=train_seed_tiers,
        seed=seed,
        builder_name=builder_name,
        origin=origin,
        on_oversize_test_5000=on_oversize_test_5000,
        on_oversize_train_seed=on_oversize_train_seed,
        root=root,
    )

    return {
        "dataset_name": DATASET_NAME,
        "test_pool_rows_total": int(len(processed_test_source)),
        "test_pool_rows": int(len(test_pool)),
        "test_tiers_written": test_written,
        "train_written": train_written,
        "hf_test_split_used": test_split_name,
    }

