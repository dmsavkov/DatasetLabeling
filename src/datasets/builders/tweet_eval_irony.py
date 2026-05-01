# pyright: basic
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import importlib
import pandas as pd

from ..io import ProcessedPaths, processed_root
from ..schema import SCHEMA, ensure_string_labels, stable_sort_for_determinism
from ..splitter import stratified_take
from ._utils import build_and_persist_seed_vault, pick_canonical_test_split_name, write_default_card


DATASET_NAME = "tweet_eval_irony"
HF_DATASET_ID = "tweet_eval"
HF_CONFIG = "irony"

# Avoid ambiguity with local `src/datasets/` package name vs HF `datasets` library.
hf_datasets = importlib.import_module("datasets")


def build_tweet_eval_irony(
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
    Build TweetEval (irony) processed tiers under `data/processed/`.

    - Loads HF `tweet_eval`, config `irony`
    - Normalizes to universal schema
    - Uses HF predefined splits to avoid overlap:
      - test_pool = HF test if present, else HF validation
      - tiers are stratified subsamples of that canonical test_pool
    """

    if not (0.0 < float(test_pool_frac) <= 1.0):
        raise ValueError("test_pool_frac must be in (0, 1]")

    load_dataset = getattr(hf_datasets, "load_dataset")
    ds = load_dataset(HF_DATASET_ID, HF_CONFIG)

    # tweet_eval/irony is typically binary: text + label (0/1)
    label_feat = ds["train"].features.get("label")
    label_names = getattr(label_feat, "names", None)
    if not label_names:
        # Fallback to stringified numeric labels
        label_names = ["0", "1"]
    label_names = [str(x) for x in label_names]

    test_split_name = pick_canonical_test_split_name(ds)

    train_df = ds["train"].to_pandas().reset_index(drop=True)
    train_df["__orig_split"] = "train"
    train_df["__orig_row"] = train_df.index.astype(int)

    test_df = ds[test_split_name].to_pandas().reset_index(drop=True)
    test_df["__orig_split"] = test_split_name
    test_df["__orig_row"] = test_df.index.astype(int)

    train_df["true_label"] = train_df["label"].map(lambda i: label_names[int(i)]).astype(str)
    test_df["true_label"] = test_df["label"].map(lambda i: label_names[int(i)]).astype(str)

    processed_train_source = pd.DataFrame(
        {
            SCHEMA.sample_id: [
                f"{s}_{int(i)}" for s, i in zip(train_df["__orig_split"].tolist(), train_df["__orig_row"].tolist())
            ],
            SCHEMA.dataset_name: [DATASET_NAME] * len(train_df),
            SCHEMA.text: train_df["text"].astype(str).tolist(),
            SCHEMA.true_label: train_df["true_label"].astype(str).tolist(),
            SCHEMA.meta_json: [
                json.dumps(
                    {
                        "hf_dataset": HF_DATASET_ID,
                        "hf_config": HF_CONFIG,
                        "orig_split": s,
                        "orig_row": int(i),
                        "label_id": int(lid),
                    },
                    ensure_ascii=True,
                )
                for s, i, lid in zip(
                    train_df["__orig_split"].tolist(),
                    train_df["__orig_row"].tolist(),
                    train_df["label"].tolist(),
                )
            ],
        }
    )
    processed_train_source = ensure_string_labels(processed_train_source, label_col=SCHEMA.true_label)
    processed_train_source = stable_sort_for_determinism(processed_train_source, id_col=SCHEMA.sample_id)

    processed_test_source = pd.DataFrame(
        {
            SCHEMA.sample_id: [f"{s}_{int(i)}" for s, i in zip(test_df["__orig_split"].tolist(), test_df["__orig_row"].tolist())],
            SCHEMA.dataset_name: [DATASET_NAME] * len(test_df),
            SCHEMA.text: test_df["text"].astype(str).tolist(),
            SCHEMA.true_label: test_df["true_label"].astype(str).tolist(),
            SCHEMA.meta_json: [
                json.dumps(
                    {
                        "hf_dataset": HF_DATASET_ID,
                        "hf_config": HF_CONFIG,
                        "orig_split": s,
                        "orig_row": int(i),
                        "label_id": int(lid),
                    },
                    ensure_ascii=True,
                )
                for s, i, lid in zip(
                    test_df["__orig_split"].tolist(),
                    test_df["__orig_row"].tolist(),
                    test_df["label"].tolist(),
                )
            ],
        }
    )
    processed_test_source = ensure_string_labels(processed_test_source, label_col=SCHEMA.true_label)
    processed_test_source = stable_sort_for_determinism(processed_test_source, id_col=SCHEMA.sample_id)

    if float(test_pool_frac) < 1.0:
        test_pool_n = max(1, int(round(len(processed_test_source) * float(test_pool_frac))))
        test_pool = stratified_take(processed_test_source, test_pool_n, SCHEMA.true_label, seed=seed)
    else:
        test_pool = processed_test_source

    origin = {"hf_dataset": HF_DATASET_ID, "hf_config": HF_CONFIG}
    builder_name = "src.datasets.builders.tweet_eval_irony:build_tweet_eval_irony"

    # Keep existing directory structure for backwards compatibility.
    out_root = processed_root(root)
    _ = ProcessedPaths(out_root)
    write_default_card(
        dataset_name=DATASET_NAME,
        description="TweetEval benchmark subset: irony classification (binary). Loaded from HF `tweet_eval` config `irony`.",
        origin=origin,
        processed_train_source=processed_train_source,
        processed_test_source=processed_test_source,
        root=root,
    )

    # basedpyright can get confused about pandas return unions in this file.
    # This file is intentionally runtime-tested by the suite.
    # pyright: ignore[reportGeneralTypeIssues]

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
        "labels": sorted(processed_test_source[SCHEMA.true_label].astype(str).unique().tolist()),
        "hf_test_split_used": test_split_name,
    }

