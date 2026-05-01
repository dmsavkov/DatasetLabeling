from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Any

import pandas as pd

# NOTE: This repo has a local package named `datasets` (under `src/datasets/`).
# Runtime imports resolve the Hugging Face library correctly, but some type-checker
# configs can misinterpret `import datasets` here as an implicit relative import.
# Using importlib avoids that ambiguity.
hf_datasets = importlib.import_module("datasets")

from ..cards import DatasetCard, write_card_json
from ..io import ProcessedPaths, processed_root
from ..schema import SCHEMA, ensure_string_labels, stable_sort_for_determinism
from ..splitter import stratified_take
from ._utils import build_and_persist_seed_vault


HF_DATASET_ID = "banking77"
DATASET_NAME = "banking-10"


def select_top_labels(counts: dict[str, int], k: int) -> list[str]:
    """
    Deterministic top-k selection:
    sort by (-count, label_name).
    """

    items = [(str(label), int(cnt)) for label, cnt in counts.items()]
    items.sort(key=lambda x: (-x[1], x[0]))
    return [label for label, _ in items[:k]]


def _hf_to_df(split_ds, *, split_name: str) -> pd.DataFrame:
    # banking77 schema is typically: {"text": str, "label": int}
    df = split_ds.to_pandas()
    df = df.reset_index(drop=True)
    df["__orig_split"] = split_name
    df["__orig_row"] = df.index.astype(int)
    return df


def _label_names_from_hf(ds_dict) -> list[str]:
    feat = ds_dict["train"].features.get("label")
    names = getattr(feat, "names", None)
    if not names:
        raise RuntimeError("Could not read label names from HF dataset features")
    return [str(x) for x in names]


def build_banking10(
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
    Build Banking10 from HF `banking77` and persist tiered test splits under `data/processed/`.

    Strategy:
    - pick top-10 labels by frequency in the HF train split
    - use HF predefined splits to avoid overlap:
      - train_pool = HF train filtered to top-10 (for future baselines)
      - test_pool = HF test if present, else HF validation
    - tiers (20/200/5000) are stratified subsamples of that canonical test_pool
    """

    if not (0.0 < float(test_pool_frac) <= 1.0):
        raise ValueError("test_pool_frac must be in (0, 1]")

    load_dataset = getattr(hf_datasets, "load_dataset")
    ds = load_dataset(HF_DATASET_ID)
    label_names = _label_names_from_hf(ds)

    # Frequency counts on train split using label names.
    train_df = _hf_to_df(ds["train"], split_name="train")
    train_counts = (
        train_df["label"]
        .map(lambda i: label_names[int(i)])
        .astype(str)
        .value_counts()
        .to_dict()
    )
    top10 = select_top_labels(train_counts, 10)

    if "test" in ds:
        test_split_name = "test"
    elif "validation" in ds:
        test_split_name = "validation"
    else:
        raise RuntimeError("HF dataset has no predefined test/validation split; refusing to create evaluation pool implicitly.")

    train_pool_df = _hf_to_df(ds["train"], split_name="train")
    test_pool_df = _hf_to_df(ds[test_split_name], split_name=test_split_name)

    train_pool_df["true_label"] = train_pool_df["label"].map(lambda i: label_names[int(i)]).astype(str)
    train_pool_df = train_pool_df[train_pool_df["true_label"].isin(list(top10))].reset_index(drop=True)

    test_pool_df["true_label"] = test_pool_df["label"].map(lambda i: label_names[int(i)]).astype(str)
    test_pool_df = test_pool_df[test_pool_df["true_label"].isin(list(top10))].reset_index(drop=True)

    # Normalize train/test sources to universal processed schema.
    processed_train_source = pd.DataFrame(
        {
            SCHEMA.sample_id: [
                f"{s}_{int(i)}"
                for s, i in zip(train_pool_df["__orig_split"].tolist(), train_pool_df["__orig_row"].tolist())
            ],
            SCHEMA.dataset_name: [DATASET_NAME] * len(train_pool_df),
            SCHEMA.text: train_pool_df["text"].astype(str).tolist(),
            SCHEMA.true_label: train_pool_df["true_label"].astype(str).tolist(),
            SCHEMA.meta_json: [
                json.dumps(
                    {
                        "hf_dataset": HF_DATASET_ID,
                        "orig_split": s,
                        "orig_row": int(i),
                        "label_id": int(lid),
                    },
                    ensure_ascii=True,
                )
                for s, i, lid in zip(
                    train_pool_df["__orig_split"].tolist(),
                    train_pool_df["__orig_row"].tolist(),
                    train_pool_df["label"].tolist(),
                )
            ],
        }
    )
    processed_train_source = ensure_string_labels(processed_train_source, label_col=SCHEMA.true_label)
    processed_train_source = stable_sort_for_determinism(processed_train_source, id_col=SCHEMA.sample_id)

    processed_test_pool = pd.DataFrame(
        {
            SCHEMA.sample_id: [
                f"{s}_{int(i)}"
                for s, i in zip(test_pool_df["__orig_split"].tolist(), test_pool_df["__orig_row"].tolist())
            ],
            SCHEMA.dataset_name: [DATASET_NAME] * len(test_pool_df),
            SCHEMA.text: test_pool_df["text"].astype(str).tolist(),
            SCHEMA.true_label: test_pool_df["true_label"].astype(str).tolist(),
            SCHEMA.meta_json: [
                json.dumps(
                    {
                        "hf_dataset": HF_DATASET_ID,
                        "orig_split": s,
                        "orig_row": int(i),
                        "label_id": int(lid),
                    },
                    ensure_ascii=True,
                )
                for s, i, lid in zip(
                    test_pool_df["__orig_split"].tolist(),
                    test_pool_df["__orig_row"].tolist(),
                    test_pool_df["label"].tolist(),
                )
            ],
        }
    )
    processed_test_pool = ensure_string_labels(processed_test_pool, label_col=SCHEMA.true_label)
    processed_test_pool = stable_sort_for_determinism(processed_test_pool, id_col=SCHEMA.sample_id)

    # Canonical test pool is the HF predefined split (optionally downsampled deterministically).
    if float(test_pool_frac) < 1.0:
        test_pool_n = max(1, int(round(len(processed_test_pool) * float(test_pool_frac))))
        test_pool = stratified_take(processed_test_pool, test_pool_n, SCHEMA.true_label, seed=seed)
    else:
        test_pool = processed_test_pool

    origin = {"hf_dataset": HF_DATASET_ID, "hf_config": None}
    builder_name = "src.datasets.builders.banking10:build_banking10"

    out_root = processed_root(root)
    paths = ProcessedPaths(out_root)
    dataset_dir = paths.dataset_dir(DATASET_NAME)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Stable label-space for the evaluator (even if tier=20 misses rare labels).
    all_labels = list(top10)
    card = DatasetCard(
        dataset_name=DATASET_NAME,
        description="Derived from HF banking77 by keeping the 10 most frequent intents (frequency counted on original train split).",
        origin=origin,
        labels=sorted(all_labels),
        sample_count=int(len(train_pool_df) + len(test_pool_df)),
    )
    write_card_json(dataset_dir / "dataset_card.json", card)

    extra_manifest = {
        "derivation": {
            "top_k": 10,
            "top_labels": top10,
            "frequency_source_split": "train",
            "tie_break": "(-count, label_name)",
        }
    }

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
        "test_pool_rows_total": int(len(processed_test_pool)),
        "test_pool_rows": int(len(test_pool)),
        "test_tiers_written": test_written,
        "train_written": train_written,
        "top_labels": top10,
        "hf_test_split_used": test_split_name,
    }

