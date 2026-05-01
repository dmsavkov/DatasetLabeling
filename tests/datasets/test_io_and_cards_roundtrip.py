from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.datasets.cards import DatasetCard, default_card_path, read_card_json, write_card_json
from src.datasets.io import load_manifest, load_processed_tier, processed_root, save_processed_tier
from src.datasets.schema import SCHEMA


def test_save_load_manifest_and_card_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            SCHEMA.sample_id: ["s1", "s2"],
            SCHEMA.dataset_name: ["ds"] * 2,
            SCHEMA.text: ["hello", "world"],
            SCHEMA.true_label: ["a", "b"],
            SCHEMA.meta_json: ["{}", "{}"],
        }
    )
    _ = save_processed_tier(
        df,
        dataset_name="ds",
        split_name="test",
        tier_size=2,
        seed=1,
        builder="test",
        origin={"x": 1},
        root=tmp_path,
    )

    pr = processed_root(tmp_path)
    loaded = load_processed_tier(dataset_name="ds", split_name="test", tier_size=2, root=tmp_path)
    assert loaded.shape[0] == 2
    man = load_manifest(dataset_name="ds", split_name="test", tier_size=2, root=tmp_path)
    assert man["dataset_name"] == "ds"
    assert man["split_name"] == "test"

    card = DatasetCard(dataset_name="ds", description="x", origin={"x": 1}, labels=["a", "b"], sample_count=2)
    cp = default_card_path(processed_root_dir=pr, dataset_name="ds")
    write_card_json(cp, card)
    card2 = read_card_json(cp)
    assert card2.dataset_name == "ds"
    assert card2.labels == ["a", "b"]

