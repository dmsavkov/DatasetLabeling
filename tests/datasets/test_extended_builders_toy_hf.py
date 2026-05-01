from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.datasets.schema import SCHEMA


class _ToySplit:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df.copy()


def test_build_implicit_hate_offline_monkeypatched_hf(tmp_path: Path, monkeypatch) -> None:
    from src.datasets.builders import implicit_hate

    train = pd.DataFrame({"text": ["a", "b", "c", "d"], "hateful_layer": ["HS", "Non-HS", "HS", "Non-HS"]})
    test = pd.DataFrame({"text": ["e", "f"], "hateful_layer": ["HS", "Non-HS"]})

    def fake_load_dataset(_id: str):
        assert _id == implicit_hate.HF_DATASET_ID
        return {"train": _ToySplit(train), "test": _ToySplit(test)}

    monkeypatch.setattr(implicit_hate.hf_datasets, "load_dataset", fake_load_dataset)

    out = implicit_hate.build_implicit_hate(
        seed=1,
        test_tiers=(2,),
        train_seed_tiers=(2,),
        root=tmp_path,
        on_oversize_test_5000="clip",
    )
    assert out["dataset_name"] == "implicit_hate"
    pr = tmp_path / "data" / "processed" / "implicit_hate"
    assert (pr / "train_seed" / "tier_2" / "samples.parquet").exists()
    assert (pr / "test" / "tier_2" / "samples.parquet").exists()


def test_build_pubmed_20k_rct_offline_monkeypatched_hf(tmp_path: Path, monkeypatch) -> None:
    from src.datasets.builders import pubmed_20k_rct

    train = pd.DataFrame({"sentence": ["s1", "s2", "s3", "s4"], "label": ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSION"]})
    test = pd.DataFrame({"sentence": ["s5", "s6"], "label": ["BACKGROUND", "RESULTS"]})

    def fake_load_dataset(_id: str):
        assert _id == pubmed_20k_rct.HF_DATASET_ID
        return {"train": _ToySplit(train), "test": _ToySplit(test)}

    monkeypatch.setattr(pubmed_20k_rct.hf_datasets, "load_dataset", fake_load_dataset)

    out = pubmed_20k_rct.build_pubmed_20k_rct(
        seed=1,
        test_tiers=(2,),
        train_seed_tiers=(2,),
        root=tmp_path,
        on_oversize_test_5000="clip",
    )
    assert out["dataset_name"] == "pubmed_20k_rct"
    pr = tmp_path / "data" / "processed" / "pubmed_20k_rct"
    assert (pr / "train_seed" / "tier_2" / "samples.parquet").exists()
    assert (pr / "test" / "tier_2" / "samples.parquet").exists()

    # sanity: universal schema present in saved parquet
    df = pd.read_parquet(pr / "test" / "tier_2" / "samples.parquet")
    for c in (SCHEMA.sample_id, SCHEMA.dataset_name, SCHEMA.text, SCHEMA.true_label):
        assert c in df.columns

