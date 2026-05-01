from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.datasets.io import processed_root, save_processed_tier
from src.datasets.schema import SCHEMA
from src.experiments.run import run_experiment


def test_run_experiment_end_to_end_toy(tmp_path: Path) -> None:
    # Build toy processed parquets resembling Seed Vault artifacts.
    train_df = pd.DataFrame(
        {
            SCHEMA.sample_id: [f"tr_{i}" for i in range(30)],
            SCHEMA.dataset_name: ["toy"] * 30,
            SCHEMA.text: [f"text {i}" for i in range(30)],
            SCHEMA.true_label: ["a"] * 15 + ["b"] * 15,
            SCHEMA.meta_json: ["{}"] * 30,
        }
    )
    test_df = pd.DataFrame(
        {
            SCHEMA.sample_id: [f"te_{i}" for i in range(20)],
            SCHEMA.dataset_name: ["toy"] * 20,
            SCHEMA.text: [f"text {i}" for i in range(20)],
            SCHEMA.true_label: ["a"] * 10 + ["b"] * 10,
            SCHEMA.meta_json: ["{}"] * 20,
        }
    )

    _ = save_processed_tier(
        train_df,
        dataset_name="toy",
        split_name="train_seed",
        tier_size=100,
        seed=1,
        builder="test",
        origin={"x": 1},
        root=tmp_path,
    )
    _ = save_processed_tier(
        test_df,
        dataset_name="toy",
        split_name="test",
        tier_size=200,
        seed=1,
        builder="test",
        origin={"x": 1},
        root=tmp_path,
    )

    pr = processed_root(tmp_path)
    cfg = {
        "name": "toy_svm",
        "seed": 1,
        "train_data": str(pr / "toy" / "train_seed" / "tier_100" / "samples.parquet"),
        "test_data": str(pr / "toy" / "test" / "tier_200" / "samples.parquet"),
        "output_dir": str(tmp_path / "out"),
        "model": {"kind": "sklearn_svm", "params": {}},
    }
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    # write as json but with .yaml suffix is not supported; use .json
    cfg_path = tmp_path / "exp.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    res = run_experiment(cfg_path)
    assert (Path(res["output_dir"]) / "predictions.csv").exists()
    assert (Path(res["output_dir"]) / "report.json").exists()

