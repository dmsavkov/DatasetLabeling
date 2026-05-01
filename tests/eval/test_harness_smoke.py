from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.datasets.io import save_processed_tier
from src.datasets.schema import SCHEMA
from src.eval.harness import evaluate_predictor_on_tier
from src.models.interfaces import Prediction, Usage


@dataclass(frozen=True, slots=True)
class DummyPredictor:
    fixed_label: str

    @property
    def name(self) -> str:
        return "dummy"

    def predict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
        lab = self.fixed_label if self.fixed_label in allowed_labels else allowed_labels[0]
        return [Prediction(pred_label=lab, confidence=None, reason=None, probs=None, usage=Usage(1, 1), raw=None) for _ in texts]


def test_eval_harness_writes_artifacts(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            SCHEMA.sample_id: ["a", "b", "c", "d"],
            SCHEMA.dataset_name: ["toy"] * 4,
            SCHEMA.text: ["t1", "t2", "t3", "t4"],
            SCHEMA.true_label: ["x", "y", "x", "y"],
            SCHEMA.meta_json: ["{}"] * 4,
        }
    )
    _ = save_processed_tier(
        df,
        dataset_name="toy",
        split_name="test",
        tier_size=4,
        seed=42,
        builder="test",
        origin={"test": True},
        root=tmp_path,
    )

    out_dir = tmp_path / "out"
    res = evaluate_predictor_on_tier(
        DummyPredictor(fixed_label="x"),
        dataset_name="toy",
        split_name="test",
        tier_size=4,
        output_dir=out_dir,
        processed_root=tmp_path / "data" / "processed",
    )

    assert (out_dir / "predictions.csv").exists()
    assert (out_dir / "report.json").exists()
    assert "metrics" in res.report

