# pyright: basic
"""
Build GEPA optimizer train/val parquets from a completed huge-prediction run.

See docs/gepa_pipeline.md. Example:
  uv run python scripts/build_gepa_optimizer_sets.py \\
    --dataset <dataset> \\
    --huge-prediction-dir data/huge_prediction_representatives/<dataset>/<stamp>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from src.data_selection.gepa_optimizer_sets import build_gepa_train_val_from_huge_prediction
from src.utils.env import load_dotenv_if_present

load_dotenv_if_present()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GEPA train/val from huge-prediction artifacts.")
    _ = ap.add_argument("--dataset", type=str, required=True)
    _ = ap.add_argument(
        "--huge-prediction-dir",
        type=Path,
        required=True,
        help="Directory with predictions.parquet, contrastive_edges.parquet, etc.",
    )
    _ = ap.add_argument("--train-total", type=int, default=50)
    _ = ap.add_argument("--train-easy-frac", type=float, default=0.2)
    _ = ap.add_argument("--val-total", type=int, default=70)
    _ = ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    huge_dir = Path(args.huge_prediction_dir)
    if not (huge_dir / "predictions.parquet").exists():
        raise FileNotFoundError(f"Missing predictions.parquet in {huge_dir}")

    logger.info("Building GEPA sets from {}", huge_dir)
    result = build_gepa_train_val_from_huge_prediction(
        huge_dir,
        dataset_name=str(args.dataset).strip(),
        train_total=int(args.train_total),
        train_easy_fraction=float(args.train_easy_frac),
        val_total=int(args.val_total),
        seed=int(args.seed),
    )
    print(json.dumps({"output_dir": str(result.output_dir), "manifest": result.manifest}, indent=2))


if __name__ == "__main__":
    main()
