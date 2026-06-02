# pyright: basic
"""
Load a compiled DSPy program from ``run_gepa_mipro_optimize.py`` and evaluate on a parquet split.

Uses label-balanced batches (batch_size=5 by default) and reports sentence-level macro-F1.

Example:
  uv run python scripts/eval_gepa_compiled_program.py \\
    --dataset banking-10 \\
    --program-dir results/gepa_mipro/banking-10/optimize/<stamp>/compiled_program

  # Quick smoke eval (10 rows from test/tier_20 when tier_10 is missing):
  uv run python scripts/eval_gepa_compiled_program.py \\
    --dataset banking-10 \\
    --program-dir results/gepa_mipro/banking-10/optimize/<stamp>/compiled_program \\
    --eval-tier 10 --eval-max-rows 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import dspy
import pandas as pd
from loguru import logger

from src.datasets.schema import stable_sort_for_determinism, validate_processed_samples_df
from src.dspy_gepa.artifacts import begin_run_dir, save_run_manifest, write_json
from src.dspy_gepa.batching import batches_to_manifest, build_label_balanced_batches, dataframe_to_sentence_rows
from src.dspy_gepa.data import card_label_ids
from src.dspy_gepa.eval_metrics import flatten_batch_predictions, sentence_level_metrics
from src.dspy_gepa.labels import labels_for_dataset, normalize_label_for_dataset
from src.dspy_gepa.program_load import load_compiled_program, resolve_compiled_program_dir, resolve_eval_parquet
from src.dspy_gepa.batch_classifier import examples_from_batches, parse_predicted_labels
from src.utils.env import load_dotenv_if_present

load_dotenv_if_present()

GOOGLE_OPENAI_BASE_URL = os.getenv(
    "GOOGLE_OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _api_key() -> str:
    for env in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        v = os.getenv(env)
        if v:
            return str(v)
    raise ValueError("Set GOOGLE_API_KEY (or GEMINI_API_KEY).")


def _cap_eval_rows(df: pd.DataFrame, *, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= int(max_rows):
        return df
    n = int(max_rows)
    logger.info("Capping eval set from {} to {} rows (seed={})", len(df), n, seed)
    return df.sample(n=n, random_state=int(seed)).pipe(stable_sort_for_determinism)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate saved MIPROv2 compiled program.")
    _ = ap.add_argument(
        "--program-dir",
        type=Path,
        required=True,
        help="compiled_program/ directory (or path to program.pkl inside it)",
    )
    _ = ap.add_argument(
        "--eval-parquet",
        type=Path,
        default=None,
        help="Override eval parquet; default: data/processed/<dataset>/test/tier_<eval-tier>/samples.parquet",
    )
    _ = ap.add_argument("--dataset", type=str, required=True)
    _ = ap.add_argument(
        "--eval-tier",
        type=int,
        default=20,
        help="Test tier folder (default 20). ",
    )
    _ = ap.add_argument(
        "--eval-max-rows",
        type=int,
        default=0,
        help="Max rows to evaluate (default 0 for all). Use 0 for no cap.",
    )
    _ = ap.add_argument("--output-dir", type=Path, default=None)
    _ = ap.add_argument("--executor-model", type=str, default="gemma-4-31b-it")
    _ = ap.add_argument("--batch-size", type=int, default=5)
    _ = ap.add_argument("--seed", type=int, default=42)
    _ = ap.add_argument("--num-threads", type=int, default=15)
    _ = ap.add_argument("--split-name", type=str, default="eval")
    args = ap.parse_args()

    repo = _repo_root()
    program_dir = resolve_compiled_program_dir(Path(args.program_dir))
    dataset_name = args.dataset.strip()

    eval_path = resolve_eval_parquet(
        repo_root=repo,
        dataset_name=dataset_name,
        eval_parquet=args.eval_parquet,
        eval_tier=int(args.eval_tier),
    )
    max_rows = int(args.eval_max_rows) if int(args.eval_max_rows) > 0 else None

    eval_df = pd.read_parquet(eval_path)
    validate_processed_samples_df(eval_df)
    eval_df = _cap_eval_rows(eval_df, max_rows=max_rows, seed=int(args.seed))

    label_ids = card_label_ids(dataset_name, repo_root=repo)
    allowed_labels = labels_for_dataset(dataset_name=dataset_name, label_ids=label_ids)
    batch_size = int(args.batch_size)

    rows = dataframe_to_sentence_rows(eval_df, dataset_name=dataset_name)
    batches = build_label_balanced_batches(rows, batch_size=batch_size, seed=int(args.seed))
    examples = examples_from_batches(batches, allowed_labels=allowed_labels)
    if not examples:
        raise RuntimeError(
            f"No full batches from {eval_path} (rows={len(eval_df)}, batch_size={batch_size}). "
            "Lower batch_size or add more eval rows."
        )

    api_key = _api_key()
    lm = dspy.LM(
        model=f"openai/{args.executor_model}",
        api_base=GOOGLE_OPENAI_BASE_URL,
        api_key=api_key,
        max_tokens=None,
        temperature=0.0,
        num_retries=int(os.getenv("MAIN_MAX_RETRIES", "25")),
    )
    dspy.settings.configure(lm=lm, num_threads=int(args.num_threads))

    program = load_compiled_program(program_dir, allow_pickle=True)
    logger.info("Loaded compiled program from {}", program_dir)

    run_dir = args.output_dir or begin_run_dir(
        repo_root=repo, dataset_name=dataset_name, run_kind="eval"
    )
    write_json(run_dir / "eval_batches.json", batches_to_manifest(batches, split=args.split_name))

    gold_batches: list[list[str]] = []
    pred_batches: list[list[str]] = []
    sentence_preds: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    for ex in examples:
        gold = [
            normalize_label_for_dataset(x, dataset_name=dataset_name) for x in ex.target_labels
        ]
        try:
            out = program(input_texts=ex.input_texts)
            pred = parse_predicted_labels(
                getattr(out, "predicted_labels", None),
                batch_size=batch_size,
                allowed_labels=allowed_labels,
                dataset_name=dataset_name,
            )
        except Exception as exc:
            logger.warning("Forward failed: {}", repr(exc))
            pred = ["error"] * batch_size
        gold_batches.append(gold)
        pred_batches.append(pred)
        for sid, g, p in zip(getattr(ex, "sample_ids", []), gold, pred, strict=True):
            sentence_preds.append(
                {
                    "sample_id": sid,
                    "true_label": g,
                    "pred_label": p,
                    "correct": g == p,
                }
            )

    elapsed = time.perf_counter() - t0
    y_true, y_pred = flatten_batch_predictions(gold_batches=gold_batches, pred_batches=pred_batches)
    metrics = sentence_level_metrics(y_true, y_pred, labels=allowed_labels)

    write_json(run_dir / "full_predictions.json", sentence_preds)
    write_json(run_dir / "metrics.json", metrics)
    if isinstance(metrics.get("classification_report_text"), str):
        (run_dir / "full_classification_report.txt").write_text(
            metrics["classification_report_text"], encoding="utf-8"
        )

    manifest = {
        "program_dir": str(program_dir),
        "eval_parquet": str(eval_path),
        "eval_tier": int(args.eval_tier),
        "eval_max_rows": max_rows,
        "eval_rows_used": int(len(eval_df)),
        "dataset": dataset_name,
        "batch_size": batch_size,
        "n_batches": len(examples),
        "n_sentences": len(y_true),
        "f1_macro": metrics.get("f1_macro"),
        "accuracy": metrics.get("accuracy"),
        "elapsed_s": elapsed,
    }
    save_run_manifest(run_dir, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
