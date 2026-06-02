# pyright: basic
"""
MIPROv2 (GEPA-style) prompt optimization on GEPA train/val pools (extended-suite datasets).

Train metric: batch mean accuracy (per sentence within batch).
Post-compile val check: sentence-level macro-F1 on gepa_val.parquet.

See docs/gepa_pipeline.md. Example:
  uv run python scripts/run_gepa_mipro_optimize.py \\
    --dataset <dataset> \\
    --gepa-sets-dir data/gepa_optimizer_sets/<dataset>/<stamp>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import dspy
from dspy.teleprompt import MIPROv2
from loguru import logger
from tqdm.auto import tqdm

from src.dspy_gepa.artifacts import begin_run_dir, save_run_manifest, write_json
from src.datasets.schema import SCHEMA
from src.dspy_gepa.batching import batches_to_manifest, build_label_balanced_batches, dataframe_to_sentence_rows
from src.dspy_gepa.data import card_label_ids, load_gepa_optimizer_sets, resolve_gepa_sets_dir
from src.dspy_gepa.eval_metrics import flatten_batch_predictions, sentence_level_metrics
from src.dspy_gepa.batch_classifier import (
    batch_metric_factory,
    classifier_for_dataset,
    examples_from_batches,
    parse_predicted_labels,
)
from src.dspy_gepa.labels import labels_for_dataset, normalize_label_for_dataset
from src.utils.env import load_dotenv_if_present

load_dotenv_if_present()

GOOGLE_OPENAI_BASE_URL = os.getenv(
    "GOOGLE_OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
DEFAULT_EXECUTOR = "gemma-4-31b-it"
DEFAULT_REFLECTOR = "gemini-3.1-flash-lite-preview"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _api_key() -> str:
    for env in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        v = os.getenv(env)
        if v:
            return str(v)
    raise ValueError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) for Google models.")


def _make_lms(*, executor: str, reflector: str, api_key: str) -> tuple[dspy.LM, dspy.LM]:
    common: dict[str, Any] = {
        "api_base": GOOGLE_OPENAI_BASE_URL,
        "api_key": api_key,
        "max_tokens": None,
        "num_retries": int(os.getenv("MAIN_MAX_RETRIES", "25")),
    }
    task_lm = dspy.LM(model=f"openai/{executor}", temperature=0.0, **common)
    # High temperature for diverse instruction proposals; reflector explores rule variants.
    reflector_lm = dspy.LM(
        model=f"openai/{reflector}",
        temperature=float(os.getenv("REFLECTOR_TEMPERATURE", "1.0")),
        **common,
    )
    return task_lm, reflector_lm


def _evaluate_program_on_examples(
    program: dspy.Module,
    examples: list[dspy.Example],
    *,
    dataset_name: str,
    batch_size: int,
    allowed_labels: list[str],
) -> dict[str, Any]:
    gold_batches: list[list[str]] = []
    pred_batches: list[list[str]] = []
    batch_rows: list[dict[str, Any]] = []

    for ex in tqdm(examples, desc="eval_batches"):
        gold = [normalize_label_for_dataset(x, dataset_name=dataset_name) for x in ex.target_labels]
        try:
            out = program(input_texts=ex.input_texts)
            pred = parse_predicted_labels(
                getattr(out, "predicted_labels", None),
                batch_size=batch_size,
                allowed_labels=allowed_labels,
                dataset_name=dataset_name,
            )
        except Exception as exc:
            logger.warning("Batch forward failed: {}", repr(exc))
            pred = ["error"] * batch_size
        gold_batches.append(gold)
        pred_batches.append(pred)
        batch_rows.append(
            {
                "batch_id": int(getattr(ex, "batch_id", -1)),
                "sample_ids": list(getattr(ex, "sample_ids", [])),
                "gold": gold,
                "pred": pred,
                "batch_accuracy": sum(1 for g, p in zip(gold, pred, strict=True) if g == p) / float(batch_size),
            }
        )

    y_true, y_pred = flatten_batch_predictions(gold_batches=gold_batches, pred_batches=pred_batches)
    metrics = sentence_level_metrics(y_true, y_pred, labels=allowed_labels)
    metrics["n_batches"] = len(examples)
    metrics["batch_results"] = batch_rows
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="MIPROv2 optimize batch classifier on GEPA pools.")
    _ = ap.add_argument("--dataset", type=str, default="pubmed_20k_rct")
    _ = ap.add_argument("--gepa-sets-dir", type=Path, default=None)
    _ = ap.add_argument("--output-dir", type=Path, default=None)
    _ = ap.add_argument("--executor-model", type=str, default=DEFAULT_EXECUTOR)
    _ = ap.add_argument("--reflector-model", type=str, default=DEFAULT_REFLECTOR)
    _ = ap.add_argument("--batch-size", type=int, default=5)
    _ = ap.add_argument("--minibatch-size", type=int, default=4)
    _ = ap.add_argument("--num-candidates", type=int, default=20)
    _ = ap.add_argument("--init-temperature", type=float, default=1.0)
    _ = ap.add_argument(
        "--num-shots",
        type=int,
        default=0,
        help="max_labeled_demos (fixed human-labeled demos in MIPROv2)",
    )
    _ = ap.add_argument(
        "--max-bootstrapped-demos",
        type=int,
        default=3,
        help="max_bootstrapped_demos (model-generated demos)",
    )
    _ = ap.add_argument("--seed", type=int, default=42)
    _ = ap.add_argument("--num-threads", type=int, default=15)
    _ = ap.add_argument("--skip-preliminary", action="store_true")
    _ = ap.add_argument("--prelim-only", action="store_true")
    args = ap.parse_args()

    repo = _repo_root()
    sets_dir = resolve_gepa_sets_dir(
        dataset_name=args.dataset.strip(),
        explicit=args.gepa_sets_dir,
        repo_root=repo,
    )
    pools = load_gepa_optimizer_sets(sets_dir)
    try:
        label_ids = card_label_ids(args.dataset.strip(), repo_root=repo)
    except Exception:
        label_ids = sorted(pools.train_df[SCHEMA.true_label].astype(str).unique().tolist())
    dataset_name = args.dataset.strip()
    allowed_labels = labels_for_dataset(dataset_name=dataset_name, label_ids=label_ids)

    batch_size = int(args.batch_size)
    seed = int(args.seed)

    train_rows = dataframe_to_sentence_rows(pools.train_df, dataset_name=dataset_name)
    val_rows = dataframe_to_sentence_rows(pools.val_df, dataset_name=dataset_name)
    train_batches = build_label_balanced_batches(train_rows, batch_size=batch_size, seed=seed)
    val_batches = build_label_balanced_batches(val_rows, batch_size=batch_size, seed=seed + 1)

    trainset = examples_from_batches(train_batches, allowed_labels=allowed_labels)
    valset = examples_from_batches(val_batches, allowed_labels=allowed_labels)
    if not trainset or not valset:
        raise RuntimeError("Need at least one full batch in train and val GEPA pools.")

    run_dir = args.output_dir or begin_run_dir(
        repo_root=repo, dataset_name=args.dataset.strip(), run_kind="optimize"
    )
    write_json(run_dir / "gepa_sets_source.json", {"gepa_sets_dir": str(sets_dir), "manifest": pools.manifest})
    write_json(run_dir / "train_batches.json", batches_to_manifest(train_batches, split="train"))
    write_json(run_dir / "val_batches.json", batches_to_manifest(val_batches, split="val"))

    api_key = _api_key()
    task_lm, reflector_lm = _make_lms(
        executor=str(args.executor_model),
        reflector=str(args.reflector_model),
        api_key=api_key,
    )
    dspy.settings.configure(lm=task_lm, num_threads=int(args.num_threads))

    metric = batch_metric_factory(
        batch_size=batch_size,
        allowed_labels=allowed_labels,
        dataset_name=dataset_name,
    )

    if not args.skip_preliminary:
        logger.info("Preliminary forward on first val batch")
        probe = classifier_for_dataset(dataset_name=dataset_name, label_ids=label_ids)
        out = probe(input_texts=valset[0].input_texts)
        parsed = parse_predicted_labels(
            getattr(out, "predicted_labels", None),
            batch_size=batch_size,
            allowed_labels=allowed_labels,
            dataset_name=dataset_name,
        )
        logger.info("Preliminary parsed labels: {}", parsed)
        if args.prelim_only:
            return

    labeled_shots = int(args.num_shots)
    bootstrapped_shots = int(args.max_bootstrapped_demos)
    optimizer = MIPROv2(
        metric=metric,
        prompt_model=reflector_lm,
        task_model=task_lm,
        num_candidates=int(args.num_candidates),
        init_temperature=float(args.init_temperature),
        max_bootstrapped_demos=bootstrapped_shots,
        max_labeled_demos=labeled_shots,
        seed=seed,
        num_threads=int(args.num_threads),
        auto=None
    )

    logger.info(
        "MIPROv2 compile: train_batches={} val_batches={} minibatch_size={}",
        len(trainset),
        len(valset),
        args.minibatch_size,
    )
    t0 = time.perf_counter()
    student = classifier_for_dataset(dataset_name=dataset_name, label_ids=label_ids)
    optimized = optimizer.compile(
        student=student,
        trainset=trainset,
        valset=valset,
        minibatch_size=int(args.minibatch_size),
        requires_permission_to_run=False,
        seed=seed,
        num_trials=int(1.5 * int(args.num_candidates)),
    )
    compile_s = time.perf_counter() - t0

    program_dir = run_dir / "compiled_program"
    try:
        optimized.save(str(program_dir), save_program=True)
        write_json(run_dir / "optimized_program_state.json", optimized.dump_state())
    except Exception as exc:
        logger.warning("Could not save compiled program bundle: {}", repr(exc))

    val_metrics = _evaluate_program_on_examples(
        optimized,
        valset,
        dataset_name=dataset_name,
        batch_size=batch_size,
        allowed_labels=allowed_labels,
    )
    write_json(run_dir / "val_eval_macro_f1.json", val_metrics)
    if isinstance(val_metrics.get("classification_report_text"), str):
        (run_dir / "val_eval_classification_report.txt").write_text(
            val_metrics["classification_report_text"], encoding="utf-8"
        )

    manifest = {
        "dataset": args.dataset.strip(),
        "gepa_sets_dir": str(sets_dir),
        "executor_model": args.executor_model,
        "reflector_model": args.reflector_model,
        "batch_size": batch_size,
        "minibatch_size": int(args.minibatch_size),
        "num_candidates": int(args.num_candidates),
        "init_temperature": float(args.init_temperature),
        "num_shots": labeled_shots,
        "max_bootstrapped_demos": bootstrapped_shots,
        "seed": seed,
        "train_batches": len(trainset),
        "val_batches": len(valset),
        "compile_seconds": compile_s,
        "val_f1_macro": val_metrics.get("f1_macro"),
        "val_accuracy": val_metrics.get("accuracy"),
        "compiled_program_dir": str(program_dir),
    }
    save_run_manifest(run_dir, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
