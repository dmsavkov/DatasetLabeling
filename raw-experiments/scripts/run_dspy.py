# pyright: basic

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Literal

import dspy
import pandas as pd

from src.data import evaluate_predictions, normalize_label, now_stamp, save_json
from src.dataset_scripts import load_prosocial_dialog_bundle, make_dspy_sample_splits
from src.model import configure_dspy_ollama_manual

SEED = 42
OLLAMA_MODEL = "phi3:3.8b"
OPT_NUM_THREADS = 30
PREDICT_CONCURRENCY = 30


class SafetyLabelSignature(dspy.Signature):
    """Classify the user context into one ProsocialDialog safety label."""

    context = dspy.InputField(desc="User utterance to classify")
    thought = dspy.OutputField(desc="Step-by-step reasoning. 3 bullets total. Brief.")
    label: Literal["casual", "possibly_needs_caution", "probably_needs_caution", "needs_caution", "needs_intervention"] = dspy.OutputField()


class SafetyClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought(SafetyLabelSignature)

    def forward(self, context: str):
        return self.cot(context=context)


def dspy_metric(example, pred, trace=None):
    return float(normalize_label(example.label) == normalize_label(getattr(pred, "label", "")))


async def predict_with_dspy(program, contexts: pd.Series, *, concurrency: int = PREDICT_CONCURRENCY) -> list[str]:
    if len(contexts) == 0:
        return []

    semaphore = asyncio.Semaphore(min(concurrency, len(contexts)))

    async def predict_one(text: str) -> str:
        async with semaphore:
            result = await asyncio.to_thread(program, context=text)
            return normalize_label(getattr(result, "label", ""))

    return await asyncio.gather(*(predict_one(str(text)) for text in contexts))


def main() -> None:
    bundle = load_prosocial_dialog_bundle()
    splits = make_dspy_sample_splits(bundle["test_df"], seed=SEED, sample_size=50, train_size=25)

    configure_dspy_ollama_manual(model=OLLAMA_MODEL)

    dspy_train_df = splits["dspy_train_df"]
    dspy_test_df = splits["dspy_test_df"]
    label_order = bundle["label_order"]
    used_indexes = {
        "sample_50": [int(v) for v in splits["sample_df"]["source_index"].tolist()],
        "train_25": [int(v) for v in dspy_train_df["source_index"].tolist()],
        "test_25": [int(v) for v in dspy_test_df["source_index"].tolist()],
    }

    dspy_trainset = [
        dspy.Example(context=row[0], label=row[1]).with_inputs("context")
        for row in dspy_train_df[["context", "safety_label"]].itertuples(index=False, name=None)
    ]

    student = SafetyClassifier()
    optimizer = dspy.BootstrapFewShot(metric=dspy_metric, max_bootstrapped_demos=4, max_labeled_demos=8)

    # Optimizer parameter may not be available. 
    with dspy.context(num_threads=OPT_NUM_THREADS):
        t0 = perf_counter()
        optimized_program = optimizer.compile(student=student, trainset=dspy_trainset)
        compile_seconds = perf_counter() - t0

    t1 = perf_counter()
    test_contexts = pd.Series(dspy_test_df["context"], dtype="string")
    pred_labels = asyncio.run(predict_with_dspy(optimized_program, test_contexts))
    infer_seconds = perf_counter() - t1

    metrics = evaluate_predictions(pd.Series(dspy_test_df["safety_label"], dtype="string"), pd.Series(pred_labels), label_order)
    stamp = now_stamp()

    results_dir = bundle["results_dir"]
    pred_path = results_dir / f"mvp_dspy_preds_25test_{stamp}.csv"
    summary_path = results_dir / f"mvp_dspy_summary_{stamp}.json"

    pd.DataFrame(
        {
            "source_index": dspy_test_df["source_index"],
            "context": dspy_test_df["context"],
            "true_label": dspy_test_df["safety_label"],
            "pred_label": pred_labels,
        }
    ).to_csv(pred_path, index=False)

    save_json(
        summary_path,
        {
            "workflow": "dspy_phi3_fewshot_cot",
            "model": OLLAMA_MODEL,
            "sample_sizes": {"sample_50": 50, "train": 25, "test": 25},
            "timing_seconds": {
                "compile": float(compile_seconds),
                "inference": float(infer_seconds),
                "total": float(compile_seconds + infer_seconds),
            },
            "parallelism": {
                "optimization_num_threads": OPT_NUM_THREADS,
                "prediction_concurrency": PREDICT_CONCURRENCY,
            },
            "metrics": metrics,
            "used_prompt": {
                "reasoning": "ChainOfThought",
                "optimizer": "BootstrapFewShot",
            },
            "used_indexes": used_indexes,
            "artifacts": {
                "predictions_csv": str(pred_path),
            },
        },
    )

    print({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "summary": str(summary_path), "train_25_idx": used_indexes["train_25"], "test_25_idx": used_indexes["test_25"]})


if __name__ == "__main__":
    main()