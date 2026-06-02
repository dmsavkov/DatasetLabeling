"""
Experiment: batched 5-sentence classification using Chain-of-Thought with few-shot demos.

Defaults:
- Train demos: 50 stratified samples (min 8 per class); use first N_DEMOS batches as demos
- Eval subset: 40 stratified samples (min 8 per class)
- Executor LM: gemma-4-31b-it
- Preliminary: one batch forward + parse assertion (no saving)
"""

from __future__ import annotations

import ast
import json
import os
import re
import time

import dspy
from loguru import logger
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

import prompt_eng_common as pec

EXPERIMENT_SLUG = "fewshot_cot_batch"

EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL", "gemma-4-31b-it")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
TRAIN_N = int(os.getenv("TRAIN_N", "50"))
EVAL_N = int(os.getenv("EVAL_N", "40"))
MIN_PER_CLASS = int(os.getenv("MIN_PER_CLASS", "8"))
N_DEMOS = int(os.getenv("N_DEMOS", "10"))

SKIP_PRELIMINARY = pec.env_bool("SKIP_PRELIMINARY", False)
PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)

LABELS = ["background", "objective", "methods", "results", "conclusions"]


def _normalize_label(s: str) -> str:
    v = (s or "").strip().lower()
    if v in ("conclusion", "concl", "concl.", "conclusive"):
        return "conclusions"
    if v == "method":
        return "methods"
    if v == "result":
        return "results"
    if v == "objective(s)":
        return "objective"
    return v


def stratified_sample(df, n: int, *, label_col: str, min_per_class: int, seed: int) -> "pd.DataFrame":
    import pandas as pd

    counts = df[label_col].value_counts()
    ok_labels = [k for k, v in counts.items() if v >= min_per_class]
    sub = df[df[label_col].isin(ok_labels)].copy()
    base = []
    for lab in LABELS:
        lab_df = sub[sub[label_col] == lab]
        if len(lab_df) >= min_per_class:
            base.append(lab_df.sample(n=min_per_class, random_state=seed))
    out = pd.concat(base, ignore_index=True) if base else sub.sample(n=min(n, len(sub)), random_state=seed)
    remaining = n - len(out)
    if remaining > 0:
        rest = sub.drop(out.index, errors="ignore")
        if len(rest) > 0:
            out = pd.concat([out, rest.sample(n=min(remaining, len(rest)), random_state=seed)], ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def create_batch_examples(df):
    examples = []
    texts = df["text"].astype(str).tolist()
    labels = df["label_name"].astype(str).str.lower().tolist()
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_labels = labels[i : i + BATCH_SIZE]
        if len(batch_texts) != BATCH_SIZE:
            continue
        formatted = "\n".join([f"{idx+1}. {txt}" for idx, txt in enumerate(batch_texts)])
        examples.append(
            dspy.Example(input_texts=formatted, target_labels=json.dumps(batch_labels)).with_inputs("input_texts")
        )
    return examples


class BatchCoTSignature(dspy.Signature):
    """
    Classify a list of 5 sentences. Chain-of-thought required.
    Return a strict JSON list of 5 labels, each matched to a sentence in order.
    """
    input_texts = dspy.InputField(desc="A numbered list of 5 sentences.")
    predicted_labels = dspy.OutputField(desc="Strict JSON list containing exactly 5 section labels, one per sentence. Only these: background, objective, methods, results, conclusions. Include your reasoning as intermediate steps and then finish with just the JSON list in the last line.")


class BatchCoTFewShotModule(dspy.Module):
    def __init__(self, demos=None):
        super().__init__()
        self.cot = dspy.ChainOfThought(BatchCoTSignature)
        # Attach demos if provided
        if demos is not None:
            self.cot.demos = demos

    def forward(self, input_texts):
        return self.cot(input_texts=input_texts)


def parse_labels(pred) -> list[str] | None:
    # Attempts to extract list of labels from cot output
    raw_attr = getattr(pred, "predicted_labels", None)
    if isinstance(raw_attr, list):
        labels = [_normalize_label(str(x)) for x in raw_attr]
        if len(labels) == BATCH_SIZE and all(l in LABELS for l in labels):
            return labels
    raw = str(raw_attr if raw_attr is not None else "")
    m = re.search(r"\[[\s\S]*\]", raw)
    blob = m.group(0) if m else raw
    data = None
    try:
        data = json.loads(blob)
    except Exception:
        try:
            data = ast.literal_eval(blob)
        except Exception:
            data = None
    if not isinstance(data, list):
        return None
    labels = [_normalize_label(str(x)) for x in data]
    if len(labels) != BATCH_SIZE or not all(l in LABELS for l in labels):
        return None
    return labels


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    eval_df, _, train_df, _ = pec.load_hf_pubmed_splits()
    train_df = train_df.copy()
    train_df["label_name"] = train_df["label"].apply(pec.label_name_from_value)

    eval_subset = stratified_sample(eval_df, EVAL_N, label_col="label_name", min_per_class=MIN_PER_CLASS, seed=pec.DEFAULT_SEED)
    train_subset = stratified_sample(train_df, TRAIN_N, label_col="label_name", min_per_class=MIN_PER_CLASS, seed=pec.DEFAULT_SEED)

    demos_all = create_batch_examples(train_subset)
    eval_examples = create_batch_examples(eval_subset)
    if not demos_all or not eval_examples:
        raise RuntimeError("Not enough full batches of 5 in train/eval subsets.")

    lm = dspy.LM(
        model=f"openai/{EXECUTOR_MODEL}",
        api_base=pec.GOOGLE_OPENAI_BASE_URL,
        api_key=pec.GOOGLE_API_KEY,
        max_tokens=None,
        num_retries=pec.MAIN_MAX_RETRIES,
    )
    dspy.configure(lm=lm)

    # Attach up to N_DEMOS batch demos, with cot finalized as JSON
    keep = demos_all[: max(0, min(N_DEMOS, len(demos_all)))]
    demo_examples = [
        dspy.Example(
            input_texts=ex.input_texts,
            predicted_labels=ex["target_labels"]  # ground truth as strict JSON list
        ).with_inputs("input_texts")
        for ex in keep
    ]
    logger.info("Using N_DEMOS={} demo batches (of {}).", len(demo_examples), len(demos_all))

    prog = BatchCoTFewShotModule(demos=demo_examples if demo_examples else None)

    if not SKIP_PRELIMINARY:
        logger.info("=== PRELIMINARY (print only): one batch forward ===")
        ex0 = eval_examples[0]
        res0 = prog(input_texts=ex0.input_texts)
        p0 = parse_labels(res0)
        logger.info("Parsed: {}", p0)
        if p0 is None:
            logger.error("Raw predicted_labels (trunc): {}", str(getattr(res0, "predicted_labels", ""))[:1200])
        assert p0 is not None, "Prelim parse failed"
        if PRELIM_ONLY:
            logger.warning("PRELIM_ONLY=1 — stop after preliminary.")
            return

    t0 = time.perf_counter()
    all_true: list[str] = []
    all_preds: list[str] = []
    for ex in tqdm(eval_examples, desc="eval_batches"):
        targets = json.loads(ex["target_labels"])
        try:
            res = prog(input_texts=ex.input_texts)
            p = parse_labels(res)
            all_preds.extend(p if p else ["error"] * BATCH_SIZE)
        except Exception:
            all_preds.extend(["error"] * BATCH_SIZE)
        all_true.extend(targets)

    dt = time.perf_counter() - t0
    acc = float(accuracy_score(all_true, all_preds))
    cr_d, cr_t = pec.sklearn_classification_reports(all_true, all_preds)

    pec.save_phase(
        run_dir,
        "full",
        metrics={
            "accuracy": acc,
            "n_samples": len(all_true),
            "n_batches": len(eval_examples),
            "classification_report": cr_d,
            "classification_report_text": cr_t,
        },
        settings={
            "EXECUTOR_MODEL": EXECUTOR_MODEL,
            "TRAIN_N": TRAIN_N,
            "EVAL_N": EVAL_N,
            "BATCH_SIZE": BATCH_SIZE,
            "MIN_PER_CLASS": MIN_PER_CLASS,
            "N_DEMOS": len(demo_examples),
        },
        predictions=[{"i": i, "true": t, "pred": p} for i, (t, p) in enumerate(zip(all_true, all_preds))],
        duration_seconds=dt,
        notes="Chain-of-Thought with static few-shot demos for batched classification.",
    )

    print(cr_t)
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()
