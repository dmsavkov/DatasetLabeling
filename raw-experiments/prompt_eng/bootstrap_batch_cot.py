"""
Experiment: CoT + few-shot + BootstrapFewShot (batched 5 sentences/request).

- Default model: gemini-3.1-flash-lite-preview (unless overridden).
- Batch prompt: single request multiple labels. Default is 5.
- Prompt always instructs LLM about available labels.

*Extracted from untitle.py "Cot, fewshot, BootstrapFewShot" and simplified for experimental scripting.*

Defaults:
- Eval subset: 50 stratified samples, min per class ≥ 8
- Executor LM default: gemma-4-31b-it (can override via environment)
- Preliminary: run ONE forward pass + JSON parse assertion (no saving)
"""

from __future__ import annotations

import json
import os
import re
import time
import ast

import dspy
from dspy.teleprompt import BootstrapFewShot
from loguru import logger
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

import prompt_eng_common as pec

EXPERIMENT_SLUG = "bootstrap_batch_cot"

# -- EXPERIMENTATION RULES PREFERENCE --
# Follows rules from .cursor/rules/experimentation-rules.mdc

DEFAULT_MODEL = "gemma-4-31b-it"
EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL", DEFAULT_MODEL)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
EVAL_N = int(os.getenv("EVAL_N", "50"))
TRAIN_N = int(os.getenv("TRAIN_N", "50"))
MIN_PER_CLASS = int(os.getenv("MIN_PER_CLASS", "8"))

SKIP_PRELIMINARY = pec.env_bool("SKIP_PRELIMINARY", False)
PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)

LABELS = ["background", "objective", "methods", "results", "conclusions"]

LABELS_STR = ", ".join(f'"{l}"' for l in LABELS)  # for instruction

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
    if len(ok_labels) < len(LABELS):
        logger.warning("Some labels have <{} rows in pool: {}", min_per_class, counts.to_dict())
    sub = df[df[label_col].isin(ok_labels)].copy()
    per = min_per_class
    base = []
    for lab in LABELS:
        lab_df = sub[sub[label_col] == lab]
        if len(lab_df) >= per:
            base.append(lab_df.sample(n=per, random_state=seed))
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
        # <<--- BATCH PROMPT: includes instruction with available labels --->
        prompt = (
            f"Classify the following {BATCH_SIZE} sentences into one of these labels per sentence: {LABELS_STR}.\n"
            "Return the labels for the sentences in a JSON list, e.g.: "
            + json.dumps([LABELS[0]] * BATCH_SIZE) + ".\n"
            "Sentences:\n"
            + "\n".join([f"{idx+1}. {txt}" for idx, txt in enumerate(batch_texts)])
        )
        examples.append(
            dspy.Example(texts=prompt, target_labels=json.dumps(batch_labels)).with_inputs("texts")
        )
    return examples

class PubMedBatchSignature(dspy.Signature):
    """Classify a list of 5 medical sentences. Output MUST be a valid JSON list of 5 strings.
    Allowed labels: background, objective, methods, results, conclusions."""
    texts = dspy.InputField(desc=f'A numbered list of {BATCH_SIZE} sentences, with prompt to classify into one of: {LABELS_STR}.')
    predicted_labels = dspy.OutputField(desc="JSON list of labels, one for each input sentence.")

class PubMedBatchCoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(PubMedBatchSignature)

    def forward(self, texts):
        return self.predictor(texts=texts)

def extract_json_labels(prediction) -> list[str] | None:
    try:
        raw_val = prediction["predicted_labels"]
        if isinstance(raw_val, list):
            labels = [_normalize_label(str(p)) for p in raw_val]
            if len(labels) == BATCH_SIZE and all(l in LABELS for l in labels):
                return labels
        raw_s = str(raw_val)
        match = re.search(r"\[[\s\S]*\]", raw_s)
        blob = match.group(0) if match else raw_s
        data = None
        try:
            data = json.loads(blob)
        except Exception:
            try:
                data = ast.literal_eval(blob)
            except Exception:
                data = None
        if isinstance(data, list):
            labels = [_normalize_label(str(p)) for p in data]
            if len(labels) == BATCH_SIZE and all(l in LABELS for l in labels):
                return labels
    except Exception:
        pass
    return None

def validate_batch(example, pred, trace=None):
    target = json.loads(example["target_labels"])
    predicted = extract_json_labels(pred)
    return target == predicted if predicted else False

def main() -> None:
    # Hypothesis: Batch CoT + explicit label instruction enables reliable classification over few-shot bootstrap, across {BATCH_SIZE} sentences/request.
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    eval_df, _, train_df, _ = pec.load_hf_pubmed_splits()
    train_df = train_df.copy()
    train_df["label_name"] = train_df["label"].apply(pec.label_name_from_value)

    eval_subset = stratified_sample(eval_df, EVAL_N, label_col="label_name", min_per_class=MIN_PER_CLASS, seed=pec.DEFAULT_SEED)
    train_subset = stratified_sample(train_df, TRAIN_N, label_col="label_name", min_per_class=MIN_PER_CLASS, seed=pec.DEFAULT_SEED)

    train_examples = create_batch_examples(train_subset)
    eval_examples = create_batch_examples(eval_subset)
    if not train_examples or not eval_examples:
        raise RuntimeError("Not enough full batches of 5 in train/eval subsets.")

    lm = dspy.LM(
        model=f"openai/{EXECUTOR_MODEL}",
        api_base=pec.GOOGLE_OPENAI_BASE_URL,
        api_key=pec.GOOGLE_API_KEY,
        max_tokens=None,
        num_retries=pec.MAIN_MAX_RETRIES,
    )
    dspy.configure(lm=lm)

    # --- Preliminary: one forward pass + parse ---
    if not SKIP_PRELIMINARY:
        logger.info("=== PRELIMINARY (print only): one batch forward ===")
        ex0 = eval_examples[0]
        res0 = PubMedBatchCoT()(texts=ex0.texts)
        parsed0 = extract_json_labels(res0)
        logger.info("Parsed labels: {}", parsed0)
        if parsed0 is None:
            raw = str(getattr(res0, "predicted_labels", ""))[:1200]
            logger.error("Raw predicted_labels (trunc): {}", raw)
        assert parsed0 is not None and len(parsed0) == BATCH_SIZE, "Prelim parse failed"
        if PRELIM_ONLY:
            logger.warning("PRELIM_ONLY=1 — stop after preliminary.")
            return

    # --- Full: bootstrap compile + evaluate ---
    t0 = time.perf_counter()
    teleprompter = BootstrapFewShot(metric=validate_batch, max_bootstrapped_demos=2)
    logger.info("Compiling BootstrapFewShot (train batches={})...", len(train_examples))
    optimized = teleprompter.compile(PubMedBatchCoT(), trainset=train_examples)

    all_preds: list[str] = []
    all_true: list[str] = []
    for ex in tqdm(eval_examples, desc="eval_batches"):
        targets = json.loads(ex["target_labels"])
        try:
            res = optimized(texts=ex.texts)
            p = extract_json_labels(res)
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
            "EVAL_N": EVAL_N,
            "TRAIN_N": TRAIN_N,
            "BATCH_SIZE": BATCH_SIZE,
            "MIN_PER_CLASS": MIN_PER_CLASS,
            "max_bootstrapped_demos": 2,
        },
        predictions=[
            {"i": i, "true": t, "pred": p}
            for i, (t, p) in enumerate(zip(all_true, all_preds))
        ],
        duration_seconds=dt,
        notes="BootstrapFewShot compile over batched 5-sentence CoT classifier, batch prompt includes allowed labels.",
    )

    print(cr_t)
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()
