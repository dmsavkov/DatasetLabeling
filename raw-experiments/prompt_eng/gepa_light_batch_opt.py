"""
Experiment: GEPA (light) optimization for batched 5-sentence classifier.

Source: extracted from untitle.py "GEPA", simplified + updated:
- Trainset: 50 stratified samples (min 8 per class) -> 10 batches of 5
- Eval set: 50 stratified samples (min 8 per class) -> 10 batches of 5
- Executor LM: gemma-4-31b-it
- Reflector LM: gemini-3.1-flash-lite-preview

Preliminary: run one forward pass + parse assertion (no saving) before GEPA compile.
"""

from __future__ import annotations

import json
import os
import re
import time
import ast

import dspy
from dspy.teleprompt import GEPA
from loguru import logger
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

import prompt_eng_common as pec

EXPERIMENT_SLUG = "gepa_light_batch_opt"

EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL", "gemma-4-31b-it")
REFLECTOR_MODEL = os.getenv("REFLECTOR_MODEL", "gemini-3.1-flash-lite-preview")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
TRAIN_N = int(os.getenv("TRAIN_N", "50"))
EVAL_N = int(os.getenv("EVAL_N", "50"))
MIN_PER_CLASS = int(os.getenv("MIN_PER_CLASS", "8"))

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


def make_trainset(df):
    trainset = []
    texts_list = df["text"].astype(str).tolist()
    labels_list = df["label_name"].astype(str).str.lower().tolist()
    for i in range(0, len(texts_list), BATCH_SIZE):
        batch_texts = texts_list[i : i + BATCH_SIZE]
        batch_labels = labels_list[i : i + BATCH_SIZE]
        if len(batch_texts) != BATCH_SIZE:
            continue
        formatted = "\n".join([f"{idx+1}. {txt}" for idx, txt in enumerate(batch_texts)])
        trainset.append(
            dspy.Example(input_texts=formatted, target_labels=batch_labels).with_inputs("input_texts")
        )
    return trainset


class PubMedBatchSignature(dspy.Signature):
    """Classify a list of 5 medical sentences into rhetorical roles."""

    input_texts = dspy.InputField(desc="Numbered list of 5 sentences.")
    predicted_labels = dspy.OutputField(
        desc="Strict JSON list of exactly 5 labels, e.g., [\"background\", \"objective\", \"methods\", \"results\", \"conclusions\"]."
    )


class PubMedBatchClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(PubMedBatchSignature)

    def forward(self, input_texts):
        return self.predictor(input_texts=input_texts)


def _parse_predicted_labels(pred) -> list[str]:
    raw_attr = getattr(pred, "predicted_labels", None)
    if isinstance(raw_attr, list):
        labels = [_normalize_label(str(l)) for l in raw_attr]
        if len(labels) == BATCH_SIZE:
            return [l if l in LABELS else "error" for l in labels]
    raw_output = str(raw_attr if raw_attr is not None else "")
    json_match = re.search(r"\[[\s\S]*\]", raw_output)
    blob = json_match.group(0) if json_match else raw_output
    data = None
    try:
        data = json.loads(blob)
    except Exception:
        try:
            data = ast.literal_eval(blob)
        except Exception:
            data = None
    if isinstance(data, list):
        labels = [_normalize_label(str(l)) for l in data]
    else:
        labels = ["error"] * BATCH_SIZE
    if len(labels) != BATCH_SIZE:
        labels = (labels + ["error"] * BATCH_SIZE)[:BATCH_SIZE]
    labels = [l if l in LABELS else "error" for l in labels]
    return labels


def gepa_batch_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    gold_labels = [str(l).lower().strip() for l in gold.target_labels]
    pred_labels = _parse_predicted_labels(pred)
    correct = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)
    score = correct / float(BATCH_SIZE)
    if score == 1.0:
        feedback = "Perfect classification."
    else:
        errors = [
            f"Pos {i+1}: Expected {gold_labels[i]}, got {pred_labels[i]}"
            for i in range(BATCH_SIZE)
            if gold_labels[i] != pred_labels[i]
        ]
        feedback = f"Partial ({correct}/{BATCH_SIZE}). Failures: " + "; ".join(errors)
    return dspy.Prediction(score=score, feedback=feedback)


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    eval_df, _, train_df, _ = pec.load_hf_pubmed_splits()
    train_df = train_df.copy()
    train_df["label_name"] = train_df["label"].apply(pec.label_name_from_value)

    eval_subset = stratified_sample(eval_df, EVAL_N, label_col="label_name", min_per_class=MIN_PER_CLASS, seed=pec.DEFAULT_SEED)
    train_subset = stratified_sample(train_df, TRAIN_N, label_col="label_name", min_per_class=MIN_PER_CLASS, seed=pec.DEFAULT_SEED)

    trainset = make_trainset(train_subset)
    evalset = make_trainset(eval_subset)
    if not trainset or not evalset:
        raise RuntimeError("Not enough full batches of 5 in train/eval subsets.")

    prediction_lm = dspy.LM(
        model=f"openai/{EXECUTOR_MODEL}",
        api_base=pec.GOOGLE_OPENAI_BASE_URL,
        api_key=pec.GOOGLE_API_KEY,
        max_tokens=None,
        num_retries=pec.MAIN_MAX_RETRIES,
    )
    reflection_lm = dspy.LM(
        model=f"openai/{REFLECTOR_MODEL}",
        api_base=pec.GOOGLE_OPENAI_BASE_URL,
        api_key=pec.GOOGLE_API_KEY,
        max_tokens=None,
        num_retries=pec.MAIN_MAX_RETRIES,
    )

    dspy.settings.configure(lm=prediction_lm, num_threads=15)

    # --- Preliminary: one forward pass + parse ---
    if not SKIP_PRELIMINARY:
        logger.info("=== PRELIMINARY (print only): one batch forward ===")
        ex0 = evalset[0]
        res0 = PubMedBatchClassifier()(input_texts=ex0.input_texts)
        p0 = _parse_predicted_labels(res0)
        logger.info("Parsed: {}", p0)
        if all(p == "error" for p in p0):
            raw = str(getattr(res0, "predicted_labels", ""))[:1200]
            logger.error("Raw predicted_labels (trunc): {}", raw)
        assert len(p0) == BATCH_SIZE and any(p != "error" for p in p0), "Prelim parse failed"
        if PRELIM_ONLY:
            logger.warning("PRELIM_ONLY=1 — stop after preliminary.")
            return

    optimizer = GEPA(
        metric=gepa_batch_metric,
        reflection_lm=reflection_lm,
        auto="light",
        reflection_minibatch_size=2,
        skip_perfect_score=True,
        candidate_selection_strategy="pareto",
    )

    logger.info("Compiling GEPA (light). Train batches={}", len(trainset))
    t0 = time.perf_counter()
    optimized_program = optimizer.compile(PubMedBatchClassifier(), trainset=trainset)

    # Persist optimization artifacts for post-hoc inspection (analyze_gepa_light_batch_opt.py).
    try:
        state_path = run_dir / "optimized_program_state.json"
        state_path.write_text(json.dumps(optimized_program.dump_state(), indent=2, default=str), encoding="utf-8")
        logger.info("Wrote {}", state_path.name)
        optimized_program.save(str(run_dir / "optimized_program"), save_program=True)
        logger.info("Wrote optimized_program/ (DSPy save bundle)")
    except Exception as e:
        logger.warning("Could not save optimized program state: {}", e)

    def _batches_explicit(df, split: str) -> list[dict]:
        rows = []
        texts = df["text"].astype(str).tolist()
        labels = df["label_name"].astype(str).str.lower().tolist()
        for bi in range(0, len(texts), BATCH_SIZE):
            bt = texts[bi : bi + BATCH_SIZE]
            bl = labels[bi : bi + BATCH_SIZE]
            if len(bt) != BATCH_SIZE:
                continue
            rows.append(
                {
                    "split": split,
                    "batch_id": bi // BATCH_SIZE,
                    "input_texts_numbered": "\n".join(f"{j+1}. {t}" for j, t in enumerate(bt)),
                    "target_labels": bl,
                    "sentences": [{"position": j + 1, "text": t, "gold": lb} for j, (t, lb) in enumerate(zip(bt, bl))],
                }
            )
        return rows

    (run_dir / "train_batches_explicit.json").write_text(
        json.dumps(_batches_explicit(train_subset, "train"), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "eval_batches_explicit.json").write_text(
        json.dumps(_batches_explicit(eval_subset, "eval"), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    train_subset.assign(sentence_idx=range(len(train_subset))).to_csv(run_dir / "train_sentences_flat.csv", index=False)
    eval_subset.assign(sentence_idx=range(len(eval_subset))).to_csv(run_dir / "eval_sentences_flat.csv", index=False)

    all_preds: list[str] = []
    all_true: list[str] = []
    logger.info("Evaluating optimized program on eval batches={}", len(evalset))
    for ex in tqdm(evalset, desc="eval_batches"):
        gold = [str(l).lower().strip() for l in ex.target_labels]
        try:
            res = optimized_program(input_texts=ex.input_texts)
            pred_labels = _parse_predicted_labels(res)
        except Exception:
            pred_labels = ["error"] * BATCH_SIZE
        all_true.extend(gold)
        all_preds.extend(pred_labels)

    dt = time.perf_counter() - t0
    acc = float(accuracy_score(all_true, all_preds))
    cr_d, cr_t = pec.sklearn_classification_reports(all_true, all_preds)

    pec.save_phase(
        run_dir,
        "full",
        metrics={
            "accuracy": acc,
            "n_samples": len(all_true),
            "n_batches": len(evalset),
            "classification_report": cr_d,
            "classification_report_text": cr_t,
        },
        settings={
            "EXECUTOR_MODEL": EXECUTOR_MODEL,
            "REFLECTOR_MODEL": REFLECTOR_MODEL,
            "TRAIN_N": TRAIN_N,
            "EVAL_N": EVAL_N,
            "BATCH_SIZE": BATCH_SIZE,
            "MIN_PER_CLASS": MIN_PER_CLASS,
            "auto": "light",
        },
        predictions=[
            {"i": i, "true": t, "pred": p}
            for i, (t, p) in enumerate(zip(all_true, all_preds))
        ],
        duration_seconds=dt,
        notes="GEPA light compile for batched 5-sentence CoT classifier (gemma executor, gemini reflector).",
    )

    print(cr_t)
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()

