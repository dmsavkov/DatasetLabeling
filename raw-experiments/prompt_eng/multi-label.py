"""
Experiment: Single-Pass Multi-Label Prediction for Rhetorical Roles

Goal:
-----
Efficiently predict, for each biomedical sentence, *all* compatible rhetorical roles (from: background, objective, methods, results, conclusions) in a **single call** to the LLM, rather than five separate verification calls. Each sentence may receive zero or more labels, reflecting true multi-label classification.

This approach is more realistic, faster, and better matches real usage for practitioners — directly asking "which of these labels (if any) apply?" in one LM request.

Risks/Assumptions:
------------------
- The LLM may struggle to return exactly the relevant labels in the required format; careful prompt design is key.
- Evaluation must compare lists/sets of predicted vs. gold labels (not single-class accuracy).

Experiment Design:
------------------
- For each text, make **one LLM call** that provides the *set* of all applicable labels.
- Process N rows (FULL_N, default 20) using the DEFAULT_MODEL (gemini-3.1-flash-lite-preview per experimentation-rules.mdc).
- Evaluate "exact match" (all labels correct), micro/macro F1, and save confusion/error cases.

Implementation:
---------------
- Uses DSPy LLM signature for multi-label prediction.
- Results and artifacts are saved as before.
- Provides explicit list of available labels in the prompt.

"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy
from loguru import logger
from sklearn.metrics import classification_report, f1_score
from tqdm.auto import tqdm

import prompt_eng_common as pec

# Optionally, you could use Pydantic to enforce the output schema.
# For demonstration, explicit label inclusion in the prompt is shown below.

EXPERIMENT_SLUG = "multi_label_joint"
FULL_N = 40
SKIP_PRELIMINARY = pec.env_bool("SKIP_PRELIMINARY", False)
PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)
DEFAULT_MODEL = pec.DEFAULT_MODEL  # Should resolve to gemini-3.1-flash-lite-preview by experimentation-rules.mdc

labels_to_check = ["background", "objective", "methods", "results", "conclusions"]

# Construct a verbose prompt which is passed to the model, listing the available label choices.
MULTILABEL_SYSTEM_PROMPT = (
    "You are a biomedical NLP system that identifies all rhetorical roles expressed in a given sentence. "
    "Available roles (labels) are: background, objective, methods, results, conclusions." "\n"
    "Given the input sentence, return ALL applicable labels as a comma-separated list from the above options. "
    "If none of the labels apply, return 'none'. Do not return any labels other than those listed."
    "\n\n"
    "Roles:\n"
    " - background\n"
    " - objective\n"
    " - methods\n"
    " - results\n"
    " - conclusions\n"
)


class MultiLabelPredictionSignature(dspy.Signature):
    """Predict which (if any) rhetorical roles apply to a biomedical sentence.

    Output: Comma-separated list of roles, or 'none'. Only allowed labels: background, objective, methods, results, conclusions.
    """
    text = dspy.InputField(desc="The medical sentence to evaluate.")
    system_prompt = dspy.InputField(desc="Explicit list of all available rhetorical role labels and instructions.")
    compatible_labels = dspy.OutputField(
        desc="Comma-separated list of all compatible labels (from the provided label set) that apply to the sentence. If none, say 'none'."
    )


def predict_labels(text: str) -> list[str]:
    predictor = dspy.Predict(MultiLabelPredictionSignature)
    logger.debug("predict_labels text_snip={}...", text[:80])
    res = predictor(
        text=text,
        system_prompt=MULTILABEL_SYSTEM_PROMPT,
    )
    val = str(res.compatible_labels or "").strip().lower()
    if val == "none" or val == "":
        return []
    # Support robust parsing
    out = [lbl.strip().lower() for lbl in val.replace(" and ", ",").split(",") if lbl.strip()]
    # Only those in the expected set
    valid_out = sorted(set(l for l in out if l in labels_to_check))
    logger.debug("predicted labels: {}", valid_out)
    return valid_out


def process_text(idx: int, text: str):
    labels = predict_labels(text)
    return idx, labels


def run_eval(eval_texts: list[str], gold_labels: list[list[str]], *, max_workers: int, lm: dspy.LM):
    dspy.configure(lm=lm)
    logger.info("Evaluating {} texts (workers={})...", len(eval_texts), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_text, i, txt): i for i, txt in enumerate(eval_texts)}
        all_results: list = [None] * len(eval_texts)
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, pred_labels = future.result()
            all_results[idx] = pred_labels

    # Exact set match per row
    exact_matches = [set(pred) == set(true) for pred, true in zip(all_results, gold_labels)]
    acc_exact = sum(exact_matches) / len(exact_matches)

    # Flatten for micro/macro F1
    all_true = [[lbl.lower()] if isinstance(lbl, str) else [l.lower() for l in lbl] for lbl in gold_labels]
    all_pred = [lbls for lbls in all_results]

    # Binarize for sklearn
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=labels_to_check)
    y_true_bin = mlb.fit_transform(all_true)
    y_pred_bin = mlb.transform(all_pred)

    cr = classification_report(y_true_bin, y_pred_bin, target_names=labels_to_check, output_dict=True, zero_division=0)
    cr_text = classification_report(y_true_bin, y_pred_bin, target_names=labels_to_check, zero_division=0)
    micro_f1 = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    confusion_cases = []
    for i, (pred, true) in enumerate(zip(all_pred, all_true)):
        if set(pred) != set(true):
            confusion_cases.append({
                "text": eval_texts[i],
                "true_labels": true,
                "predicted_labels": pred,
            })

    return {
        "exact_match_accuracy": float(acc_exact),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "n_texts": len(eval_texts),
        "n_mismatched": len(confusion_cases),
        "confusion_cases_sample": confusion_cases[:10],
        "classification_report": cr,
        "classification_report_text": cr_text,
        "predictions_pairs": [
            {"pred": p, "true": t, "text": tx[:300]} for p, t, tx in zip(all_pred, all_true, eval_texts)
        ],
    }


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    eval_df, _, _, _ = pec.load_hf_pubmed_splits()
    # We assume the 'label_name' column now may contain multiple (comma-separated) labels per row for multilabel ground truth
    # If not, convert to list for compatibility
    def parse_labels(val):
        if isinstance(val, str):
            # Accept comma-separated or single-string
            parts = [v.strip().lower() for v in val.split(",") if v.strip()]
            return [l for l in parts if l in labels_to_check]
        if isinstance(val, list):
            return [v.lower() for v in val if v in labels_to_check]
        return []

    # --- Use DEFAULT_MODEL ("gemini-3.1-flash-lite-preview") everywhere for model consistency
    lm = dspy.LM(
        model=f"openai/{DEFAULT_MODEL}",
        api_base=pec.GOOGLE_OPENAI_BASE_URL,
        api_key=pec.GOOGLE_API_KEY,
        max_tokens=4000,
        num_retries=pec.MAIN_MAX_RETRIES,
    )

    # ----- Preliminary: single text -----
    if not SKIP_PRELIMINARY:
        logger.info("=== PRELIMINARY: one text (single multi-label call) ===")
        pre_text = eval_df["text"].iloc[0]
        pre_true = parse_labels(eval_df["label_name"].iloc[0])
        t0 = time.perf_counter()
        dspy.configure(lm=lm)
        pred_pre = predict_labels(pre_text)
        dt = time.perf_counter() - t0
        logger.info("Prelim predicted labels: {}", pred_pre)
        logger.info("Prelim true label(s): {}", pre_true)
        pec.save_phase(
            run_dir,
            "preliminary",
            metrics={
                "n_labels_available": len(labels_to_check),
                "predicted_labels": pred_pre,
                "true_labels": pre_true,
                "exact_match": set(pred_pre) == set(pre_true),
            },
            settings={"model": f"openai/{DEFAULT_MODEL}", "num_retries": pec.MAIN_MAX_RETRIES},
            predictions=[{"text": pre_text[:400], "predicted_labels": pred_pre, "true_labels": pre_true}],
            duration_seconds=dt,
            notes="Single sentence; single multi-label prediction call.",
        )
        if PRELIM_ONLY:
            logger.warning("PRELIM_ONLY=1 — skip full run.")
            return

    # ----- Full -----
    logger.info("=== FULL: {} texts (single multi-label call per sentence) ===", FULL_N)
    subset = eval_df.head(FULL_N)
    eval_texts = subset["text"].tolist()
    true_labels_all = [parse_labels(l) for l in subset["label_name"]]
    t1 = time.perf_counter()
    metrics = run_eval(eval_texts, true_labels_all, max_workers=2, lm=lm)
    dt_full = time.perf_counter() - t1

    logger.info(
        "Full exact match accuracy: {:.2%} | confusion cases: {} | micro F1: {:.3f} | macro F1: {:.3f}",
        metrics["exact_match_accuracy"],
        metrics["n_mismatched"],
        metrics["micro_f1"],
        metrics["macro_f1"],
    )

    pec.save_phase(
        run_dir,
        "full",
        metrics={k: v for k, v in metrics.items() if k != "predictions_pairs"},
        settings={
            "model": f"openai/{DEFAULT_MODEL}",
            "FULL_N": FULL_N,
            "num_retries": pec.MAIN_MAX_RETRIES,
            "multi_label_joint": True,
        },
        predictions=metrics["predictions_pairs"],
        duration_seconds=dt_full,
        notes="Parallel per-text; each text runs a single LM call for joint multi-label prediction.",
    )
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()
