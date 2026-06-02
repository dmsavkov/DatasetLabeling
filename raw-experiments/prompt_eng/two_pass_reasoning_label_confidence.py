# %% [markdown]
# ### Two pass: reasoning + label & confidence (forced verbalized JSON)
# Preliminary: PRELIM_N rows. Full: FULL_N rows. Plot saved under run dir.

# %%
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report as clf_report_str
from tqdm.auto import tqdm

import prompt_eng_common as pec

EXPERIMENT_SLUG = "two_pass_reasoning_label_confidence"
FULL_N = int(os.getenv("FULL_N", "50"))
PRELIM_N = int(os.getenv("PRELIM_N", "2"))
SKIP_PRELIMINARY = pec.env_bool("SKIP_PRELIMINARY", False)
PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)

VALID_LABELS = {"background", "objective", "methods", "results", "conclusions"}
LABEL_FORMATTED_LIST = sorted(list(VALID_LABELS))

# --- Slightly instructive prompt tweaks below ---

class ConfidenceReasoningSignature(dspy.Signature):
    """
    Analyze a medical sentence with respect to one of the following rhetorical roles:
    {labels}

    Provide a short explanation for which single label fits best, and why the other alternatives are less appropriate.
    """.format(labels=", ".join(f'"{lab}"' for lab in LABEL_FORMATTED_LIST))
    text = dspy.InputField(desc="The input sentence or fragment to analyze.")
    reasoning_over_alternatives = dspy.OutputField(
        desc=(
            "Explain your selection by describing why the single selected label fits best, "
            "and briefly note why the other possible labels are not as suitable. "
            f"Choose one label ONLY from: {', '.join(LABEL_FORMATTED_LIST)}."
        )
    )

class ForcedConfidenceVerdictSignature(dspy.Signature):
    """
    Based on the text below, and your reasoning, output your final answer in the following strict JSON format:
    {{\"predicted_label\": \"...\", \"confidence_score\": 0.XX}}
    Only choose the predicted_label from: {labels}
    """.format(labels=", ".join(f'"{lab}"' for lab in LABEL_FORMATTED_LIST))
    text = dspy.InputField()
    reasoning = dspy.InputField()
    verdict_json = dspy.OutputField(
        desc=(
            'Strict JSON object: {"predicted_label": "<one label from: ' +
            f"{', '.join(LABEL_FORMATTED_LIST)}" +
            '>", "confidence_score": 0.XX} (confidence 0 to 1.0)'
        )
    )

class ForcedElicitationClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.reasoner = dspy.Predict(ConfidenceReasoningSignature)
        self.verdict = dspy.Predict(ForcedConfidenceVerdictSignature)

    def forward(self, text):
        reasoning_res = self.reasoner(text=text)
        verdict_res = self.verdict(text=text, reasoning=reasoning_res.reasoning_over_alternatives)
        return verdict_res

def process_item(idx: int, text: str, true_label: str, classifier: ForcedElicitationClassifier, lm: dspy.LM):
    debug_info = {
        "idx": idx,
        "true_label_raw": true_label,
        "text_snippet": text[:120].replace('\n', ' ') + ("..." if len(text) > 120 else "")
    }
    try:
        logger.debug("Item {} pass1+2...", idx)
        with dspy.context(lm=lm):
            res = classifier(text=text)
        raw_json = str(getattr(res, "verdict_json", ""))
        debug_info["raw_json"] = raw_json

        if not raw_json:
            logger.warning(f"idx={idx} - verdict_json missing or empty: {res}")
        json_match = re.search(r"\{.*?\}", raw_json, re.DOTALL)
        debug_info["json_match"] = json_match.group(0) if json_match else None
        if json_match:
            try:
                data = json.loads(json_match.group(0))
            except Exception as e:
                logger.warning(f"idx={idx} - JSON decode error: {e}, raw: {json_match.group(0)}")
                data = {}
            pred = str(data.get("predicted_label", "error")).lower().strip()
            conf = data.get("confidence_score", 0.0)
            try:
                conf = float(conf)
            except Exception:
                logger.warning(f"idx={idx} - confidence_score not float: {conf}")
                conf = 0.0
        else:
            logger.warning(f"idx={idx} - verdict_json did not contain valid JSON object: {raw_json}")
            pred, conf = "error", 0.0
    except Exception as e:
        logger.warning("Item {} failed: {}", idx, e)
        debug_info["exception"] = str(e)
        pred, conf = "error", 0.0

    # Defensive normalization for prediction and true label
    pred_norm = str(pred).lower().strip()
    true_clean = str(true_label).lower().strip()

    if pred_norm not in VALID_LABELS and pred_norm != "error":
        logger.warning(f"idx={idx} - pred '{pred_norm}' not a valid label (VALID_LABELS: {VALID_LABELS})")
        pred_norm = "error"
    if true_clean not in VALID_LABELS:
        logger.warning(f"idx={idx} - true_label '{true_clean}' not a valid label (VALID_LABELS: {VALID_LABELS})")
    debug_info["pred_norm"] = pred_norm
    debug_info["true_norm"] = true_clean
    debug_info["conf"] = conf

    logger.debug(f"[process_item] Debug info: {json.dumps(debug_info)}")
    return {"idx": idx, "true": true_clean, "pred": pred_norm, "conf": conf, "debug": debug_info}

def run_subset(classifier, lm: dspy.LM, eval_subset: pd.DataFrame, *, workers: int, phase: str) -> tuple[pd.DataFrame, float]:
    results: list[dict] = []
    t0 = time.perf_counter()
    if not isinstance(eval_subset, pd.DataFrame):
        logger.error("Provided eval_subset is not a DataFrame. Got type: {}", type(eval_subset))
        raise TypeError(f"Expected eval_subset to be a DataFrame, got {type(eval_subset)}")
    required_cols = {"text", "label_name"}
    missing_cols = required_cols.difference(set(eval_subset.columns))
    if missing_cols:
        logger.error("eval_subset missing required columns: {}", missing_cols)
        raise ValueError(f"eval_subset missing columns: {missing_cols}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_item, i, row["text"], row["label_name"], classifier, lm)
            for i, (_, row) in enumerate(eval_subset.iterrows())
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=phase):
            try:
                item_result = future.result()
                results.append(item_result)
            except Exception as e:
                logger.error(f"[run_subset] Future failed: {e}")
    dt = time.perf_counter() - t0
    res_df = pd.DataFrame(results)
    # Robust equality, log mismatches for easy debugging
    def _row_is_correct(row):
        correct = str(row["true"]).strip().lower() == str(row["pred"]).strip().lower()
        if not correct:
            logger.debug(f"[is_correct-check] idx={row.get('idx')} - true='{row['true']}' pred='{row['pred']}' - mismatch")
        return correct
    res_df["is_correct"] = res_df.apply(_row_is_correct, axis=1)
    return res_df, dt

def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    eval_df, _, _, _ = pec.load_hf_pubmed_splits()

    # Defensive check on gold label column
    expected_gold_col = "label_name"
    if expected_gold_col not in eval_df.columns:
        logger.error(f"eval_df missing expected gold label column '{expected_gold_col}'. Columns: {list(eval_df.columns)}")
        raise RuntimeError(
            f"Expected eval_df to have gold label column '{expected_gold_col}', got columns: {eval_df.columns}"
        )

    lm_prelim = dspy.LM(
        model=f"openai/{pec.DEFAULT_MODEL}",
        api_base=pec.GOOGLE_OPENAI_BASE_URL,
        api_key=pec.GOOGLE_API_KEY,
        max_tokens=5000,
        num_retries=pec.PRELIM_MAX_RETRIES,
    )
    lm_full = dspy.LM(
        model=f"openai/{pec.DEFAULT_MODEL}",
        api_base=pec.GOOGLE_OPENAI_BASE_URL,
        api_key=pec.GOOGLE_API_KEY,
        max_tokens=5000,
        num_retries=pec.MAIN_MAX_RETRIES,
    )
    classifier = ForcedElicitationClassifier()

    if not SKIP_PRELIMINARY:
        logger.info("=== PRELIMINARY: {} rows ===", PRELIM_N)
        pre_df = eval_df.head(PRELIM_N)
        logger.debug(f"pre_df shape: {pre_df.shape}, columns: {list(pre_df.columns)}")
        res_pre, dt_pre = run_subset(classifier, lm_prelim, pre_df, workers=2, phase="prelim")
        logger.debug(f"res_pre shape: {res_pre.shape}, first rows: {res_pre.head(2).to_dict(orient='records')}")
        acc_pre = float(res_pre["is_correct"].mean())
        logger.info("Prelim accuracy {:.2%}", acc_pre)
        pre_cr_d, pre_cr_t = pec.sklearn_classification_reports(res_pre["true"], res_pre["pred"])

        pec.save_phase(
            run_dir,
            "preliminary",
            metrics={
                "accuracy": acc_pre,
                "mean_conf_correct": float(res_pre[res_pre["is_correct"]]["conf"].mean())
                if res_pre["is_correct"].any()
                else None,
                "mean_conf_wrong": float(res_pre[~res_pre["is_correct"]]["conf"].mean())
                if (~res_pre["is_correct"]).any()
                else None,
                "n_rows": len(res_pre),
                "classification_report": pre_cr_d,
                "classification_report_text": pre_cr_t,
            },
            settings={"model": pec.DEFAULT_MODEL, "PRELIM_N": PRELIM_N, "num_retries": pec.PRELIM_MAX_RETRIES},
            predictions=res_pre.to_dict(orient="records"),
            duration_seconds=dt_pre,
        )
        if PRELIM_ONLY:
            logger.warning("PRELIM_ONLY=1 — skip full.")
            return

    logger.info("=== FULL: {} rows ===", FULL_N)
    eval_subset = eval_df.head(FULL_N)
    logger.debug(f"eval_subset shape: {eval_subset.shape}, columns: {list(eval_subset.columns)}")
    res_df, dt_full = run_subset(classifier, lm_full, eval_subset, workers=2, phase="full")
    logger.debug(f"res_df shape: {res_df.shape}, first rows: {res_df.head(2).to_dict(orient='records')}")

    acc = float(res_df["is_correct"].mean())
    logger.info("Full accuracy {:.2%}", acc)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="is_correct", y="conf", data=res_df, palette="Set1")
    sns.stripplot(x="is_correct", y="conf", data=res_df, color="black", alpha=0.3, jitter=True)
    plt.title("Forced Elicitation: Confidence vs. Accuracy")
    plt.xlabel("Prediction Correctness")
    plt.ylabel("Self-Declared Confidence Score")
    plt.xticks([0, 1], ["Incorrect", "Correct"])
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path = run_dir / "confidence_vs_accuracy.png"
    plt.savefig(plot_path)
    logger.info("Saved plot {}", plot_path)
    plt.close()

    report_d, report_txt = pec.sklearn_classification_reports(res_df["true"], res_df["pred"])

    pec.save_phase(
        run_dir,
        "full",
        metrics={
            "accuracy": acc,
            "mean_conf_correct": float(res_df[res_df["is_correct"]]["conf"].mean())
            if res_df["is_correct"].any()
            else None,
            "mean_conf_wrong": float(res_df[~res_df["is_correct"]]["conf"].mean())
            if (~res_df["is_correct"]).any()
            else None,
            "n_rows": len(res_df),
            "classification_report": report_d,
            "classification_report_text": report_txt,
            "plot": str(plot_path.name),
        },
        settings={"model": pec.DEFAULT_MODEL, "FULL_N": FULL_N, "num_retries": pec.MAIN_MAX_RETRIES},
        predictions=res_df.to_dict(orient="records"),
        duration_seconds=dt_full,
    )

    print(clf_report_str(res_df["true"], res_df["pred"]))

if __name__ == "__main__":
    main()
