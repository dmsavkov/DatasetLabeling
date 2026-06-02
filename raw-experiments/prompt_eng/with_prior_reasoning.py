# %% [markdown]
# ### With prior reasoning — two-step (26b analysis → 31b verdict)
# Preliminary: one row. Full: FULL_N rows. See experimentation-rules.mdc.

# %%
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

import prompt_eng_common as pec

EXPERIMENT_SLUG = "with_prior_reasoning"
FULL_N = int(os.getenv("FULL_N", "30"))
SKIP_PRELIMINARY = pec.env_bool("SKIP_PRELIMINARY", False)
PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)


class MultiLabelReasoningSignature(dspy.Signature):
    """Deep analysis of a medical sentence against all 5 rhetorical roles."""

    text = dspy.InputField(desc="The medical sentence to analyze.")
    comprehensive_analysis = dspy.OutputField(
        desc="Detailed breakdown for Background, Objective, Methods, Results, and Conclusions."
    )


class ObjectiveVerdictSignature(dspy.Signature):
    """Decide the most objective label from prior reasoning."""

    text = dspy.InputField(desc="The original sentence.")
    reasoning_evidence = dspy.InputField(desc="The multi-label analysis from the first pass.")
    final_label = dspy.OutputField(
        desc="Exactly one from [background, objective, methods, results, conclusions] OR 'confusing'."
    )


def make_lms(*, prelim: bool):
    nr = pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES
    reasoning_lm = dspy.LM(
        model="openai/gemma-4-31b-it",
        api_base=pec.GOOGLE_OPENAI_BASE_URL,
        api_key=pec.GOOGLE_API_KEY,
        max_tokens=1500,
        num_retries=nr,
    )
    verdict_lm = dspy.LM(
        model="openai/gemini-3.1-flash-lite-preview",
        api_base=pec.GOOGLE_OPENAI_BASE_URL,
        api_key=pec.GOOGLE_API_KEY,
        max_tokens=500,
        num_retries=nr,
    )
    return reasoning_lm, verdict_lm


def run_optimized_two_step(text: str, reasoning_lm: dspy.LM, verdict_lm: dspy.LM) -> tuple[str, str]:
    logger.debug("Pass 1 reasoning...")
    with dspy.context(lm=reasoning_lm):
        reasoner = dspy.Predict(MultiLabelReasoningSignature)
        evidence = reasoner(text=text).comprehensive_analysis
    logger.debug("Pass 2 verdict (evidence chars={})...", len(evidence))
    with dspy.context(lm=verdict_lm):
        verdictor = dspy.Predict(ObjectiveVerdictSignature)
        res = verdictor(text=text, reasoning_evidence=evidence)
    return res.final_label.strip().lower(), evidence


def process_row(row, reasoning_lm: dspy.LM, verdict_lm: dspy.LM):
    try:
        pred, evidence = run_optimized_two_step(row["text"], reasoning_lm, verdict_lm)
        return {"text": row["text"], "true": row["label_name"], "pred": pred, "evidence": evidence}
    except Exception as e:
        logger.exception("Row failed: {}", e)
        return {"text": row["text"], "true": row["label_name"], "pred": "error", "evidence": str(e)}


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    eval_df, _, _, _ = pec.load_hf_pubmed_splits()

    if not SKIP_PRELIMINARY:
        logger.info("=== PRELIMINARY: one row (cheap retries) ===")
        r_lm, v_lm = make_lms(prelim=True)
        row0 = eval_df.iloc[0]
        t0 = time.perf_counter()
        res = process_row(row0, r_lm, v_lm)
        dt = time.perf_counter() - t0
        logger.info("Prelim pred={} true={}", res["pred"], res["true"])
        logger.info("Evidence (trunc): {}...", str(res["evidence"])[:400])

        pre_cr_d, pre_cr_t = pec.sklearn_classification_reports([res["true"]], [res["pred"]])
        pec.save_phase(
            run_dir,
            "preliminary",
            metrics={
                "accuracy": float(res["pred"] == res["true"]),
                "n_rows": 1,
                "classification_report": pre_cr_d,
                "classification_report_text": pre_cr_t,
            },
            settings={
                "reasoning_model": "openai/gemma-4-31b-it",
                "verdict_model": "openai/gemini-3.1-flash-lite-preview",
                "num_retries": pec.PRELIM_MAX_RETRIES,
            },
            predictions=[{**res, "evidence": str(res["evidence"])[:2000]}],
            duration_seconds=dt,
        )
        if PRELIM_ONLY:
            logger.warning("PRELIM_ONLY=1 — skip full run.")
            return

    logger.info("=== FULL: {} rows ===", FULL_N)
    eval_subset = eval_df.sample(FULL_N, random_state=pec.DEFAULT_SEED)
    r_lm_f, v_lm_f = make_lms(prelim=False)
    results: list[dict] = []
    t1 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_row, row, r_lm_f, v_lm_f) for _, row in eval_subset.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())
    dt_full = time.perf_counter() - t1

    results_df = pd.DataFrame(results)
    accuracy = float((results_df["true"] == results_df["pred"]).mean())
    confusion_rate = float((results_df["pred"] == "confusing").mean())

    logger.info("Accuracy {:.2%} | confusing rate {:.2%} | wall {:.2f}s", accuracy, confusion_rate, dt_full)

    cr_d, cr_t = pec.sklearn_classification_reports(
        results_df["true"].tolist(),
        results_df["pred"].tolist(),
    )
    pec.save_phase(
        run_dir,
        "full",
        metrics={
            "accuracy": accuracy,
            "confusion_rate": confusion_rate,
            "n_rows": len(results_df),
            "classification_report": cr_d,
            "classification_report_text": cr_t,
        },
        settings={
            "reasoning_model": "openai/gemma-4-31b-it",
            "verdict_model": "openai/gemini-3.1-flash-lite-preview",
            "FULL_N": FULL_N,
            "num_retries": pec.MAIN_MAX_RETRIES,
        },
        predictions=results_df.to_dict(orient="records"),
        duration_seconds=dt_full,
    )


if __name__ == "__main__":
    main()
