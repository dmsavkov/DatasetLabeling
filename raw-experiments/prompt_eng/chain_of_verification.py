# %% [markdown]
# ### Experiment: Chain-of-verification style pipeline (draft → questions → blind answers → judge)
# Preliminary: print + assert only. Full: save. google.genai only.

# %%
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

import prompt_eng_common as pec
import prompt_eng_gemini as peg

EXPERIMENT_SLUG = "chain_of_verification"
SMART_MODEL = os.getenv("SMART_MODEL", "gemma-4-31b-it")
EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL", os.getenv("CHEAP_MODEL", "gemma-4-31b-it"))
EVAL_N = int(os.getenv("EVAL_N", "30"))
MIN_PER_CLASS = int(os.getenv("MIN_PER_CLASS", "5"))
FEWSHOT_N = int(os.getenv("FEWSHOT_N", "8"))

SKIP_PRELIMINARY = pec.env_bool("SKIP_PRELIMINARY", False)
PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)


def _norm(s: str) -> str | None:
    v = (s or "").strip().lower()
    if v in ("conclusion", "concl"):
        v = "conclusions"
    if v == "method":
        v = "methods"
    if v == "result":
        v = "results"
    return v if v in set(pec.VALID_LABELS) else None


def _fewshot(train_df: pd.DataFrame) -> str:
    tdf = train_df.copy()
    tdf["label_name"] = tdf["label"].apply(pec.label_name_from_value)
    fs = pec.sample_balanced_train_fewshot(tdf, FEWSHOT_N, label_col="label_name", seed=pec.DEFAULT_SEED)
    return pec.format_fewshot_block(fs)


def _initial_sys(fewshot: str) -> str:
    labs = peg.labels_csv()
    return (
        f"Task: assign exactly ONE label from [{labs}] to a PubMed-RCT sentence.\n\n"
        "Process:\n"
        "  1. Reason step by step in the reasoning field.\n"
        "  2. In alternatives_analysis, list other labels you considered and why you reject or accept them.\n"
        "  3. Output one JSON object only (no markdown, no text outside JSON).\n\n"
        "Required JSON keys: reasoning, alternatives_analysis, label.\n\n"
        "Valid JSON example (replace content with your own analysis):\n"
        + json.dumps(
            {
                "reasoning": "Step-by-step...",
                "alternatives_analysis": "Considered results vs methods because...",
                "label": "methods",
            },
            ensure_ascii=False,
        )
        + "\n\n"
        f"Reference few-shot (style only):\n{fewshot}\n"
    )


def _parse_initial(raw: str) -> dict | None:
    d = peg.extract_json_object(raw)
    if not d or not _norm(str(d.get("label", ""))):
        return None
    return {
        "reasoning": str(d.get("reasoning", ""))[:4000],
        "alternatives_analysis": str(d.get("alternatives_analysis", ""))[:4000],
        "label": _norm(str(d.get("label", ""))),
    }


def _questioner_sys() -> str:
    return (
        "Task: read the draft classification for a sentence and list verification questions.\n"
        "Output one JSON object only. Key: \"questions\" — array of 3 to 5 strings.\n"
        "Each question: specific, checkable, logically crucial (assumptions or facts that must hold).\n"
        "Maximum 5 questions.\n\n"
        "Valid JSON example:\n"
        f"{peg.EXAMPLE_QUESTIONS_JSON}\n"
    )


def _parse_questions(raw: str) -> list[str]:
    d = peg.extract_json_object(raw)
    if not d:
        return []
    qs = d.get("questions")
    if not isinstance(qs, list):
        return []
    out = [str(q).strip() for q in qs if str(q).strip()]
    return out[:5] if len(out) > 5 else out


def _blind_sys() -> str:
    return (
        "Task: answer each question using ONLY the sentence provided in the user message.\n"
        "You do not see any draft classification.\n"
        "Rules per answer: short factual phrase grounded in the sentence, OR exactly \"not stated\".\n"
        "Output one JSON object only. Key \"answers\": array of objects in the SAME order as the questions, "
        "each object has keys \"question\" and \"answer\".\n\n"
        "Valid JSON example:\n"
        f"{peg.EXAMPLE_BLIND_ANSWERS_JSON}\n"
    )


def _parse_blind(raw: str, questions: list[str]) -> list[dict]:
    d = peg.extract_json_object(raw)
    if not d or not isinstance(d.get("answers"), list):
        return [{"question": q, "answer": "not stated"} for q in questions]
    arr = d["answers"]
    out: list[dict] = []
    for i, q in enumerate(questions):
        ans = "not stated"
        if i < len(arr) and isinstance(arr[i], dict):
            ans = str(arr[i].get("answer", "not stated")).strip() or "not stated"
        out.append({"question": q, "answer": ans})
    return out


def _final_judge_sys(fewshot: str) -> str:
    labs = peg.labels_csv()
    return (
        f"Task: final judge. Allowed final_label values: [{labs}] or exactly \"confusing\".\n"
        "Inputs (in user JSON): sentence, draft, questions, blind_answers.\n"
        "If blind answers undermine the draft or multiple labels fit equally, set final_label to \"confusing\".\n"
        "Otherwise pick one label from the allowed list.\n"
        "Output one JSON object only. Keys: final_label, synthesis.\n\n"
        "Valid JSON example:\n"
        f"{peg.EXAMPLE_FINAL_JUDGE_JSON}\n\n"
        f"Reference few-shot:\n{fewshot}\n"
    )


def _not_stated_frac(answers: list[dict]) -> float:
    n = len(answers)
    if n == 0:
        return 1.0
    ns = sum(1 for a in answers if str(a.get("answer", "")).strip().lower() in ("not stated", "not stated.", "unknown"))
    return ns / float(n)


def run_pipeline(client, text: str, *, fewshot: str, prelim: bool) -> dict:
    mr = pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES
    raw0 = peg.generate_with_retries(
        client,
        model=SMART_MODEL,
        user_text=f"Sentence:\n{text}\n\nReturn JSON only.",
        system_instruction=_initial_sys(fewshot),
        temperature=0.3,
        max_output_tokens=None,
        max_retries=mr,
        label="initial",
    )
    init = _parse_initial(raw0) or {"reasoning": "", "alternatives_analysis": "", "label": "error"}

    raw_q = peg.generate_with_retries(
        client,
        model=SMART_MODEL,
        user_text=json.dumps({"sentence": text[:2000], "draft_label": init["label"], "draft_reasoning": init["reasoning"][:2500]}),
        system_instruction=_questioner_sys(),
        temperature=0.2,
        max_output_tokens=None,
        max_retries=mr,
        label="questioner",
    )
    questions = _parse_questions(raw_q)
    if len(questions) < 3:
        questions = (questions + [f"Q{i}?" for i in range(3)])[:3]

    qblock = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    blind_user = f"Sentence:\n{text}\n\nQuestions (answer in order):\n{qblock}\n\nReturn JSON only."
    raw_b = peg.generate_with_retries(
        client,
        model=EXECUTOR_MODEL,
        user_text=blind_user,
        system_instruction=_blind_sys(),
        temperature=0.0,
        max_output_tokens=None,
        max_retries=mr,
        label="blind",
    )
    answers = _parse_blind(raw_b, questions)
    ns_frac = _not_stated_frac(answers)
    coverage_conf = 1.0 - ns_frac

    judge_user = json.dumps(
        {
            "sentence": text[:2000],
            "draft": init,
            "questions": questions,
            "blind_answers": answers,
        },
        indent=2,
    )
    raw_f = peg.generate_with_retries(
        client,
        model=SMART_MODEL,
        user_text=judge_user,
        system_instruction=_final_judge_sys(fewshot),
        temperature=0.0,
        max_output_tokens=None,
        max_retries=mr,
        label="final_judge",
    )
    fd = peg.extract_json_object(raw_f) or {}
    fl = str(fd.get("final_label", "")).strip().lower()
    if fl == "confusing":
        pred = "confusing"
    else:
        pred = _norm(fl) or "error"

    return {
        "initial": init,
        "questions": questions,
        "blind_raw": raw_b[:3000],
        "answers": answers,
        "not_stated_fraction": ns_frac,
        "coverage_confidence": coverage_conf,
        "final_label": pred,
        "synthesis": str(fd.get("synthesis", ""))[:2000],
    }


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")
    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    eval_df, _, train_df, _ = pec.load_hf_pubmed_splits()
    train_df = train_df.copy()
    train_df["label_name"] = train_df["label"].apply(pec.label_name_from_value)
    fewshot = _fewshot(train_df)
    subset = pec.sample_stratified_eval_subset(
        eval_df, n_total=EVAL_N, min_per_class=MIN_PER_CLASS, seed=pec.DEFAULT_SEED
    )
    texts = subset["text"].astype(str).tolist()
    y_true = subset["label_name"].astype(str).str.lower().tolist()
    client = peg.get_genai_client()

    if not SKIP_PRELIMINARY and texts:
        logger.info("=== PRELIMINARY (print only) ===")
        r0 = run_pipeline(client, texts[0], fewshot=fewshot, prelim=True)
        assert len(r0["questions"]) >= 3
        assert len(r0["answers"]) == len(r0["questions"])
        logger.info("prelim final={} gold={}", r0["final_label"], y_true[0])
        if PRELIM_ONLY:
            return

    preds: list[str] = []
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, tx in enumerate(tqdm(texts, desc="chain_of_verification")):
        r = run_pipeline(client, tx, fewshot=fewshot, prelim=False)
        preds.append(r["final_label"])
        rows.append({"i": i, "gold": y_true[i], **r})
    dt = time.perf_counter() - t0

    mask = [p not in ("confusing", "error") for p in preds]
    yt_e = [y_true[i] for i in range(len(y_true)) if mask[i]]
    pr_e = [preds[i] for i in range(len(preds)) if mask[i]]
    acc = float(accuracy_score(yt_e, pr_e)) if yt_e else 0.0
    cr_d, cr_t = pec.sklearn_classification_reports(yt_e, pr_e) if yt_e else ({}, "")
    mean_cov = float(np.mean([r["coverage_confidence"] for r in rows]))

    pec.save_phase(
        run_dir,
        "full",
        metrics={
            "accuracy_excl_confusing": acc,
            "n_confusing": sum(1 for p in preds if p == "confusing"),
            "mean_coverage_confidence": mean_cov,
            "classification_report": cr_d,
            "classification_report_text": cr_t,
        },
        settings={"SMART_MODEL": SMART_MODEL, "EXECUTOR_MODEL": EXECUTOR_MODEL, "EVAL_N": EVAL_N},
        predictions=rows,
        duration_seconds=dt,
        notes="CoVE-style; blind cheap answers; confidence from confusing + 1 - not_stated_frac.",
    )
    if cr_t:
        print(cr_t)
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()
