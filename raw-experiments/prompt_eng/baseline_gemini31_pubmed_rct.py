# %% [markdown]
# ### Baseline: single-shot PubMed-RCT classification with Gemini 3.1 (balanced n=30, min 5/class)
# google.genai. Prelim: print + assert. Full: save.

# %%
from __future__ import annotations

import json
import os
import time

import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

import prompt_eng_common as pec
import prompt_eng_gemini as peg

EXPERIMENT_SLUG = "baseline_gemini31_pubmed_rct"
MODEL = os.getenv("SMART_MODEL", "gemma-4-31b-it")
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


def _fewshot_block(train_df) -> str:
    tdf = train_df.copy()
    tdf["label_name"] = tdf["label"].apply(pec.label_name_from_value)
    fs = pec.sample_balanced_train_fewshot(tdf, FEWSHOT_N, label_col="label_name", seed=pec.DEFAULT_SEED)
    return pec.format_fewshot_block(fs)


def _system_instruction(fewshot: str) -> str:
    labs = peg.labels_csv()
    return (
        f"Task: assign exactly ONE rhetorical label from [{labs}] to a PubMed-RCT sentence.\n\n"
        "Output one JSON object only (no markdown fences, no text outside JSON).\n"
        "Keys must match the example: reasoning, label, confidence (integer 1-100 for the chosen label).\n\n"
        "Valid JSON example (replace with your own content):\n"
        f"{peg.EXAMPLE_TRACE_JSON}\n\n"
        f"Reference few-shot (style only):\n{fewshot}\n"
    )


def _parse(raw: str) -> tuple[str | None, str]:
    d = peg.extract_json_object(raw)
    if not d:
        return None, raw[:2000]
    lab = _norm(str(d.get("label", "")))
    return lab, raw[:2000]


def run_one(client, text: str, *, fewshot: str, prelim: bool) -> dict:
    mr = pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES
    raw = peg.generate_with_retries(
        client,
        model=MODEL,
        user_text=f"Sentence:\n{text}\n\nReturn one JSON object only as specified in instructions.",
        system_instruction=_system_instruction(fewshot),
        temperature=0.2,
        max_output_tokens=None,
        max_retries=mr,
        label="baseline",
    )
    pred, snippet = _parse(raw)
    out_label = pred if pred is not None else "error"
    return {"pred": out_label, "raw": raw[:4000], "parse_snippet": snippet}


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    eval_df, _, train_df, _ = pec.load_hf_pubmed_splits()
    train_df = train_df.copy()
    train_df["label_name"] = train_df["label"].apply(pec.label_name_from_value)
    fewshot = _fewshot_block(train_df)

    subset = pec.sample_stratified_eval_subset(
        eval_df, n_total=EVAL_N, min_per_class=MIN_PER_CLASS, seed=pec.DEFAULT_SEED
    )
    texts = subset["text"].astype(str).tolist()
    logger.info(texts[:6])
    y_true = subset["label_name"].astype(str).str.lower().tolist()
    logger.info(y_true[:6])
    client = peg.get_genai_client()

    if not SKIP_PRELIMINARY and texts:
        logger.info("=== PRELIMINARY ===")
        r0 = run_one(client, texts[0], fewshot=fewshot, prelim=True)
        assert r0["pred"] in set(pec.VALID_LABELS) | {"error"}
        logger.info("prelim pred={} gold={}", r0["pred"], y_true[0])
        if PRELIM_ONLY:
            return

    preds: list[str] = []
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, tx in enumerate(tqdm(texts, desc=EXPERIMENT_SLUG)):
        r = run_one(client, tx, fewshot=fewshot, prelim=False)
        preds.append(r["pred"])
        rows.append({"i": i, "gold": y_true[i], **r})
        logger.info(f"pred={r['pred']} gold={y_true[i]}")
    dt = time.perf_counter() - t0

    mask = [p != "error" for p in preds]
    yt_e = [y_true[i] for i in range(len(y_true)) if mask[i]]
    pr_e = [preds[i] for i in range(len(preds)) if mask[i]]
    acc = float(accuracy_score(yt_e, pr_e)) if yt_e else 0.0
    cr_d, cr_t = pec.sklearn_classification_reports(yt_e, pr_e) if yt_e else ({}, "")

    pec.save_phase(
        run_dir,
        "full",
        metrics={
            "accuracy_excl_error": acc,
            "n_error": sum(1 for p in preds if p == "error"),
            "classification_report": cr_d,
            "classification_report_text": cr_t,
        },
        settings={
            "MODEL": MODEL,
            "EVAL_N": EVAL_N,
            "MIN_PER_CLASS": MIN_PER_CLASS,
            "FEWSHOT_N": FEWSHOT_N,
        },
        predictions=rows,
        duration_seconds=dt,
        notes="Single-shot JSON classification; Gemini 3.1 default; stratified eval subset.",
    )
    if cr_t:
        print(cr_t)
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()
