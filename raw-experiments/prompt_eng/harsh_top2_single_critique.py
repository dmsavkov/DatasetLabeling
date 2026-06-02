# %% [markdown]
# ### Experiment: Harsh top-2 single critique — pass1 multilabel/top2 + ONE combined harsh critique → final label
# google.genai. Prelim: print + assert. Fewer calls than harsh_critic_committee (no 3 critics + resolver).

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

EXPERIMENT_SLUG = "harsh_top2_single_critique"
SMART_MODEL = os.getenv("SMART_MODEL", pec.DEFAULT_MODEL)
EVAL_N = int(os.getenv("EVAL_N", "40"))
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


def _pass1_sys(fewshot: str) -> str:
    labs = peg.labels_csv()
    return (
        f"Task: PubMed-RCT sentence — labels [{labs}].\n"
        "Output one JSON object only with keys: multilabel (array), top_two (array of two objects with label and probability 0-1), notes (string).\n"
        "top_two = the two most likely single labels.\n\n"
        f"Reference few-shot:\n{fewshot}\n"
        "Valid JSON example:\n"
        f"{peg.EXAMPLE_PASS1_TOPTWO_JSON}\n\n"
    )


def _parse_pass1(raw: str) -> dict | None:
    d = peg.extract_json_object(raw)
    if not d:
        return None
    ml = d.get("multilabel")
    if not isinstance(ml, list):
        ml = []
    ml = [x for x in (_norm(str(x)) for x in ml) if x]
    tt = d.get("top_two")
    top: list[dict] = []
    if isinstance(tt, list):
        for it in tt[:2]:
            if isinstance(it, dict):
                lb = _norm(str(it.get("label", "")))
                if lb:
                    try:
                        p = float(it.get("probability", 0))
                    except (TypeError, ValueError):
                        p = 0.0
                    top.append({"label": lb, "probability": p})
    return {"multilabel": ml, "top_two": top, "notes": str(d.get("notes", ""))[:1500]}


def _label_a_b(p1: dict) -> tuple[str, str]:
    """Two hypotheses for the single critique (from top_two, else multilabel fallbacks)."""
    tt = p1.get("top_two") or []
    la = str(tt[0]["label"]) if tt else ""
    lb = str(tt[1]["label"]) if len(tt) > 1 else ""
    if not la:
        la = (p1.get("multilabel") or ["methods"])[0]
    if not lb:
        for x in p1.get("multilabel") or []:
            if x != la:
                lb = x
                break
    if not lb:
        lb = la
    return la, lb


def _single_critique_sys(fewshot: str) -> str:
    labs = peg.labels_csv()
    return (
        f"Task: single harsh critique. Labels [{labs}].\n"
        "You receive the sentence, pass1 JSON, and explicit label_a / label_b (the two leading single-label hypotheses).\n"
        "In one pass: harshly stress-test label_a and label_b; consider whether another label from the list fits better; "
        "if two or more labels remain equally defensible, output final_label exactly \"confusing\".\n"
        "Output one JSON object only. Required keys: critique_of_label_a, critique_of_label_b, alternatives_considered, final_label.\n"
        f"final_label must be one of [{labs}] or exactly \"confusing\".\n\n"
        f"Reference few-shot:\n{fewshot}\n"
        "Valid JSON example:\n"
        f"{peg.EXAMPLE_TOP2_SINGLE_JSON}\n\n"
    )


def _parse_single_critique(raw: str) -> dict:
    d = peg.extract_json_object(raw) or {}
    fl = str(d.get("final_label", "")).strip().lower()
    if fl == "confusing":
        pred = "confusing"
    else:
        pred = _norm(fl) or "error"
    return {
        "critique_of_label_a": str(d.get("critique_of_label_a", ""))[:4000],
        "critique_of_label_b": str(d.get("critique_of_label_b", ""))[:4000],
        "alternatives_considered": str(d.get("alternatives_considered", ""))[:4000],
        "final_label": pred,
        "raw": raw[:4000],
    }


def run_one(client, text: str, *, fewshot: str, prelim: bool) -> dict:
    mr = pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES
    raw_p = peg.generate_with_retries(
        client,
        model=SMART_MODEL,
        user_text=f"Sentence:\n{text}\n\nReturn one JSON object only as specified in instructions.",
        system_instruction=_pass1_sys(fewshot),
        temperature=0.2,
        max_output_tokens=None,
        max_retries=mr,
        label="pass1",
    )
    p1 = _parse_pass1(raw_p) or {"multilabel": [], "top_two": [], "notes": ""}
    la, lb = _label_a_b(p1)
    crit_user = (
        f"Sentence:\n{text}\n\n"
        f"pass1:\n{json.dumps(p1, indent=2)}\n\n"
        f"label_a: {la}\n"
        f"label_b: {lb}\n\n"
        "Return one JSON object only as specified in instructions."
    )
    raw_c = peg.generate_with_retries(
        client,
        model=SMART_MODEL,
        user_text=crit_user,
        system_instruction=_single_critique_sys(fewshot),
        temperature=0.35,
        max_output_tokens=None,
        max_retries=mr,
        label="single_critique",
    )
    crit = _parse_single_critique(raw_c)
    return {"pass1": p1, "pass1_raw": raw_p[:3500], "label_a": la, "label_b": lb, "critique": crit}


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
    y_true = subset["label_name"].astype(str).str.lower().tolist()
    client = peg.get_genai_client()

    if not SKIP_PRELIMINARY and texts:
        logger.info("=== PRELIMINARY ===")
        r0 = run_one(client, texts[0], fewshot=fewshot, prelim=True)
        assert "pass1" in r0 and "critique" in r0
        assert r0["critique"]["final_label"] in set(pec.VALID_LABELS) | {"confusing", "error"}
        logger.info("prelim final={} gold={}", r0["critique"]["final_label"], y_true[0])
        if PRELIM_ONLY:
            return

    preds: list[str] = []
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, tx in enumerate(tqdm(texts, desc="harsh_top2_single_critique")):
        r = run_one(client, tx, fewshot=fewshot, prelim=False)
        preds.append(r["critique"]["final_label"])
        rows.append({"i": i, "gold": y_true[i], **r})
    dt = time.perf_counter() - t0
    mask = [p not in ("confusing", "error") for p in preds]
    yt_e = [y_true[i] for i in range(len(y_true)) if mask[i]]
    pr_e = [preds[i] for i in range(len(preds)) if mask[i]]
    acc = float(accuracy_score(yt_e, pr_e)) if yt_e else 0.0
    cr_d, cr_t = pec.sklearn_classification_reports(yt_e, pr_e) if yt_e else ({}, "")

    pec.save_phase(
        run_dir,
        "full",
        metrics={
            "accuracy_excl_confusing": acc,
            "n_confusing": sum(1 for p in preds if p == "confusing"),
            "classification_report": cr_d,
            "classification_report_text": cr_t,
        },
        settings={"SMART_MODEL": SMART_MODEL},
        predictions=rows,
        duration_seconds=dt,
        notes="Pass1 top_two + one combined harsh critique JSON → final_label.",
    )
    if cr_t:
        print(cr_t)
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()
