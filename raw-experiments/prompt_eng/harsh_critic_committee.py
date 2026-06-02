# %% [markdown]
# ### Experiment: Harsh critic committee — pass1 multilabel/top2 + 3 critic calls + resolver
# google.genai. Prelim: print + assert.

# %%
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

import prompt_eng_common as pec
import prompt_eng_gemini as peg

EXPERIMENT_SLUG = "harsh_critic_committee"
SMART_MODEL = os.getenv("SMART_MODEL", pec.DEFAULT_MODEL)
EVAL_N = int(os.getenv("EVAL_N", "30"))
MIN_PER_CLASS = int(os.getenv("MIN_PER_CLASS", "5"))
FEWSHOT_N = int(os.getenv("FEWSHOT_N", "8"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))

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
        "Valid JSON example:\n"
        f"{peg.EXAMPLE_PASS1_TOPTWO_JSON}\n\n"
        f"Reference few-shot:\n{fewshot}\n"
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
    top = []
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


def _critic_sys(which: int, fewshot: str) -> str:
    labs = peg.labels_csv()
    base = (
        f"Task: critic. Labels [{labs}]. You receive sentence + pass1 JSON (multilabel + top_two).\n"
        "Output one JSON object only. Keys: critique (string), recommended_label (one label or exactly \"confusing\"), confidence (integer 1-100).\n\n"
        f"Valid JSON example:\n{peg.EXAMPLE_CRITIC_JSON}\n\n"
    )
    if which == 1:
        role = "Role: Critic A — harsh skeptic. Attack the strongest (top-1) hypothesis.\n"
    elif which == 2:
        role = "Role: Critic B — different harsh angle; stress-test top_two.\n"
    else:
        role = "Role: Critic C — alternatives specialist; argue for labels outside top_two if warranted.\n"
    return base + role + f"Reference few-shot:\n{fewshot}\n"


def _parse_critic(raw: str) -> dict:
    d = peg.extract_json_object(raw) or {}
    rec = str(d.get("recommended_label", "")).strip().lower()
    if rec == "confusing":
        lab = "confusing"
    else:
        lab = _norm(rec) or "error"
    try:
        conf = int(d.get("confidence", 1))
    except (TypeError, ValueError):
        conf = 1
    return {"critique": str(d.get("critique", ""))[:4000], "recommended_label": lab, "confidence": max(1, min(100, conf)), "raw": raw[:2500]}


def _resolver_sys(fewshot: str) -> str:
    labs = peg.labels_csv()
    return (
        f"Task: resolver. Allowed final_label: [{labs}] or exactly \"confusing\".\n"
        "Input: sentence + pass1 + three critic JSON objects.\n"
        "Rules: tie / equal support for multiple labels → \"confusing\". Clear consensus → that label.\n"
        "Output one JSON object only. Keys: final_label, resolution_notes.\n\n"
        f"Valid JSON example:\n{peg.EXAMPLE_RESOLVER_JSON}\n\n"
        f"Reference few-shot:\n{fewshot}\n"
    )


def _parse_resolver(raw: str) -> dict:
    d = peg.extract_json_object(raw) or {}
    fl = str(d.get("final_label", "")).strip().lower()
    if fl == "confusing":
        pred = "confusing"
    else:
        pred = _norm(fl) or "error"
    return {"final_label": pred, "resolution_notes": str(d.get("resolution_notes", ""))[:2000], "raw": raw[:2000]}


def run_one(client, text: str, *, fewshot: str, prelim: bool) -> dict:
    mr = pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES
    raw_p = peg.generate_with_retries(
        client,
        model=SMART_MODEL,
        user_text=f"Sentence:\n{text}\n\nReturn JSON only.",
        system_instruction=_pass1_sys(fewshot),
        temperature=0.2,
        max_output_tokens=None,
        max_retries=mr,
        label="pass1",
    )
    p1 = _parse_pass1(raw_p) or {"multilabel": [], "top_two": [], "notes": ""}
    ctx = json.dumps(p1, indent=2)
    user_c = f"Sentence:\n{text}\n\nPass1:\n{ctx}\n\nReturn JSON only."

    def work(which: int) -> tuple[int, dict]:
        raw = peg.generate_with_retries(
            client,
            model=SMART_MODEL,
            user_text=user_c,
            system_instruction=_critic_sys(which, fewshot),
            temperature=0.4,
            max_output_tokens=None,
            max_retries=mr,
            label=f"critic{which}",
        )
        return which, _parse_critic(raw)

    critics: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(work, w) for w in (1, 2, 3)]
        for fut in as_completed(futs):
            w, d = fut.result()
            critics[w] = d

    res_user = json.dumps(
        {"sentence": text[:2000], "pass1": p1, "critic1": critics[1], "critic2": critics[2], "critic3": critics[3]},
        indent=2,
    )
    raw_r = peg.generate_with_retries(
        client,
        model=SMART_MODEL,
        user_text=res_user,
        system_instruction=_resolver_sys(fewshot),
        temperature=0.0,
        max_output_tokens=None,
        max_retries=mr,
        label="resolver",
    )
    res = _parse_resolver(raw_r)
    return {"pass1": p1, "pass1_raw": raw_p[:3500], "critics": critics, "resolver": res}


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
        assert 1 in r0["critics"] and 2 in r0["critics"] and 3 in r0["critics"]
        logger.info("prelim final={} gold={}", r0["resolver"]["final_label"], y_true[0])
        if PRELIM_ONLY:
            return

    preds: list[str] = []
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, tx in enumerate(tqdm(texts, desc="harsh_critic_committee")):
        r = run_one(client, tx, fewshot=fewshot, prelim=False)
        preds.append(r["resolver"]["final_label"])
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
        settings={"SMART_MODEL": SMART_MODEL, "MAX_WORKERS": MAX_WORKERS},
        predictions=rows,
        duration_seconds=dt,
        notes="Pass1 + 3 parallel harsh critics + resolver.",
    )
    if cr_t:
        print(cr_t)
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()
