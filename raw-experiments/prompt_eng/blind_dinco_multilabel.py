# %% [markdown]
# ### Experiment: Blind DiNCo — multilabel claims + normalized confidence (2 calls / sample)
# google.genai only. Prelim: print + assert.

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

EXPERIMENT_SLUG = "blind_dinco_multilabel"
SMART_MODEL = os.getenv("SMART_MODEL", pec.DEFAULT_MODEL)
EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL", os.getenv("CHEAP_MODEL", "gemma-4-31b-it"))
EVAL_N = int(os.getenv("EVAL_N", "30"))
MIN_PER_CLASS = int(os.getenv("MIN_PER_CLASS", "5"))
TIE_EPS = float(os.getenv("TIE_EPS", "1e-6"))

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


def _pass1_sys() -> str:
    labs = peg.labels_csv()
    return (
        f"Task: analyze one PubMed-RCT sentence.\n"
        f"Step A — list all applicable labels from [{labs}] (zero or more). Pick one primary_label if a single role dominates.\n"
        "Step B — write critical_reasoning comparing each serious option.\n"
        "Step C — emit mutually exclusive textual claims for blind scoring: one claim per predicted label, "
        "plus ONE extra claim meaning none of the labels is adequate (ambiguous/outside list). Use claim ids c0, c1, ...; "
        "the none-of-the-above claim must have label null.\n\n"
        "Output one JSON object only. Required keys: multilabel, primary_label, critical_reasoning, claims.\n\n"
        "Valid JSON example (structure only):\n"
        f"{peg.EXAMPLE_DINCO_PASS1_JSON}\n"
    )


def _parse_pass1(raw: str) -> dict | None:
    d = peg.extract_json_object(raw)
    if not d or not isinstance(d.get("claims"), list):
        return None
    ml = d.get("multilabel")
    if not isinstance(ml, list):
        ml = []
    ml = [x for x in (_norm(str(x)) for x in ml) if x]
    prim = _norm(str(d.get("primary_label", "")))
    if not prim and ml:
        prim = ml[0]
    claims = []
    for c in d["claims"]:
        if not isinstance(c, dict):
            continue
        claims.append(
            {
                "id": str(c.get("id", f"c{len(claims)}")),
                "label": c.get("label"),
                "text": str(c.get("text", ""))[:800],
            }
        )
    return {
        "multilabel": ml,
        "primary_label": prim or "error",
        "critical_reasoning": str(d.get("critical_reasoning", ""))[:6000],
        "claims": claims,
    }


def _pass2_sys() -> str:
    return (
        "Task: blind scoring matrix.\n"
        "You receive one sentence and a list of claim ids with text. Score each claim independently (0-100): "
        "how well the sentence supports that claim on first principles.\n"
        "Do not compare claims to each other; treat each in isolation.\n"
        "Output one JSON object only. Key \"scores\": object mapping each claim id string to an integer 0-100.\n\n"
        "Valid JSON example:\n"
        f"{peg.EXAMPLE_DINCO_SCORES_JSON}\n"
    )


def _parse_scores(raw: str, claim_ids: list[str]) -> dict[str, float]:
    d = peg.extract_json_object(raw) or {}
    sc = d.get("scores")
    out: dict[str, float] = {}
    if isinstance(sc, dict):
        for k, v in sc.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                out[str(k)] = 0.0
    for cid in claim_ids:
        out.setdefault(cid, 0.0)
    return out


def resolve_dinco(p1: dict, scores: dict[str, float]) -> dict:
    ids = [c["id"] for c in p1["claims"]]
    total = sum(max(0.0, scores.get(i, 0.0)) for i in ids) + 1e-12
    best_id = max(ids, key=lambda i: scores.get(i, 0.0))
    best_s = scores.get(best_id, 0.0)
    vals = sorted([scores.get(i, 0.0) for i in ids], reverse=True)
    s1 = vals[0] if vals else 0.0
    s2 = vals[1] if len(vals) > 1 else 0.0
    tie = len(vals) > 1 and abs(s1 - s2) <= TIE_EPS and s1 > 0
    id_to_label = {c["id"]: c.get("label") for c in p1["claims"]}
    alt_ids = [c["id"] for c in p1["claims"] if c.get("label") is None or str(c.get("label")).lower() in ("null", "none", "")]
    pred = "confusing"
    if tie:
        pred = "confusing"
    else:
        lab = id_to_label.get(best_id)
        if best_id in alt_ids or lab is None:
            pred = "confusing"
        else:
            nl = _norm(str(lab)) if lab else None
            pred = nl if nl else "confusing"
    target_id = None
    for c in p1["claims"]:
        if str(c.get("label", "")).lower() == str(p1.get("primary_label", "")).lower():
            target_id = c["id"]
            break
    if target_id is None and p1["claims"]:
        target_id = p1["claims"][0]["id"]
    t_score = scores.get(target_id or "", 0.0) if target_id else 0.0
    norm_conf = float(t_score / total)
    return {
        "pred": pred,
        "normalized_target_confidence": norm_conf,
        "total_score_mass": float(total),
        "best_claim_id": best_id,
        "tie": tie,
        "scores": scores,
    }


def run_two(client, text: str, *, prelim: bool) -> dict:
    mr = pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES
    r1 = peg.generate_with_retries(
        client,
        model=SMART_MODEL,
        user_text=f"Sentence:\n{text}\n\nReturn JSON only.",
        system_instruction=_pass1_sys(),
        temperature=0.2,
        max_output_tokens=None,
        max_retries=mr,
        label="pass1",
    )
    p1 = _parse_pass1(r1) or {"multilabel": [], "primary_label": "error", "critical_reasoning": "", "claims": []}
    if not p1["claims"]:
        p1["claims"] = [{"id": "c0", "label": p1.get("primary_label"), "text": "fallback claim"}]

    lines = "\n".join([f'{c["id"]}: {c["text"]}' for c in p1["claims"]])
    u2 = f"Sentence:\n{text}\n\nClaims:\n{lines}\n\nReturn JSON scores only."
    r2 = peg.generate_with_retries(
        client,
        model=EXECUTOR_MODEL,
        user_text=u2,
        system_instruction=_pass2_sys(),
        temperature=0.0,
        max_output_tokens=None,
        max_retries=mr,
        label="pass2",
    )
    ids = [c["id"] for c in p1["claims"]]
    sc = _parse_scores(r2, ids)
    res = resolve_dinco(p1, sc)
    return {"pass1": p1, "pass1_raw": r1[:4000], "pass2_raw": r2[:3000], **res}


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")
    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    eval_df, _, _, _ = pec.load_hf_pubmed_splits()
    subset = pec.sample_stratified_eval_subset(
        eval_df, n_total=EVAL_N, min_per_class=MIN_PER_CLASS, seed=pec.DEFAULT_SEED
    )
    texts = subset["text"].astype(str).tolist()
    y_true = subset["label_name"].astype(str).str.lower().tolist()
    client = peg.get_genai_client()

    if not SKIP_PRELIMINARY and texts:
        logger.info("=== PRELIMINARY ===")
        t0 = run_two(client, texts[0], prelim=True)
        assert "normalized_target_confidence" in t0
        logger.info("prelim pred={} gold={}", t0["pred"], y_true[0])
        if PRELIM_ONLY:
            return

    preds: list[str] = []
    rows: list[dict] = []
    wall = time.perf_counter()
    for i, tx in enumerate(tqdm(texts, desc="blind_dinco")):
        r = run_two(client, tx, prelim=False)
        preds.append(r["pred"])
        rows.append({"i": i, "gold": y_true[i], **r})
    dt = time.perf_counter() - wall
    mask = [p != "confusing" and p != "error" for p in preds]
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
            "mean_normalized_target_conf": float(np.mean([r["normalized_target_confidence"] for r in rows])),
            "mean_multilabel_count": float(np.mean([len(r["pass1"].get("multilabel", [])) for r in rows])),
            "classification_report": cr_d,
            "classification_report_text": cr_t,
        },
        settings={"SMART_MODEL": SMART_MODEL, "EXECUTOR_MODEL": EXECUTOR_MODEL},
        predictions=rows,
        duration_seconds=dt,
        notes="2-call blind DiNCo on multilabel-derived claims.",
    )
    if cr_t:
        print(cr_t)
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()
