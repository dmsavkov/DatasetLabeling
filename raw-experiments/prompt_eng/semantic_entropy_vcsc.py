# %% [markdown]
# ### Experiment: Semantic entropy (high-T ensemble) + SC / VCavg / VCSC (label math only, no extra judge call)
# Preliminary: print + assertions only. Full: save. google.genai only.

# %%
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter

import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

import prompt_eng_common as pec
import prompt_eng_gemini as peg

EXPERIMENT_SLUG = "semantic_entropy_vcsc"
# High-temperature trace sampling: executor-tier model (open-weight).
EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL", os.getenv("CHEAP_MODEL", "gemma-4-31b-it"))
M_TRACES = int(os.getenv("M_TRACES", "3"))
VCSC_LAMBDA = float(os.getenv("VCSC_LAMBDA", "0.5"))
EVAL_N = int(os.getenv("EVAL_N", "30"))
MIN_PER_CLASS = int(os.getenv("MIN_PER_CLASS", "5"))
FEWSHOT_N = int(os.getenv("FEWSHOT_N", "8"))
EXEC_TEMP = float(os.getenv("EXEC_TEMP", "0.95"))
EXEC_TOP_P = float(os.getenv("EXEC_TOP_P", "0.95"))

SKIP_PRELIMINARY = pec.env_bool("SKIP_PRELIMINARY", False)
PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)


def _norm_label(s: str) -> str | None:
    v = (s or "").strip().lower()
    if v in ("conclusion", "concl"):
        v = "conclusions"
    if v == "method":
        v = "methods"
    if v == "result":
        v = "results"
    return v if v in set(pec.VALID_LABELS) else None


def _entropy_labels(labels: list[str]) -> float:
    c = Counter(labels)
    n = sum(c.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for ct in c.values():
        p = ct / n
        h -= p * math.log(p + 1e-12, 2)
    return float(h)


def _plurality_or_confusing(labels: list[str]) -> tuple[str, str | None]:
    """
    Returns (status, label).
    status: 'ok' | 'confusing' | 'all_error'
    label: winning label or None
    """
    good = [lb for lb in labels if lb != "error"]
    if not good:
        return "all_error", None
    c = Counter(good)
    mc = c.most_common(2)
    top_lab, top_n = mc[0]
    if len(mc) == 1:
        return "ok", top_lab
    second_n = mc[1][1]
    if top_n > second_n:
        return "ok", top_lab
    return "confusing", None


def _fewshot_block(train_df) -> str:
    fs = pec.sample_balanced_train_fewshot(
        train_df, FEWSHOT_N, label_col="label_name", seed=pec.DEFAULT_SEED
    )
    return pec.format_fewshot_block(fs)


def _executor_instructions(fewshot: str) -> str:
    labs = peg.labels_csv()
    return (
        f"Task: classify ONE PubMed-RCT rhetorical label from: [{labs}].\n\n"
        "Epistemic confidence (integer 1-100, for your chosen label only — do not re-pick the label after setting it):\n"
        "  1 = random guess; 25 = significant doubts; 50 = mixed; 75 = mostly confident; 100 = completely certain.\n\n"
        "Reasoning: write exactly four bullet lines (each line starts with \"- \") showing steps toward your label.\n\n"
        "Reference few-shot (style only):\n"
        f"{fewshot}\n"
        "Output rules:\n"
        "  1. Respond with a single JSON object only (no markdown fences, no prose outside JSON).\n"
        "  2. Keys must be exactly: reasoning, label, confidence.\n\n"
        "Valid example (structure only — use your own reasoning and label):\n"
        f"{peg.EXAMPLE_TRACE_JSON}\n\n"
    )


def _executor_user(text: str) -> str:
    return f"Sentence to classify:\n{text}\n"


def _parse_executor(raw: str) -> dict | None:
    d = peg.extract_json_object(raw)
    if not d:
        return None
    lab = _norm_label(str(d.get("label", "")))
    if lab is None:
        return None
    try:
        conf = int(d.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0
    conf = max(1, min(100, conf))
    r = str(d.get("reasoning", "")).strip()
    return {"label": lab, "confidence": conf, "reasoning": r, "raw": raw[:2000]}


def run_one(client, text: str, *, fewshot: str, prelim: bool) -> dict:
    mr = pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES
    sys_ex = _executor_instructions(fewshot)
    traces: list[dict] = []
    for m in range(M_TRACES):
        raw = peg.generate_with_retries(
            client,
            model=EXECUTOR_MODEL,
            user_text=_executor_user(text),
            system_instruction=sys_ex,
            temperature=EXEC_TEMP,
            top_p=EXEC_TOP_P,
            max_output_tokens=None,
            max_retries=mr,
            label=f"exec_{m}",
        )
        parsed = _parse_executor(raw)
        if parsed is None:
            parsed = {"label": "error", "confidence": 1, "reasoning": "", "raw": raw[:1500]}
        traces.append(parsed)

    labels_m = [t["label"] for t in traces]
    st, winner = _plurality_or_confusing(labels_m)
    unanimous = len(set(labels_m)) == 1 and labels_m[0] != "error"
    entailment_same_label = unanimous

    if st == "confusing" or winner is None:
        pred_for_acc = "confusing"
        a_hat: str | None = None
        sc = 0.0
        vcavg = float(np.mean([t["confidence"] for t in traces])) if traces else 0.0
    elif st == "all_error":
        pred_for_acc = "error"
        a_hat = None
        sc = 0.0
        vcavg = float(np.mean([t["confidence"] for t in traces])) if traces else 0.0
    else:
        a_hat = winner
        pred_for_acc = a_hat
        sc = sum(1 for lb in labels_m if lb == a_hat) / float(M_TRACES)
        confs_match = [t["confidence"] for t in traces if t["label"] == a_hat]
        vcavg = float(np.mean(confs_match)) if confs_match else float(
            np.mean([t["confidence"] for t in traces])
        )

    vcsc = VCSC_LAMBDA * sc + (1.0 - VCSC_LAMBDA) * (vcavg / 100.0)
    ent = _entropy_labels(labels_m)
    varc = float(np.var([t["confidence"] for t in traces])) if len(traces) > 1 else 0.0

    return {
        "traces": traces,
        "majority_label_resolved": a_hat,
        "consensus_status": st,
        "entailment_same_label_math": entailment_same_label,
        "SC": sc,
        "VCavg": vcavg,
        "VCSC": vcsc,
        "label_entropy": ent,
        "confidence_variance": varc,
        "pred_for_eval": pred_for_acc,
    }


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    eval_df, _, train_df, _ = pec.load_hf_pubmed_splits()
    train_df = train_df.copy()
    train_df["label_name"] = train_df["label"].apply(pec.label_name_from_value)
    subset = pec.sample_stratified_eval_subset(
        eval_df, n_total=EVAL_N, min_per_class=MIN_PER_CLASS, seed=pec.DEFAULT_SEED
    )
    texts = subset["text"].astype(str).tolist()
    y_true = subset["label_name"].astype(str).str.lower().tolist()

    fewshot = _fewshot_block(train_df)
    client = peg.get_genai_client()

    if not SKIP_PRELIMINARY and texts:
        logger.info("=== PRELIMINARY (print only, no save) ===")
        r0 = run_one(client, texts[0], fewshot=fewshot, prelim=True)
        assert len(r0["traces"]) == M_TRACES
        assert "VCSC" in r0 and isinstance(r0["VCSC"], float)
        assert 0.0 <= r0["SC"] <= 1.0
        assert all("label" in t for t in r0["traces"])
        logger.info(
            "prelim pred={} gold={} SC={:.2f} VCSC={:.3f} status={}",
            r0["pred_for_eval"],
            y_true[0],
            r0["SC"],
            r0["VCSC"],
            r0["consensus_status"],
        )
        if PRELIM_ONLY:
            logger.warning("PRELIM_ONLY=1 — exit.")
            return

    preds: list[str] = []
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, tx in enumerate(tqdm(texts, desc="semantic_entropy_vcsc")):
        row = run_one(client, tx, fewshot=fewshot, prelim=False)
        preds.append(row["pred_for_eval"])
        rows.append({"i": i, "gold": y_true[i], **{k: v for k, v in row.items() if k != "traces"}, "traces": row["traces"]})

    dt = time.perf_counter() - t0
    eval_mask = [p not in ("confusing", "error") for p in preds]
    yt_e = [y_true[i] for i in range(len(y_true)) if eval_mask[i]]
    pr_e = [preds[i] for i in range(len(preds)) if eval_mask[i]]
    acc = float(accuracy_score(yt_e, pr_e)) if yt_e else 0.0
    cr_d, cr_t = pec.sklearn_classification_reports(yt_e, pr_e) if yt_e else ({}, "")

    pec.save_phase(
        run_dir,
        "full",
        metrics={
            "accuracy_excl_confusing": acc,
            "n": len(texts),
            "n_confusing": sum(1 for p in preds if p == "confusing"),
            "n_error": sum(1 for p in preds if p == "error"),
            "mean_VCSC": float(np.mean([r["VCSC"] for r in rows])),
            "mean_label_entropy": float(np.mean([r["label_entropy"] for r in rows])),
            "classification_report": cr_d,
            "classification_report_text": cr_t,
        },
        settings={
            "EXECUTOR_MODEL": EXECUTOR_MODEL,
            "M_TRACES": M_TRACES,
            "VCSC_LAMBDA": VCSC_LAMBDA,
            "EVAL_N": EVAL_N,
            "MIN_PER_CLASS": MIN_PER_CLASS,
        },
        predictions=rows,
        duration_seconds=dt,
        notes="High-T ensemble; SC and VCavg from label matching only; VCSC = λ*SC+(1-λ)*VCavg/100; no separate judge API call.",
    )
    if cr_t:
        print(cr_t)
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()
