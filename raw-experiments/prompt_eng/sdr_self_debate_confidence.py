"""
Experiment: Sequential self-debate for (label, confidence) + critique calibration.

Workflow:
1) Pass A: model outputs JSON {label, confidence, reasoning}
2) Pass B (forced critique): same model sees A, must find flaw, outputs revised JSON {label, confidence, reasoning}
3) If labels differ => "confusing" and exclude from evaluation; else keep label + (optionally) revised confidence.

Preliminary: print only (no saving). Full: save results.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

import prompt_eng_common as pec

EXPERIMENT_SLUG = "sdr_self_debate_confidence"

MODEL = os.getenv("MODEL", "gemini-3.1-flash-lite-preview")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
FULL_N = int(os.getenv("FULL_N", "40"))

SKIP_PRELIMINARY = pec.env_bool("SKIP_PRELIMINARY", False)
PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)

LABELS = ["background", "objective", "methods", "results", "conclusions"]


def _extract_json_obj(raw: str) -> dict[str, Any] | None:
    raw = raw or ""
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


@dataclass(frozen=True)
class PassOut:
    label: str
    confidence: float
    reasoning: str
    raw: str


def _parse_pass(raw: str) -> PassOut:
    obj = _extract_json_obj(raw) or {}
    lab = str(obj.get("label", "error")).strip().lower()
    if lab not in LABELS:
        lab = "error"
    conf = _clamp01(obj.get("confidence", 0.0))
    reasoning = str(obj.get("reasoning", "")).strip()
    return PassOut(label=lab, confidence=conf, reasoning=reasoning, raw=raw)


def pass_a(text: str, *, prelim: bool) -> PassOut:
    client = pec.get_openai_client(max_retries=pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES)
    labels_str = ", ".join(LABELS)
    system = (
        "You are a biomedical rhetorical-role classifier.\n"
        f"Allowed labels: [{labels_str}].\n"
        "Return ONLY strict JSON with keys: label, confidence, reasoning.\n"
        "confidence must be a number between 0.0 and 1.0.\n"
    )
    user = (
        "Classify this sentence.\n"
        f"Sentence: {text}\n"
        "Be concise and objective.\n"
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=None,
    )
    raw = resp.choices[0].message.content or ""
    return _parse_pass(raw)


def pass_b_critique(text: str, a: PassOut, *, prelim: bool) -> PassOut:
    client = pec.get_openai_client(max_retries=pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES)
    labels_str = ", ".join(LABELS)
    system = (
        "You are the same classifier, but now you must critique a prior answer.\n"
        f"Allowed labels: [{labels_str}].\n"
        "The first response often (but not always) contains flaws or overconfidence.\n"
        "Your job is to find the flaw and revise if needed.\n"
        "Return ONLY strict JSON with keys: label, confidence, reasoning.\n"
        "confidence must be 0.0..1.0.\n"
    )
    user = (
        "Here is a proposed answer A. It contains a hidden flaw. Find the flaw.\n"
        "Then produce a revised answer B.\n\n"
        f"Sentence: {text}\n\n"
        f"Answer A JSON:\n{a.raw}\n"
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.5,  # induce some variance during critique
        max_tokens=None,
    )
    raw = resp.choices[0].message.content or ""
    return _parse_pass(raw)


def run_one(text: str, true_label: str, *, prelim: bool) -> dict[str, Any]:
    a = pass_a(text, prelim=prelim)
    b = pass_b_critique(text, a, prelim=prelim)
    final = b
    confusing = a.label != b.label and a.label != "error" and b.label != "error"
    final_label = "confusing" if confusing else final.label
    return {
        "text": text[:500],
        "true": true_label,
        "a_label": a.label,
        "a_conf": a.confidence,
        "a_reasoning": a.reasoning[:900],
        "b_label": b.label,
        "b_conf": b.confidence,
        "b_reasoning": b.reasoning[:900],
        "pred": final_label,
        "confusing": confusing,
    }


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    eval_df, _, _, _ = pec.load_hf_pubmed_splits()

    subset = eval_df.sample(n=min(FULL_N, len(eval_df)), random_state=pec.DEFAULT_SEED).copy()
    texts = subset["text"].astype(str).tolist()
    y_true = subset["label_name"].astype(str).str.lower().tolist()

    if not SKIP_PRELIMINARY and texts:
        logger.info("=== PRELIMINARY (print only): n=1 model={} ===", MODEL)
        r0 = run_one(texts[0], y_true[0], prelim=True)
        logger.info("A: {} ({:.2f}) | B: {} ({:.2f}) | final={} | true={}", r0["a_label"], r0["a_conf"], r0["b_label"], r0["b_conf"], r0["pred"], r0["true"])
        logger.info("A reasoning (trunc): {}...", str(r0["a_reasoning"])[:220])
        logger.info("B reasoning (trunc): {}...", str(r0["b_reasoning"])[:220])
        if PRELIM_ONLY:
            logger.warning("PRELIM_ONLY=1 — stop after preliminary.")
            return

    logger.info("=== FULL: n={} model={} ===", len(texts), MODEL)
    t0 = time.perf_counter()
    records = [run_one(tx, tr, prelim=False) for tx, tr in zip(texts, y_true)]
    dt = time.perf_counter() - t0

    eval_true: list[str] = []
    eval_pred: list[str] = []
    n_confusing = 0
    for r in records:
        if r["pred"] == "confusing":
            n_confusing += 1
            continue
        eval_true.append(r["true"])
        eval_pred.append(r["pred"])

    acc = float(np.mean([t == p for t, p in zip(eval_true, eval_pred)])) if eval_true else 0.0
    cr_d, cr_t = pec.sklearn_classification_reports(eval_true, eval_pred)

    pec.save_phase(
        run_dir,
        "full",
        metrics={
            "accuracy": acc,
            "n_texts": len(texts),
            "n_eval": len(eval_true),
            "n_confusing": n_confusing,
            "classification_report": cr_d,
            "classification_report_text": cr_t,
        },
        settings={"MODEL": MODEL, "FULL_N": FULL_N, "BATCH_SIZE": BATCH_SIZE},
        predictions=records,
        duration_seconds=dt,
        notes="Pass A label+confidence, then forced critique Pass B; if labels differ => confusing exclude.",
    )

    print(cr_t)
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()

