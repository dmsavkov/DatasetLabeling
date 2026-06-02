# %% [markdown]
# ### Experiment: All-label confidence (structured Gemini) + softmax / entropy reliability
#
# Single call per sentence:
#   - Few-shot from train (balanced, like other prompt_eng scripts).
#   - Model returns confidence 0.0–1.0 for **every** label in the label set.
#   - Post-hoc: softmax over scores → `label_probs`, argmax → `pred`, entropy & max-prob → reliability.
#
# Eval: first **20** hardcoded `test_*` samples (armanc/pubmed-rct20k).
#
# Run: `uv run python raw-experiments/prompt_eng/dinco_2pass_gemini_structured.py`

# %%
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from google.genai import types
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

_PROMPT_ENG = Path(__file__).resolve().parent
if str(_PROMPT_ENG) not in sys.path:
    sys.path.insert(0, str(_PROMPT_ENG))

import prompt_eng_common as pec
import prompt_eng_gemini as peg
from gemma_top2_gemini_boolean_match import load_hardcoded_eval_rows

EXPERIMENT_SLUG = "dinco_alllabel_confidence_gemini"
MODEL = os.getenv("MODEL", os.getenv("SMART_MODEL", pec.DEFAULT_MODEL))
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "47"))
FEWSHOT_N = int(os.getenv("FEWSHOT_N", "8"))

SKIP_PRELIMINARY = pec.env_bool("SKIP_PRELIMINARY", False)
PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)

LABELS = list(pec.VALID_LABELS)
_MAX_ENTROPY_BITS = math.log2(len(LABELS)) if LABELS else 1.0

EXAMPLE_ALL_LABEL_CONF_JSON = json.dumps(
    {
        "evaluations": [
            {"label": "background", "confidence": 0.05},
            {"label": "objective", "confidence": 0.1},
            {"label": "methods", "confidence": 0.75},
            {"label": "results", "confidence": 0.08},
            {"label": "conclusions", "confidence": 0.02},
        ]
    },
    ensure_ascii=False,
)


class LabelConfidence(BaseModel):
    label: str
    confidence: float = Field(
        description="Probability between 0.0 and 1.0 that this label is the correct category for the sentence."
    )


class ConfidenceEvaluation(BaseModel):
    evaluations: list[LabelConfidence] = Field(
        description=f"Exactly one entry per label in {LABELS}, all labels must appear once."
    )


def _norm(s: str) -> str | None:
    v = (s or "").strip().lower()
    if v in ("conclusion", "concl"):
        v = "conclusions"
    if v == "method":
        v = "methods"
    if v == "result":
        v = "results"
    return v if v in LABELS else None


def _fewshot_block(train_df) -> str:
    tdf = train_df.copy()
    tdf["label_name"] = tdf["label"].apply(pec.label_name_from_value)
    fs = pec.sample_balanced_train_fewshot(tdf, FEWSHOT_N, label_col="label_name", seed=pec.DEFAULT_SEED)
    return pec.format_fewshot_block(fs)


def _system_instruction(fewshot: str) -> str:
    labs = peg.labels_csv()
    return (
        f"Task: PubMed-RCT sentence — for each label in [{labs}], rate how likely that label is correct.\n\n"
        "Output one JSON object only. Key evaluations: array with exactly one object per allowed label.\n"
        "Each object: label (string from the list), confidence (float 0.0 to 1.0).\n"
        "Include all labels exactly once; confidences need not sum to 1 (they will be normalized later).\n\n"
        f"Valid JSON example:\n{EXAMPLE_ALL_LABEL_CONF_JSON}\n\n"
        f"Reference few-shot (style only):\n{fewshot}\n"
    )


def _generate_structured(
    client,
    *,
    model: str,
    contents: str,
    system_instruction: str,
    schema: type[BaseModel],
    temperature: float,
    max_retries: int,
    label: str,
):
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            cfg = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
                system_instruction=system_instruction,
            )
            return client.models.generate_content(model=model, contents=contents, config=cfg)
        except Exception as e:
            last = e
            logger.warning("{} attempt {}/{} failed: {}", label, attempt, max_retries, e)
            time.sleep(min(8.0, 0.5 * attempt))
    raise RuntimeError(f"{label} failed after {max_retries} attempts: {last}")


def _scores_vector(evaluations: list[LabelConfidence]) -> dict[str, float]:
    scores = {lab: 0.0 for lab in LABELS}
    for e in evaluations:
        lb = _norm(e.label)
        if lb:
            scores[lb] = max(0.0, min(1.0, float(e.confidence)))
    return scores


def _softmax_probs(scores: dict[str, float]) -> dict[str, float]:
    vec = np.array([scores[lab] for lab in LABELS], dtype=np.float64)
    vec = vec - float(np.max(vec))
    ex = np.exp(vec)
    s = float(ex.sum())
    if s <= 0:
        uniform = 1.0 / len(LABELS)
        return {lab: uniform for lab in LABELS}
    probs = ex / s
    return {lab: float(probs[i]) for i, lab in enumerate(LABELS)}


def _entropy_bits(probs: dict[str, float]) -> float:
    vec = np.array([probs[lab] for lab in LABELS], dtype=np.float64)
    vec = vec[vec > 0]
    if len(vec) == 0:
        return 0.0
    return float(-np.sum(vec * np.log2(vec)))


def _reliability_from_probs(probs: dict[str, float]) -> dict[str, float]:
    ent = _entropy_bits(probs)
    sorted_p = sorted(probs.values(), reverse=True)
    max_p = sorted_p[0] if sorted_p else 0.0
    margin = (sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else max_p
    # High max prob + low entropy → more reliable; scale entropy to [0,1] via max entropy log2(K)
    certainty_from_entropy = 1.0 - (ent / _MAX_ENTROPY_BITS) if _MAX_ENTROPY_BITS > 0 else 0.0
    certainty_from_entropy = max(0.0, min(1.0, certainty_from_entropy))
    return {
        "label_entropy_bits": round(ent, 4),
        "label_entropy_normalized": round(certainty_from_entropy, 4),
        "softmax_max_prob": round(float(max_p), 4),
        "softmax_margin_top1_top2": round(float(margin), 4),
        "reliability_combined": round(float(0.5 * max_p + 0.5 * certainty_from_entropy), 4),
    }


def evaluate_one(client, text: str, gold: str, *, fewshot: str, prelim: bool) -> dict:
    mr = pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES
    user = f"Sentence:\n{text}\n\nReturn one JSON object with confidence for every label."

    res = _generate_structured(
        client,
        model=MODEL,
        contents=user,
        system_instruction=_system_instruction(fewshot),
        schema=ConfidenceEvaluation,
        temperature=0.0,
        max_retries=mr,
        label="all_label_confidence",
    )
    if res.parsed is None:
        raise ValueError(f"Parse failed: {(res.text or '')[:500]}")

    evaluations = res.parsed.evaluations
    raw_scores = _scores_vector(evaluations)
    probs = _softmax_probs(raw_scores)
    rel = _reliability_from_probs(probs)
    pred = max(probs, key=probs.get)

    eval_rows = [{"label": lab, "raw_confidence": raw_scores[lab], "softmax_prob": probs[lab]} for lab in LABELS]

    return {
        "pred": pred,
        "raw_scores": raw_scores,
        "softmax_probs": probs,
        "evaluations": eval_rows,
        "is_correct": pred == gold,
        **rel,
        "response_raw": (res.text or "")[:3000],
    }


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    rows = load_hardcoded_eval_rows(EVAL_LIMIT)
    logger.info("Eval rows: {} (limit={})", len(rows), EVAL_LIMIT)

    _, _, train_df, _ = pec.load_hf_pubmed_splits()
    train_df = train_df.copy()
    fewshot = _fewshot_block(train_df)

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    client = peg.get_genai_client()

    if not SKIP_PRELIMINARY and rows:
        logger.info("=== PRELIMINARY ===")
        r0 = evaluate_one(client, rows[0]["text"], rows[0]["gold"], fewshot=fewshot, prelim=True)
        assert len(r0["evaluations"]) == len(LABELS)
        logger.info(
            "prelim {} pred={} gold={} max_p={:.3f} H={:.3f} rel={:.3f}",
            rows[0]["sample_id"],
            r0["pred"],
            rows[0]["gold"],
            r0["softmax_max_prob"],
            r0["label_entropy_bits"],
            r0["reliability_combined"],
        )
        if PRELIM_ONLY:
            return

    preds: list[str] = []
    out_rows: list[dict] = []
    t0 = time.perf_counter()
    for row in tqdm(rows, desc=EXPERIMENT_SLUG):
        try:
            r = evaluate_one(client, row["text"], row["gold"], fewshot=fewshot, prelim=False)
        except Exception as e:
            logger.exception("sample {} failed: {}", row["sample_id"], e)
            r = {
                "pred": "error",
                "evaluations": [],
                "softmax_max_prob": 0.0,
                "label_entropy_bits": 0.0,
                "reliability_combined": 0.0,
                "is_correct": False,
                "error": str(e),
            }
        preds.append(r["pred"])
        out_rows.append({**row, **r})
    dt = time.perf_counter() - t0

    golds = [row["gold"] for row in rows]
    mask = [p != "error" for p in preds]
    yt_e = [golds[i] for i in range(len(golds)) if mask[i]]
    pr_e = [preds[i] for i in range(len(preds)) if mask[i]]
    acc = float(accuracy_score(yt_e, pr_e)) if yt_e else 0.0
    cr_d, cr_t = pec.sklearn_classification_reports(yt_e, pr_e) if yt_e else ({}, "")

    ok_rows = [r for r in out_rows if r.get("pred") != "error"]
    pec.save_phase(
        run_dir,
        "full",
        metrics={
            "accuracy": acc,
            "n": len(rows),
            "n_error": sum(1 for p in preds if p == "error"),
            "mean_softmax_max_prob": float(np.mean([r["softmax_max_prob"] for r in ok_rows])) if ok_rows else 0.0,
            "mean_label_entropy_bits": float(np.mean([r["label_entropy_bits"] for r in ok_rows])) if ok_rows else 0.0,
            "mean_reliability_combined": float(np.mean([r["reliability_combined"] for r in ok_rows])) if ok_rows else 0.0,
            "classification_report": cr_d,
            "classification_report_text": cr_t,
        },
        settings={"MODEL": MODEL, "EVAL_LIMIT": EVAL_LIMIT, "FEWSHOT_N": FEWSHOT_N},
        predictions=out_rows,
        duration_seconds=dt,
        notes="Single pass: all-label confidences + softmax; pred=argmax(softmax); reliability from max prob and normalized entropy.",
    )
    if cr_t:
        print(cr_t)
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()
