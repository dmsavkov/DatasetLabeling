# %% [markdown]
# ### Experiment: Discrete confidence rubric (1–4) + perfect16 few-shot pool
#
# Same pipeline as `rubric_confidence_gemini.py`; few-shot comes from
# `data/artifacts/fewshot/perfect16_google_studio.json` instead of balanced HF sampling.
#
# Rubric calibration (3 hardcoded rubric-boundary examples) is unchanged.
#
# Run: `uv run python raw-experiments/prompt_eng/rubric_confidence_gemini_perfect16.py`

# %%
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

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

EXPERIMENT_SLUG = "rubric_confidence_gemini_perfect16"
MODEL = os.getenv("MODEL", os.getenv("SMART_MODEL", pec.DEFAULT_MODEL))
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "47"))
RUBRIC_LEVELS = 4

DEFAULT_FEWSHOT_PATH = (
    pec.PROJECT_ROOT / "data" / "artifacts" / "fewshot" / "perfect16_google_studio.json"
)
FEWSHOT_JSON_PATH = Path(os.getenv("FEWSHOT_JSON_PATH", str(DEFAULT_FEWSHOT_PATH))).resolve()

SKIP_PRELIMINARY = pec.env_bool("SKIP_PRELIMINARY", False)
PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)

LABELS = list(pec.VALID_LABELS)
RubricLevel = Literal[1, 2, 3, 4]

RUBRIC_CALIBRATION_EXAMPLES: tuple[dict, ...] = (
    {
        "sentence": (
            "This narrative review outlines recent trials of checkpoint inhibitors in solid tumors "
            "and how they changed standard care."
        ),
        "output": {"label": "background", "confidence_rubric": 3},
        "note": "background vs objective — both plausible; background slightly better.",
    },
    {
        "sentence": (
            "Mean HbA1c fell from 8.1% to 6.9% in the treatment arm versus 8.0% to 7.8% in placebo (p=0.02)."
        ),
        "output": {"label": "results", "confidence_rubric": 2},
        "note": "results vs methods — outcome reporting vs procedure framing.",
    },
    {
        "sentence": "See supplementary Table S3.",
        "output": {"label": "abstain", "confidence_rubric": 1},
        "note": "Fragment — no rhetorical section can be inferred.",
    },
)

EXAMPLE_OUTPUT_JSON = json.dumps(
    {"label": "methods", "confidence_rubric": 4},
    ensure_ascii=False,
)


class RubricClassification(BaseModel):
    label: str = Field(
        description=(
            "Single PubMed-RCT section label, or exactly 'abstain' when confidence_rubric is 1. "
            f"Allowed labels: {LABELS} or abstain."
        )
    )
    confidence_rubric: int = Field(
        ge=1,
        le=4,
        description=(
            "Discrete certainty rubric: 4=certain single label; 3=best label but one alternate plausible; "
            "2=two+ labels plausible, pick most likely; 1=unclassifiable, use abstain."
        ),
    )


def _norm(s: str) -> str | None:
    v = (s or "").strip().lower()
    if v in ("conclusion", "concl"):
        v = "conclusions"
    if v == "method":
        v = "methods"
    if v == "result":
        v = "results"
    if v in ("abstain", "none", "unclear", "unknown"):
        return "abstain"
    return v if v in LABELS else None


def _norm_rubric(value: object) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= RUBRIC_LEVELS else None


def _load_perfect16_pool(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Few-shot pool not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Few-shot pool must be a non-empty JSON array: {path}")
    out: list[dict] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"Few-shot entry {i} is not an object")
        text = str(row.get("text", "")).strip()
        lab = _norm(str(row.get("correct_label", row.get("label", ""))))
        if not text or lab is None or lab == "abstain":
            raise ValueError(f"Few-shot entry {i} needs text and valid correct_label")
        out.append(
            {
                "text": text,
                "correct_label": lab,
                "difficulty": str(row.get("difficulty", "")).strip(),
                "rationale": str(row.get("rationale", "")).strip(),
            }
        )
    return out


def _perfect16_fewshot_block(path: Path) -> str:
    pool = _load_perfect16_pool(path)
    lines: list[str] = []
    for i, ex in enumerate(pool, start=1):
        diff = ex["difficulty"] or "n/a"
        lines.append(f"Example {i} [label={ex['correct_label']}, difficulty={diff}]:")
        lines.append(f"  Sentence: {ex['text']}")
        if ex["rationale"]:
            lines.append(f"  (Teacher note — do not output: {ex['rationale']})")
    return "\n".join(lines)


def _calibration_block() -> str:
    lines = [
        "The three examples below show how to apply the rubric (output shape only — do not copy labels blindly).",
    ]
    for i, ex in enumerate(RUBRIC_CALIBRATION_EXAMPLES, start=1):
        out = ex["output"]
        lines.append(f"Calibration {i}:")
        lines.append(f"  Sentence: {ex['sentence']}")
        lines.append(f"  Output JSON: {json.dumps(out, ensure_ascii=False)}")
        lines.append(f"  (Rationale for you only: {ex['note']})")
    return "\n".join(lines)


def _system_instruction(perfect16_fewshot: str) -> str:
    labs = peg.labels_csv()
    rubric_table = "\n".join(
        [
            "| Level | When to use | Label field |",
            "|-------|-------------|-------------|",
            "| **4** | Exactly one label fits; no other label is plausible. | That label |",
            "| **3** | One label is best, but **one** other label could also fit. | Best label |",
            "| **2** | **Two or more** labels are genuinely plausible; pick the single most likely. | Most likely label |",
            "| **1** | Too fragmentary, meta, or vague to assign any section label. | `abstain` only |",
        ]
    )
    label_blurbs = "\n".join(
        [
            "- **background**: context, prior work, or motivation (not the study aim).",
            "- **objective**: study aim, purpose, or hypothesis.",
            "- **methods**: design, participants, procedures, or interventions.",
            "- **results**: findings, outcomes, or observed data.",
            "- **conclusions**: interpretation, implications, or summary takeaway.",
        ]
    )
    return (
        "## Task\n"
        "Classify one PubMed-RCT sentence into exactly one rhetorical section label.\n"
        "Also assign a **confidence_rubric** integer 1–4 (definitions below).\n\n"
        "## Allowed labels\n"
        f"[{labs}] plus `abstain` (only when rubric = 1).\n\n"
        "## Label hints\n"
        f"{label_blurbs}\n"
        "## Confidence rubric (required)\n"
        f"{rubric_table}\n\n"
        "## Output rules\n"
        "- Return **one JSON object only** (no markdown, no reasoning, no extra keys).\n"
        "- Keys: `label` (string), `confidence_rubric` (integer 1–4).\n"
        "- If rubric is 1, `label` must be `abstain`. If rubric is 2–4, `label` must be one of the five section labels.\n"
        "- Do not output probabilities, explanations, or alternate labels.\n\n"
        f"## Valid JSON shape\n{EXAMPLE_OUTPUT_JSON}\n\n"
        "## Reference few-shot (perfect16 pool; treat as rubric 4 unless obviously ambiguous)\n"
        f"{perfect16_fewshot}\n\n"
        "## Rubric calibration (study these boundaries)\n"
        f"{_calibration_block()}\n"
    )


def _generate_structured(
    client,
    *,
    model: str,
    contents: str,
    system_instruction: str,
    temperature: float,
    max_retries: int,
    label: str,
):
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            cfg = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RubricClassification,
                temperature=temperature,
                system_instruction=system_instruction,
            )
            return client.models.generate_content(model=model, contents=contents, config=cfg)
        except Exception as e:
            last = e
            logger.warning("{} attempt {}/{} failed: {}", label, attempt, max_retries, e)
            time.sleep(min(8.0, 0.5 * attempt))
    raise RuntimeError(f"{label} failed after {max_retries} attempts: {last}")


def _coerce_parsed(parsed: RubricClassification | None, raw_text: str) -> tuple[str | None, int | None]:
    if parsed is not None:
        lab = _norm(parsed.label)
        rub = _norm_rubric(parsed.confidence_rubric)
        return lab, rub
    d = peg.extract_json_object(raw_text or "")
    if not d:
        return None, None
    return _norm(str(d.get("label", ""))), _norm_rubric(d.get("confidence_rubric"))


def _validate_rubric_label(label: str | None, rubric: int | None) -> tuple[str, int]:
    if rubric is None:
        rubric = 2
    if label is None:
        label = "error"
    if rubric == 1:
        return "abstain", 1
    if label == "abstain":
        return "error", rubric
    if label not in LABELS:
        return "error", rubric
    return label, rubric


def evaluate_one(client, text: str, gold: str, *, system_instruction: str, prelim: bool) -> dict:
    mr = pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES
    user = f"Sentence:\n{text}\n\nReturn one JSON object: label and confidence_rubric."
    res = _generate_structured(
        client,
        model=MODEL,
        contents=user,
        system_instruction=system_instruction,
        temperature=0.0,
        max_retries=mr,
        label="rubric_classify",
    )
    raw_lab, raw_rubric = _coerce_parsed(res.parsed, res.text or "")
    pred, rubric = _validate_rubric_label(raw_lab, raw_rubric)
    rubric_scaled = rubric / float(RUBRIC_LEVELS)

    in_eval = gold in LABELS and pred in LABELS
    is_correct = bool(in_eval and gold == pred) if in_eval else False
    if pred == "abstain":
        is_correct = False

    return {
        "pred": pred,
        "confidence_rubric": rubric,
        "confidence_rubric_scaled": round(rubric_scaled, 4),
        "is_correct": is_correct,
        "response_raw": (res.text or "")[:3000],
    }


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    rows = load_hardcoded_eval_rows(EVAL_LIMIT)
    logger.info("Eval rows: {} (limit={})", len(rows), EVAL_LIMIT)

    pool = _load_perfect16_pool(FEWSHOT_JSON_PATH)
    logger.info("Few-shot pool: {} examples from {}", len(pool), FEWSHOT_JSON_PATH)
    perfect16 = _perfect16_fewshot_block(FEWSHOT_JSON_PATH)
    system_instruction = _system_instruction(perfect16)

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    client = peg.get_genai_client()

    if not SKIP_PRELIMINARY and rows:
        logger.info("=== PRELIMINARY ===")
        r0 = evaluate_one(
            client, rows[0]["text"], rows[0]["gold"], system_instruction=system_instruction, prelim=True
        )
        assert r0["confidence_rubric"] in (1, 2, 3, 4)
        logger.info(
            "prelim {} pred={} rubric={} gold={} ok={}",
            rows[0]["sample_id"],
            r0["pred"],
            r0["confidence_rubric"],
            rows[0]["gold"],
            r0["is_correct"],
        )
        if PRELIM_ONLY:
            return

    preds: list[str] = []
    out_rows: list[dict] = []
    t0 = time.perf_counter()
    for row in tqdm(rows, desc=EXPERIMENT_SLUG):
        try:
            r = evaluate_one(
                client, row["text"], row["gold"], system_instruction=system_instruction, prelim=False
            )
        except Exception as e:
            logger.exception("sample {} failed: {}", row["sample_id"], e)
            r = {
                "pred": "error",
                "confidence_rubric": 1,
                "confidence_rubric_scaled": 0.0,
                "is_correct": False,
                "error": str(e),
            }
        preds.append(r["pred"])
        out_rows.append({**row, **r})
    dt = time.perf_counter() - t0

    golds = [row["gold"] for row in rows]
    mask = [p not in ("error", "abstain") for p in preds]
    yt_e = [golds[i] for i in range(len(golds)) if mask[i]]
    pr_e = [preds[i] for i in range(len(preds)) if mask[i]]
    acc = float(accuracy_score(yt_e, pr_e)) if yt_e else 0.0
    n_abstain = sum(1 for p in preds if p == "abstain")
    cr_d, cr_t = pec.sklearn_classification_reports(yt_e, pr_e) if yt_e else ({}, "")

    rubrics = [r["confidence_rubric"] for r in out_rows if r.get("pred") != "error"]
    pec.save_phase(
        run_dir,
        "full",
        metrics={
            "accuracy_excluding_abstain_error": acc,
            "n": len(rows),
            "n_abstain": n_abstain,
            "n_error": sum(1 for p in preds if p == "error"),
            "rubric_histogram": {str(k): rubrics.count(k) for k in range(1, RUBRIC_LEVELS + 1)},
            "mean_confidence_rubric": float(sum(rubrics) / len(rubrics)) if rubrics else 0.0,
            "classification_report": cr_d,
            "classification_report_text": cr_t,
        },
        settings={
            "MODEL": MODEL,
            "EVAL_LIMIT": EVAL_LIMIT,
            "FEWSHOT_JSON_PATH": str(FEWSHOT_JSON_PATH),
            "fewshot_n": len(pool),
            "RUBRIC_LEVELS": RUBRIC_LEVELS,
        },
        predictions=out_rows,
        duration_seconds=dt,
        notes=(
            "Discrete rubric 1–4; abstain only at level 1. "
            f"Few-shot: perfect16 pool ({len(pool)} examples) + {len(RUBRIC_CALIBRATION_EXAMPLES)} rubric calibration."
        ),
    )
    if cr_t:
        print(cr_t)
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()
