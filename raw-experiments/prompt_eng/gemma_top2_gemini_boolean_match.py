# %% [markdown]
# ### Experiment: Gemma-4 top-2 shortlist → Gemini 3.1 per-label True/False match
#
# Pipeline (|labels| > 2 → always true for PubMed 5-class):
#   1. **Gemma 4** — pick the 2 most likely single labels.
#   2. **Gemini 3.1** — for each shortlisted label: "Does this label match the text?" → True or False only.
#   3. Resolve: exactly one True → that label; zero or multiple True → `confusing`.
#
# Eval: hardcoded `test_*` IDs (armanc/pubmed-rct20k), **first 20** only.
#
# Run: `uv run python raw-experiments/prompt_eng/gemma_top2_gemini_boolean_match.py`

# %%
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

_PROMPT_ENG = Path(__file__).resolve().parent
if str(_PROMPT_ENG) not in sys.path:
    sys.path.insert(0, str(_PROMPT_ENG))

import prompt_eng_common as pec
import prompt_eng_gemini as peg

EXPERIMENT_SLUG = "gemma_top2_gemini_boolean_match"
TOP2_MODEL = os.getenv("TOP2_MODEL", os.getenv("EXECUTOR_MODEL", "gemma-4-31b-it"))
MATCH_MODEL = os.getenv("MATCH_MODEL", os.getenv("SMART_MODEL", pec.DEFAULT_MODEL))
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "20"))

SKIP_PRELIMINARY = pec.env_bool("SKIP_PRELIMINARY", False)
PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)

# Hardcoded mistake-slice test IDs (armanc/pubmed-rct20k test split indices).
HARDCODED_TEST_SAMPLE_IDS: tuple[str, ...] = (
    "test_17516",
    "test_14870",
    "test_8133",
    "test_17934",
    "test_18599",
    "test_11745",
    "test_27024",
    "test_13952",
    "test_10035",
    "test_18679",
    "test_5379",
    "test_3875",
    "test_21665",
    "test_1627",
    "test_6718",
    "test_24798",
    "test_17367",
    "test_12888",
    "test_13953",
    "test_24822",
    "test_8544",
    "test_9673",
    "test_24350",
    "test_4536",
    "test_27410",
    "test_10510",
    "test_20579",
    "test_14175",
    "test_5002",
    "test_23692",
    "test_24457",
    "test_27232",
    "test_11647",
    "test_10690",
    "test_1353",
    "test_1317",
    "test_23218",
    "test_10406",
    "test_27282",
    "test_443",
    "test_14450",
    "test_19370",
    "test_21605",
    "test_25549",
    "test_7292",
    "test_21442",
    "test_16987",
)

_ID_RE = re.compile(r"^(?P<split>train|test)_(?P<idx>\d+)$", re.IGNORECASE)

EXAMPLE_TOP2_JSON = json.dumps(
    {"top_two": [{"label": "methods", "probability": 0.7}, {"label": "results", "probability": 0.2}]},
    ensure_ascii=False,
)
EXAMPLE_MATCH_JSON = json.dumps({"matches": True}, ensure_ascii=False)


def _norm(s: str) -> str | None:
    v = (s or "").strip().lower()
    if v in ("conclusion", "concl"):
        v = "conclusions"
    if v == "method":
        v = "methods"
    if v == "result":
        v = "results"
    return v if v in set(pec.VALID_LABELS) else None


def _parse_sample_id(sid: str) -> tuple[str, int]:
    m = _ID_RE.match(sid.strip())
    if not m:
        raise ValueError(f"Bad sample id {sid!r}; expected test_<int> or train_<int>")
    return m.group("split").lower(), int(m.group("idx"))


def load_hardcoded_eval_rows(limit: int) -> list[dict]:
    from datasets import load_dataset

    ids = list(HARDCODED_TEST_SAMPLE_IDS[:limit])
    name = os.getenv("PUBMED_HF_DATASET", "armanc/pubmed-rct20k")
    raw = load_dataset(name)
    tables = {split: raw[split] for split in ("train", "test") if split in raw}
    rows: list[dict] = []
    for sid in ids:
        split, idx = _parse_sample_id(sid)
        if split not in tables:
            raise KeyError(f"Split {split!r} missing in {name}")
        ds = tables[split]
        if idx < 0 or idx >= len(ds):
            raise IndexError(f"{sid}: index {idx} out of range (n={len(ds)})")
        r = ds[idx]
        gold = pec.label_name_from_value(r["label"])
        rows.append(
            {
                "sample_id": sid,
                "hf_split": split,
                "hf_index": idx,
                "text": str(r["text"]),
                "gold": str(gold).lower() if gold else "error",
            }
        )
    return rows


def _top2_sys() -> str:
    labs = peg.labels_csv()
    return (
        f"Task: PubMed-RCT sentence — choose the TWO most likely single labels from [{labs}].\n"
        "Output one JSON object only. Key top_two: array of exactly two objects with label and probability (0-1).\n"
        "Labels must be distinct.\n\n"
        f"Valid JSON example:\n{EXAMPLE_TOP2_JSON}\n"
    )


def _parse_top2(raw: str) -> list[str]:
    d = peg.extract_json_object(raw) or {}
    tt = d.get("top_two")
    out: list[str] = []
    if isinstance(tt, list):
        for it in tt[:2]:
            if isinstance(it, dict):
                lb = _norm(str(it.get("label", "")))
                if lb and lb not in out:
                    out.append(lb)
    if len(out) < 2:
        for key in ("label_a", "label_b"):
            lb = _norm(str(d.get(key, "")))
            if lb and lb not in out:
                out.append(lb)
    return out[:2]


def _pick_top2(client, text: str, *, all_labels: list[str], prelim: bool) -> tuple[list[str], str]:
    if len(all_labels) <= 2:
        return list(all_labels)[:2], ""
    mr = pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES
    raw = peg.generate_with_retries(
        client,
        model=TOP2_MODEL,
        user_text=f"Sentence:\n{text}\n\nReturn one JSON object only.",
        system_instruction=_top2_sys(),
        temperature=0.2,
        max_output_tokens=None,
        max_retries=mr,
        label="top2",
    )
    picked = _parse_top2(raw)
    if len(picked) < 2:
        picked = (picked + [lb for lb in all_labels if lb not in picked])[:2]
    return picked[:2], raw[:3000]


def _match_sys(label: str) -> str:
    return (
        f"Task: decide if the rhetorical label \"{label}\" applies to the sentence.\n"
        "Answer with one JSON object only. Key matches: boolean true or false (lowercase JSON booleans).\n"
        "Do not output any other keys or prose.\n\n"
        f"Valid JSON example when the label fits:\n{EXAMPLE_MATCH_JSON}\n"
        f"Valid JSON example when it does not:\n{json.dumps({'matches': False}, ensure_ascii=False)}\n"
    )


def _parse_matches(raw: str) -> bool | None:
    d = peg.extract_json_object(raw)
    if d is not None and "matches" in d:
        v = d["matches"]
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "yes"):
                return True
            if s in ("false", "no"):
                return False
    raw_l = (raw or "").strip().lower()
    if raw_l in ("true", "yes"):
        return True
    if raw_l in ("false", "no"):
        return False
    if "true" in raw_l and "false" not in raw_l:
        return True
    if "false" in raw_l and "true" not in raw_l:
        return False
    return None


def _verify_label(client, text: str, label: str, *, prelim: bool) -> dict:
    mr = pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES
    raw = peg.generate_with_retries(
        client,
        model=MATCH_MODEL,
        user_text=f"Sentence:\n{text}\n\nCandidate label: {label}\n\nReturn JSON only.",
        system_instruction=_match_sys(label),
        temperature=0.0,
        max_output_tokens=None,
        max_retries=mr,
        label=f"match_{label}",
    )
    parsed = _parse_matches(raw)
    return {"label": label, "matches": parsed, "raw": raw[:2000]}


def resolve_from_verdicts(verdicts: list[dict]) -> tuple[str, list[str]]:
    matched = [v["label"] for v in verdicts if v.get("matches") is True]
    if len(matched) == 1:
        return matched[0], matched
    if len(matched) == 0:
        return "confusing", matched
    return "confusing", matched


def run_one(client, row: dict, *, prelim: bool) -> dict:
    text = row["text"]
    all_labels = list(pec.VALID_LABELS)
    top2, top2_raw = _pick_top2(client, text, all_labels=all_labels, prelim=prelim)
    verdicts = [_verify_label(client, text, lb, prelim=prelim) for lb in top2]
    pred, matched = resolve_from_verdicts(verdicts)
    n_true = sum(1 for v in verdicts if v.get("matches") is True)
    return {
        "top_two": top2,
        "top2_raw": top2_raw,
        "verdicts": verdicts,
        "matched_labels": matched,
        "n_true": n_true,
        "pred": pred,
        "resolution": "single_match" if len(matched) == 1 else ("no_match" if len(matched) == 0 else "multi_match"),
    }


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    rows = load_hardcoded_eval_rows(EVAL_LIMIT)
    logger.info("Loaded {} hardcoded eval rows (limit={})", len(rows), EVAL_LIMIT)

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    client = peg.get_genai_client()

    if not SKIP_PRELIMINARY and rows:
        logger.info("=== PRELIMINARY ===")
        r0 = run_one(client, rows[0], prelim=True)
        assert len(r0["top_two"]) == 2
        logger.info("prelim sample={} pred={} gold={} matched={}", rows[0]["sample_id"], r0["pred"], rows[0]["gold"], r0["matched_labels"])
        if PRELIM_ONLY:
            return

    preds: list[str] = []
    out_rows: list[dict] = []
    t0 = time.perf_counter()
    for row in tqdm(rows, desc=EXPERIMENT_SLUG):
        r = run_one(client, row, prelim=False)
        preds.append(r["pred"])
        out_rows.append({**row, **r})
    dt = time.perf_counter() - t0

    golds = [row["gold"] for row in rows]
    mask = [p not in ("confusing", "error") for p in preds]
    yt_e = [golds[i] for i in range(len(golds)) if mask[i]]
    pr_e = [preds[i] for i in range(len(preds)) if mask[i]]
    acc = float(accuracy_score(yt_e, pr_e)) if yt_e else 0.0
    cr_d, cr_t = pec.sklearn_classification_reports(yt_e, pr_e) if yt_e else ({}, "")

    pec.save_phase(
        run_dir,
        "full",
        metrics={
            "accuracy_excl_confusing": acc,
            "n": len(rows),
            "n_confusing": sum(1 for p in preds if p == "confusing"),
            "n_multi_match": sum(1 for r in out_rows if r["resolution"] == "multi_match"),
            "n_no_match": sum(1 for r in out_rows if r["resolution"] == "no_match"),
            "classification_report": cr_d,
            "classification_report_text": cr_t,
        },
        settings={
            "TOP2_MODEL": TOP2_MODEL,
            "MATCH_MODEL": MATCH_MODEL,
            "EVAL_LIMIT": EVAL_LIMIT,
            "sample_ids": list(HARDCODED_TEST_SAMPLE_IDS[:EVAL_LIMIT]),
        },
        predictions=out_rows,
        duration_seconds=dt,
        notes="Gemma top-2 shortlist + Gemini boolean verifier per candidate; multi/no match → confusing.",
    )
    if cr_t:
        print(cr_t)
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()
