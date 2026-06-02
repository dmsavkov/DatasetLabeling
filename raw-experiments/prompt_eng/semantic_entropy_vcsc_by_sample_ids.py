# %% [markdown]
# ### VCSC (semantic entropy) on explicit PubMed-RCT sample IDs
#
# Resolves IDs like `test_17516` / `train_42` against **armanc/pubmed-rct20k** (positional index in that split).
# If you generated IDs from another pipeline, they usually match this dataset when `orig_split` + `orig_row` align.
#
# Default executor model for this script: **pec.DEFAULT_MODEL** (gemini-3.1-flash-lite-preview). Override with `EXECUTOR_MODEL`.
#
# Run from repo root:
#   uv run python raw-experiments/prompt_eng/semantic_entropy_vcsc_by_sample_ids.py
#
# Optional: `VCSC_SAMPLE_IDS=test_1,test_2` (comma-separated). `SKIP_PRELIMINARY=1` to skip smoke call.

# %%
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from loguru import logger
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

# Same-dir imports when launched from repo root
_PROMPT_ENG = Path(__file__).resolve().parent
if str(_PROMPT_ENG) not in sys.path:
    sys.path.insert(0, str(_PROMPT_ENG))

import prompt_eng_common as pec
import prompt_eng_gemini as peg

EXPERIMENT_SLUG = "semantic_entropy_vcsc_by_sample_ids"

# Mistake slice from user (armanc/pubmed-rct20k test indices).
DEFAULT_SAMPLE_IDS: tuple[str, ...] = (
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

_SKIP_PRELIM = pec.env_bool("SKIP_PRELIMINARY", False)
_PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)

_ID_RE = re.compile(r"^(?P<split>train|test)_(?P<idx>\d+)$", re.IGNORECASE)


def _parse_sample_id(sid: str) -> tuple[str, int]:
    sid = sid.strip()
    m = _ID_RE.match(sid)
    if not m:
        raise ValueError(
            f"Unrecognized sample id {sid!r}. Expected 'test_<int>' or 'train_<int>' for armanc/pubmed-rct20k."
        )
    return m.group("split").lower(), int(m.group("idx"))


def _load_split_tables() -> dict[str, pd.DataFrame]:
    """Raw HF splits (no prompt_eng subsampling)."""
    name = os.getenv("PUBMED_HF_DATASET", "armanc/pubmed-rct20k")
    logger.info("Loading HF dataset: {}", name)
    raw = load_dataset(name)
    out: dict[str, pd.DataFrame] = {}
    for split in ("train", "test"):
        if split not in raw:
            continue
        df = pd.DataFrame(raw[split])
        df["label_name"] = df["label"].apply(pec.label_name_from_value)
        out[split] = df
    return out


def resolve_rows(sample_ids: list[str], tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for sid in sample_ids:
        split, idx = _parse_sample_id(sid)
        if split not in tables:
            raise KeyError(f"No split {split!r} in dataset (have {list(tables.keys())})")
        df = tables[split]
        if idx < 0 or idx >= len(df):
            raise IndexError(f"{sid}: index {idx} out of range for split {split!r} (n={len(df)})")
        r = df.iloc[idx]
        rows.append(
            {
                "sample_id": sid,
                "hf_split": split,
                "hf_positional_index": idx,
                "abstract_id": r.get("abstract_id"),
                "sentence_id": r.get("sentence_id"),
                "text": str(r["text"]),
                "gold": str(r["label_name"]).lower(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    if not os.getenv("EXECUTOR_MODEL") and not os.getenv("CHEAP_MODEL"):
        os.environ["EXECUTOR_MODEL"] = pec.DEFAULT_MODEL

    import semantic_entropy_vcsc as svc

    raw_ids = os.getenv("VCSC_SAMPLE_IDS", "").strip()
    if raw_ids:
        sample_ids = [x.strip() for x in raw_ids.replace(";", ",").split(",") if x.strip()]
    else:
        sample_ids = list(DEFAULT_SAMPLE_IDS)

    tables = _load_split_tables()
    subset = resolve_rows(sample_ids, tables)

    # Default model for this run: Gemini 3.1 (user request). Still overridable via EXECUTOR_MODEL.
    if os.getenv("EXECUTOR_MODEL") is None and os.getenv("CHEAP_MODEL") is None:
        os.environ["EXECUTOR_MODEL"] = pec.DEFAULT_MODEL

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    _, _, train_df, _ = pec.load_hf_pubmed_splits()
    train_df = train_df.copy()
    train_df["label_name"] = train_df["label"].apply(pec.label_name_from_value)
    fewshot = svc._fewshot_block(train_df)
    client = peg.get_genai_client()

    texts = subset["text"].tolist()
    y_true = subset["gold"].tolist()
    ids = subset["sample_id"].tolist()

    if not _SKIP_PRELIM and texts:
        logger.info("=== PRELIMINARY (first row) ===")
        r0 = svc.run_one(client, texts[0], fewshot=fewshot, prelim=True)
        assert len(r0["traces"]) == svc.M_TRACES
        logger.info("prelim sample={} pred={} gold={} VCSC={:.3f}", ids[0], r0["pred_for_eval"], y_true[0], r0["VCSC"])
        if _PRELIM_ONLY:
            return

    preds: list[str] = []
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, tx in enumerate(tqdm(texts, desc="vcsc_by_sample_id")):
        row = svc.run_one(client, tx, fewshot=fewshot, prelim=False)
        preds.append(row["pred_for_eval"])
        base = {
            "i": i,
            "sample_id": ids[i],
            "hf_split": subset.iloc[i]["hf_split"],
            "hf_positional_index": int(subset.iloc[i]["hf_positional_index"]),
            "abstract_id": subset.iloc[i]["abstract_id"],
            "sentence_id": subset.iloc[i]["sentence_id"],
            "gold": y_true[i],
            **{k: v for k, v in row.items() if k != "traces"},
            "traces": row["traces"],
        }
        rows.append(base)
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
            "EXECUTOR_MODEL": svc.EXECUTOR_MODEL,
            "M_TRACES": svc.M_TRACES,
            "VCSC_LAMBDA": svc.VCSC_LAMBDA,
            "PUBMED_HF_DATASET": os.getenv("PUBMED_HF_DATASET", "armanc/pubmed-rct20k"),
            "n_sample_ids": len(sample_ids),
        },
        predictions=rows,
        duration_seconds=dt,
        notes="VCSC on explicit test_/train_ IDs = positional index in armanc/pubmed-rct20k split. Default executor=DEFAULT_MODEL (Gemini 3.1) unless EXECUTOR_MODEL set.",
    )
    if cr_t:
        print(cr_t)
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()
