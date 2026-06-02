# %% [markdown]
# ### Per-text classification via per-label boolean batches
# For each batch of 10 texts, iterate through each of the 5 labels:
# - 1 label, 10 texts, 1 request → JSON list of 10 booleans
# Aggregate 5×10 booleans, then for each text:
# - exactly one True → predicted label
# - multiple True → confusing case (report separately)
# - zero True → "none"
#
# IMPORTANT: each request prompt contains ONLY the currently evaluated label (no other label names).

# %%
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm

import prompt_eng_common as pec

EXPERIMENT_SLUG = "per_label_boolean_matrix_batch10"

LABELS = ["background", "objective", "methods", "results", "conclusions"]

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
FULL_N = int(os.getenv("FULL_N", "20"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))

SKIP_PRELIMINARY = True #pec.env_bool("SKIP_PRELIMINARY", False)
PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)

MODEL = "gemma-4-31b-it" # os.getenv("MODEL", pec.DEFAULT_MODEL)


def build_prompt(texts: list[str], *, target_label: str) -> tuple[str, str]:
    # Only include the currently evaluated label. No other label names anywhere.
    system_msg = (
        "You are performing binary verification for a single label.\n"
        f"Target label: {target_label!r}.\n"
        "For each sentence, answer whether the target label applies.\n"
        "Return ONLY a JSON list of N booleans (true/false) in the same order.\n"
        "No extra keys. No extra text.\n"
    )
    numbered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
    user_msg = f"N={len(texts)}. Provide booleans.\n\n{numbered}"
    return system_msg, user_msg


def parse_bool_list(raw: str, n: int) -> list[bool]:
    raw = raw or ""
    m = re.search(r"\[[\s\S]*\]", raw)
    data = None
    if m:
        try:
            data = json.loads(m.group(0))
        except Exception:
            data = None
    if data is None:
        try:
            data = json.loads(raw)
        except Exception:
            data = None

    out: list[bool] = []
    if isinstance(data, list):
        for v in data:
            if isinstance(v, bool):
                out.append(v)
            else:
                s = str(v).strip().lower()
                out.append(s in ("true", "t", "1", "yes"))

    if len(out) < n:
        out.extend([False] * (n - len(out)))
    return out[:n]


def verify_batch(texts: list[str], *, target_label: str, prelim: bool) -> list[bool]:
    client = pec.get_openai_client(max_retries=pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES)
    system_msg, user_msg = build_prompt(texts, target_label=target_label)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0.0,
        max_tokens=None,
    )
    dt = time.perf_counter() - t0
    raw = resp.choices[0].message.content or ""
    logger.info("Batch n={} completed in {:.2f}s (chars={})", len(texts), dt, len(raw))
    logger.debug("Raw (trunc): {}...", raw[:400])
    return parse_bool_list(raw, len(texts))


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    eval_df, _, _, _ = pec.load_hf_pubmed_splits()

    subset = eval_df.sample(n=min(FULL_N, len(eval_df)), random_state=pec.DEFAULT_SEED).copy()
    texts = subset["text"].astype(str).tolist()
    y_true_labels = subset["label_name"].astype(str).str.lower().tolist()

    # --- Preliminary: first batch only, print only ---
    if not SKIP_PRELIMINARY and texts:
        logger.info("=== PRELIMINARY (print only): labels={} batch_size={} ===", LABELS, BATCH_SIZE)
        pre_texts = texts[: min(BATCH_SIZE, len(texts))]
        pre_true = y_true_labels[: len(pre_texts)]

        pre_matrix: list[list[bool]] = []
        for lab in LABELS:
            preds = verify_batch(pre_texts, target_label=lab, prelim=True)
            assert len(preds) == len(pre_texts), "Preliminary parse length mismatch"
            pre_matrix.append(preds)

        for i in range(len(pre_texts)):
            selected = [LABELS[j] for j in range(len(LABELS)) if pre_matrix[j][i]]
            outcome = "none" if not selected else ("confusing" if len(selected) > 1 else selected[0])
            logger.info("pre[{}] true={} selected={} -> {}", i, pre_true[i], selected, outcome)
        if PRELIM_ONLY:
            logger.warning("PRELIM_ONLY=1 — stop after preliminary.")
            return

    # --- Full: async label × batches ---
    batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    logger.info("=== FULL: n={} batches={} labels={} ===", len(texts), len(batches), LABELS)

    bool_matrix: list[list[bool]] = [[False] * len(texts) for _ in range(len(LABELS))]
    t1 = time.perf_counter()

    def work(label_idx: int, label_name: str, batch_idx: int, batch_texts: list[str]) -> tuple[int, int, list[bool]]:
        preds = verify_batch(batch_texts, target_label=label_name, prelim=False)
        return label_idx, batch_idx, preds

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {}
        for label_idx, label_name in enumerate(LABELS):
            for batch_idx, batch_texts in enumerate(batches):
                futs[ex.submit(work, label_idx, label_name, batch_idx, batch_texts)] = (label_idx, batch_idx)

        for fut in tqdm(as_completed(futs), total=len(futs), desc="label×batches"):
            label_idx, batch_idx, preds = fut.result()
            start = batch_idx * BATCH_SIZE
            for j, v in enumerate(preds):
                text_idx = start + j
                if text_idx < len(texts):
                    bool_matrix[label_idx][text_idx] = bool(v)

    dt_full = time.perf_counter() - t1

    pred_labels: list[str] = []
    confusing_cases: list[dict] = []
    for i in range(len(texts)):
        selected = [LABELS[j] for j in range(len(LABELS)) if bool_matrix[j][i]]
        if len(selected) == 0:
            pred_labels.append("none")
        elif len(selected) == 1:
            pred_labels.append(selected[0])
        else:
            pred_labels.append("confusing")
            confusing_cases.append(
                {
                    "i": i,
                    "true": y_true_labels[i],
                    "selected_true_labels": selected,
                    "text": texts[i][:400],
                }
            )

    eval_true: list[str] = []
    eval_pred: list[str] = []
    for t, p in zip(y_true_labels, pred_labels):
        if p != "confusing":
            eval_true.append(t)
            eval_pred.append(p)

    acc = float(accuracy_score(eval_true, eval_pred))
    cr_d, cr_t = pec.sklearn_classification_reports(eval_true, eval_pred)

    logger.info(
        "Accuracy (excluding confusing) {:.2%} | confusing_cases {} | wall {:.2f}s",
        acc,
        len(confusing_cases),
        dt_full,
    )

    pec.save_phase(
        run_dir,
        "full",
        metrics={
            "accuracy": acc,
            "n_texts": len(texts),
            "n_confusing": len(confusing_cases),
            "labels": LABELS,
            "classification_report": cr_d,
            "classification_report_text": cr_t,
        },
        settings={
            "MODEL": MODEL,
            "BATCH_SIZE": BATCH_SIZE,
            "FULL_N": FULL_N,
            "MAX_WORKERS": MAX_WORKERS,
        },
        predictions=[
            {
                "i": i,
                "true": y_true_labels[i],
                "pred": pred_labels[i],
                "per_label": {LABELS[j]: bool_matrix[j][i] for j in range(len(LABELS))},
                "text": texts[i][:300],
            }
            for i in range(len(texts))
        ]
        + (confusing_cases[:200] if confusing_cases else []),
        duration_seconds=dt_full,
        notes="Five labels; one request per (label × 10 texts); aggregate booleans; multi-true => confusing.",
    )

    print(classification_report(eval_true, eval_pred))
    if confusing_cases:
        logger.info("Sample confusing cases (first 5):")
        for c in confusing_cases[:5]:
            logger.info("confusing[{}] true={} selected={}", c["i"], c["true"], c["selected_true_labels"])
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()
