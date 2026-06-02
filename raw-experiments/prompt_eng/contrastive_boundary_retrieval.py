"""
Experiment: Stratified Contrastive Retrieval + Cross-Encoder rerank + contrastive final judge

Workflow (cheap → robust):
1) First pass (Gemma 31B): multi-label *candidates* per text (batched).
2) For each candidate label: retrieve up to K examples from that label (embedding kNN),
   then rerank with a Cross-Encoder and keep TOP_M examples per label.
3) Second pass (Gemini flash-lite): contrast candidates using retrieved boundary references,
   output final label or "confusing" (batched).

Notes:
- Preliminary phase prints only (and asserts basic shape), no saving.
- Full phase saves under results/raw/prompt_eng/<experiment>/<run_id>/.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

import prompt_eng_common as pec

EXPERIMENT_SLUG = "contrastive_boundary_retrieval"

# Rules defaults
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))

PRELIM_N = int(os.getenv("PRELIM_N", str(BATCH_SIZE)))
# Requested default: 30 eval samples (sampled with DEFAULT_SEED)
FULL_N = int(os.getenv("FULL_N", "30"))

RETRIEVE_PER_LABEL = int(os.getenv("RETRIEVE_PER_LABEL", "5"))  # <=5 per your note
TOP_M_PER_LABEL = int(os.getenv("TOP_M_PER_LABEL", "2"))  # "2 per label"

# IMPORTANT: when using the OpenAI client against Google's OpenAI-compatible endpoint,
# pass the model name as-is (do NOT prefix with "openai/").
FIRST_PASS_MODEL = os.getenv("FIRST_PASS_MODEL", "gemma-4-31b-it")  # user requested default
SECOND_PASS_MODEL = os.getenv("SECOND_PASS_MODEL", "gemini-3.1-flash-lite-preview")

SKIP_PRELIMINARY = pec.env_bool("SKIP_PRELIMINARY", False)
PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)

LABELS = ["background", "objective", "methods", "results", "conclusions"]
FINAL_LABELS = LABELS + ["confusing"]


def _extract_json_list(raw: str) -> list | None:
    m = re.search(r"\[[\s\S]*\]", raw or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _safe_lower(x: Any) -> str:
    return str(x).strip().lower()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (d,), b: (n,d)
    a_n = a / (np.linalg.norm(a) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return b_n @ a_n


@dataclass(frozen=True)
class CandidatePack:
    text: str
    true_label: str
    candidates: list[str]  # first-pass candidates (ordered)


def first_pass_batch(texts: list[str], *, prelim: bool) -> list[list[str]]:
    """
    Returns list-of-candidates per text (length <=3), each candidate in LABELS.
    """
    client = pec.get_openai_client(max_retries=pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES)
    labels_str = ", ".join(LABELS)
    # Add JSON-template force example
    json_example = (
        '[{"candidates": ["background", "objective"]}, '
        '{"candidates": ["methods", "results", "conclusions"]}]'
    )
    system_msg = (
        "You are doing multi-label candidate generation for rhetorical roles.\n"
        f"Allowed labels: [{labels_str}].\n"
        "For each sentence, return a JSON list of N objects. Each object has:\n"
        '- "candidates": an ordered list of 1-3 labels from the allowed set (most likely first).\n'
        f"Format: Example for N=2:\n{json_example}\n"
        "Return ONLY JSON."
    )
    numbered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
    user_msg = f"N={len(texts)}. Provide candidates for each.\n\n{numbered}"

    resp = client.chat.completions.create(
        model=FIRST_PASS_MODEL,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0.0,
        max_tokens=None,
    )
    logger.critical(resp.choices[0].message.content)
    raw = resp.choices[0].message.content or ""
    logger.debug("First pass raw (trunc): {}...", raw[:600])
    data = _extract_json_list(raw)
    out: list[list[str]] = []
    if isinstance(data, list):
        for item in data:
            cands: list[str] = []
            if isinstance(item, dict):
                cands_raw = item.get("candidates", [])
                if isinstance(cands_raw, list):
                    cands = [_safe_lower(x) for x in cands_raw]
            cands = [c for c in cands if c in LABELS]
            out.append(cands[:3] if cands else [])
    if len(out) < len(texts):
        out.extend([[]] * (len(texts) - len(out)))
    return out[: len(texts)]


def load_cross_encoder():
    # Minimal: generic cross-encoder; swap via env if desired.
    model_name = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder: {}", model_name)
    return CrossEncoder(model_name)


def fit_umap_10d(train_embeddings: np.ndarray):
    """
    Fit UMAP to 10 dims on train embeddings, then use it to transform query embeddings.
    This is just for retrieval geometry (KNN), not for the cross-encoder stage.
    """
    import umap

    reducer = umap.UMAP(
        n_components=10,
        random_state=pec.DEFAULT_SEED,
        metric="cosine",
        n_neighbors=int(os.getenv("UMAP_N_NEIGHBORS", "15")),
        min_dist=float(os.getenv("UMAP_MIN_DIST", "0.1")),
    )
    logger.info("Fitting UMAP(10d) on train embeddings: shape={}", train_embeddings.shape)
    t0 = time.perf_counter()
    reduced = reducer.fit_transform(train_embeddings)
    logger.info("UMAP fit_transform done in {:.2f}s -> shape={}", time.perf_counter() - t0, reduced.shape)
    return reducer, reduced.astype(np.float32)


def retrieve_and_rerank(
    *,
    query_text: str,
    candidate_label: str,
    train_df: pd.DataFrame,
    train_embeddings_10d: np.ndarray,
    reducer,
    ce,
    k: int,
    top_m: int,
) -> list[dict]:
    """
    Return TOP_M examples for this label: [{"text":..., "score":..., "idx":...}, ...]
    """
    # train_df here is the sampled train pool with a 0..N-1 index aligned to embeddings arrays.
    subset_pos = train_df.index[train_df["label_name"] == candidate_label].to_numpy()
    if subset_pos.size == 0:
        return []

    # Explicit fastembed usage (pec.embed_texts uses fastembed)
    q_emb = pec.embed_texts([query_text])[0]
    q_10d = reducer.transform(np.asarray([q_emb], dtype=np.float32))[0].astype(np.float32)
    sims = _cosine_sim(q_10d, train_embeddings_10d[subset_pos])
    top_k_local = np.argsort(-sims)[: min(k, len(sims))]
    chosen_pos = subset_pos[top_k_local]
    chosen_texts = train_df.loc[chosen_pos, "text"].astype(str).tolist()

    # Cross-encoder scores query/example pairs
    pairs = [(query_text, t) for t in chosen_texts]
    scores = ce.predict(pairs)
    scored = []
    for j, (pos, t, s) in enumerate(zip(chosen_pos.tolist(), chosen_texts, scores)):
        orig_idx = None
        if "orig_idx" in train_df.columns:
            try:
                orig_idx = int(train_df.loc[pos, "orig_idx"])
            except Exception:
                orig_idx = None
        scored.append(
            {
                "pos": int(pos),
                "orig_idx": orig_idx,
                "text": t,
                "ce_score": float(s),
                "embed_sim": float(sims[j]),
            }
        )
    scored.sort(key=lambda r: r["ce_score"], reverse=True)
    return scored[:top_m]


def second_pass_batch(
    packs: list[CandidatePack],
    *,
    retrieved: list[dict[str, list[dict]]],
    prelim: bool,
) -> list[str]:
    """
    packs[i] has candidates; retrieved[i] maps label -> top examples.
    Returns final label per pack (one of FINAL_LABELS).
    """
    client = pec.get_openai_client(max_retries=pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES)
    labels_str = ", ".join(LABELS)
    # Example template for the output, to force format.
    json_example = '[{"final_label": "background"}, {"final_label": "confusing"}]'
    system_msg = (
        "You are classifying a biomedical sentence into a single rhetorical role.\n"
        f"Allowed labels: [{labels_str}] and you may output 'confusing' if insufficient evidence.\n"
        "You will receive, for each item, candidate labels with reference examples per label.\n"
        "Task: Compare the target sentence against references and pick the best label, or 'confusing'.\n"
        f"Format: Your response should be a JSON list like this:\n{json_example}\n"
        "Return ONLY a JSON list of N objects: {\"final_label\": <label|confusing>}."
    )

    # Build a single batched payload to keep it cheap.
    blocks: list[str] = []
    for i, pack in enumerate(packs):
        cand = pack.candidates[:2] if pack.candidates else []
        # If first pass gave nothing, still allow judge to pick or confusing.
        cand = cand or LABELS[:2]
        lines = [f"Item {i+1}:", f"Target: {pack.text}"]
        lines.append(f"Candidates: {', '.join(cand)}")
        for lab in cand:
            exs = retrieved[i].get(lab, [])
            lines.append(f"Reference points for {lab}:")
            if not exs:
                lines.append("- (none)")
            for j, ex in enumerate(exs[:TOP_M_PER_LABEL]):
                lines.append(f"- {j+1}. {ex['text']}")
        blocks.append("\n".join(lines))
    user_msg = f"N={len(packs)}. Decide final labels.\n\n" + "\n\n---\n\n".join(blocks)

    resp = client.chat.completions.create(
        model=SECOND_PASS_MODEL,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0.0,
        max_tokens=None,
    )
    raw = resp.choices[0].message.content or ""
    logger.debug("Second pass raw (trunc): {}...", raw[:600])
    data = _extract_json_list(raw)
    out: list[str] = []
    if isinstance(data, list):
        for item in data:
            lab = "confusing"
            if isinstance(item, dict):
                lab = _safe_lower(item.get("final_label", "confusing"))
            if lab not in FINAL_LABELS:
                lab = "confusing"
            out.append(lab)
    if len(out) < len(packs):
        out.extend(["confusing"] * (len(packs) - len(out)))
    return out[: len(packs)]


def run_pipeline(eval_df: pd.DataFrame, train_df: pd.DataFrame, *, n_rows: int, prelim: bool) -> tuple[list[dict], float]:
    """
    Returns: (prediction records, duration_seconds)
    """
    subset = eval_df.sample(n=min(n_rows, len(eval_df)), random_state=pec.DEFAULT_SEED).copy()
    texts = subset["text"].astype(str).tolist()
    true = subset["label_name"].astype(str).str.lower().tolist()

    # First pass in batches
    t0 = time.perf_counter()
    first_cands: list[list[str]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        c = first_pass_batch(batch, prelim=prelim)
        first_cands.extend(c)
        logger.info("First pass batch {}/{} done.", i // BATCH_SIZE + 1, (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE)
    first_cands = first_cands[: len(texts)]

    packs = [
        CandidatePack(text=tx, true_label=tr, candidates=cands) for tx, tr, cands in zip(texts, true, first_cands)
    ]

    # Retrieval pool embeddings (compute once)
    # Keep it small: sample a pool for speed
    pool_n = 100
    train_pool = train_df.sample(n=min(pool_n, len(train_df)), random_state=pec.DEFAULT_SEED).copy()
    # Align train pool rows 1:1 with embedding arrays; keep original index for debugging.
    train_pool = train_pool.reset_index(drop=False).rename(columns={"index": "orig_idx"})
    train_pool["label_name"] = train_pool["label"].apply(pec.label_name_from_value)
    logger.info("Embedding train pool with fastembed (n={})...", len(train_pool))
    train_emb = pec.embed_texts(train_pool["text"].astype(str).tolist())
    logger.info("Train pool embeddings computed: shape={}", train_emb.shape)
    reducer, train_emb_10d = fit_umap_10d(train_emb)

    ce = load_cross_encoder()

    # For each item: retrieve per candidate label
    retrieved: list[dict[str, list[dict]]] = []
    for pack in packs:
        labs = pack.candidates[:2] if pack.candidates else []
        labs = [l for l in labs if l in LABELS]
        labs = labs or LABELS[:2]
        per_label: dict[str, list[dict]] = {}
        for lab in labs:
            per_label[lab] = retrieve_and_rerank(
                query_text=pack.text,
                candidate_label=lab,
                train_df=train_pool,
                train_embeddings_10d=train_emb_10d,
                reducer=reducer,
                ce=ce,
                k=RETRIEVE_PER_LABEL,
                top_m=TOP_M_PER_LABEL,
            )
        retrieved.append(per_label)

    # Second pass in batches
    final_preds: list[str] = []
    for i in range(0, len(packs), BATCH_SIZE):
        batch_packs = packs[i : i + BATCH_SIZE]
        batch_ret = retrieved[i : i + BATCH_SIZE]
        preds = second_pass_batch(batch_packs, retrieved=batch_ret, prelim=prelim)
        final_preds.extend(preds)
        logger.info("Second pass batch {}/{} done.", i // BATCH_SIZE + 1, (len(packs) + BATCH_SIZE - 1) // BATCH_SIZE)

    dt = time.perf_counter() - t0

    records: list[dict] = []
    for i, (pack, pred) in enumerate(zip(packs, final_preds)):
        records.append(
            {
                "i": i,
                "true": pack.true_label,
                "pred": pred,
                "candidates": pack.candidates,
                "retrieved": retrieved[i],
                "text": pack.text[:500],
            }
        )
    return records, dt


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    eval_df, _, train_df, _ = pec.load_hf_pubmed_splits()

    # ----- Preliminary: prints + assert only -----
    if not SKIP_PRELIMINARY:
        logger.info("=== PRELIMINARY (prints only): n={} ===", PRELIM_N)
        records, dt = run_pipeline(eval_df, train_df, n_rows=PRELIM_N, prelim=True)
        logger.info("Prelim finished in {:.2f}s", dt)
        for r in records[: min(3, len(records))]:
            logger.info("pre[{}] true={} pred={} cands={}", r["i"], r["true"], r["pred"], r["candidates"])
        assert all(isinstance(r["pred"], str) for r in records), "Bad prelim prediction type"
        if PRELIM_ONLY:
            logger.warning("PRELIM_ONLY=1 — stop after preliminary.")
            return

    # ----- Full: save artifacts -----
    logger.info("=== FULL: n={} ===", FULL_N)
    records, dt_full = run_pipeline(eval_df, train_df, n_rows=FULL_N, prelim=False)
    y_true = [r["true"] for r in records]
    y_pred = [r["pred"] for r in records]

    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    cr_d, cr_t = pec.sklearn_classification_reports(y_true, y_pred)
    logger.info("Accuracy {:.2%} wall {:.2f}s", acc, dt_full)

    pec.save_phase(
        run_dir,
        "full",
        metrics={
            "accuracy": acc,
            "n_rows": len(records),
            "classification_report": cr_d,
            "classification_report_text": cr_t,
        },
        settings={
            "FIRST_PASS_MODEL": FIRST_PASS_MODEL,
            "SECOND_PASS_MODEL": SECOND_PASS_MODEL,
            "BATCH_SIZE": BATCH_SIZE,
            "RETRIEVE_PER_LABEL": RETRIEVE_PER_LABEL,
            "TOP_M_PER_LABEL": TOP_M_PER_LABEL,
            "FULL_N": FULL_N,
        },
        predictions=records,
        duration_seconds=dt_full,
        notes="First pass multi-label candidates → stratified retrieval + cross-encoder rerank → contrastive final judge.",
    )

    print(clf_report_str(y_true, y_pred))
    logger.info("Artifacts: {}", run_dir)


if __name__ == "__main__":
    main()
