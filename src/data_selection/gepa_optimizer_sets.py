
# pyright: basic
"""
Build GEPA-oriented train / validation pools: K-means centroids on an embedding pool,
baseline LLM **only on centroids**, contrastive neighbors from centroid errors (embedding search
on the pool), then train/val over centroids + contrast partners (disjoint from test tiers 20/200).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans

from src.datasets.cards import default_card_path, read_card_json
from src.datasets.io import load_processed_tier, processed_root
from src.datasets.schema import SCHEMA, stable_sort_for_determinism, validate_processed_samples_df
from src.experiments.suites.extended_suite import DATASETS, DatasetSpec
from src.models.embeddings.fastembed_backend import FastEmbedder
from src.models.llm.google_genai_batch import GoogleGenaiBatchParams, GoogleGenaiBatchPredictor
from src.prompts.baseline import normalize_label

# --- Train/val overlap resolution (val ids are never modified) ---


def swap_train_overlap_with_val(
    train_ids: set[str],
    val_ids: set[str] | frozenset[str],
    candidates_df: pd.DataFrame,
    label_ids: list[str],
    *,
    seed: int,
) -> set[str]:
    """
    Remove sample ids shared with val from train, then top up train from ``candidates_df``.

    Val ids are never modified. Overlap usually comes from the same physical row appearing
    as anchor in one contrast edge and contrast in another (disjoint pair ids, shared endpoint).
    """
    val_frozen = frozenset(val_ids)
    overlap = train_ids & val_frozen
    if not overlap:
        return train_ids

    logger.warning(
        "Train/val overlap: {} sample id(s) {} — swapping out of train (val unchanged)",
        len(overlap),
        sorted(overlap)[:5],
    )
    train_ids = train_ids - overlap
    n_topup = len(overlap)
    pool = candidates_df[
        ~candidates_df[SCHEMA.sample_id].astype(str).isin(train_ids | val_frozen)
    ]
    if pool.empty:
        logger.warning(
            "No candidates to replace {} overlapped train ids; train may be undersized",
            n_topup,
        )
        return train_ids

    extra_ids = _stratified_sample_ids(
        pool,
        n_total=n_topup,
        labels=label_ids,
        seed=int(seed) + 9876,
    )
    train_ids |= set(extra_ids)

    remaining = train_ids & val_frozen
    if remaining:
        logger.warning(
            "After swap, {} train/val overlap id(s) remain; dropping from train",
            len(remaining),
        )
        train_ids -= remaining
    return train_ids

# Benchmark test tiers to keep out of GEPA mining pools (tier_5000 is not excluded).
GOLDEN_TEST_TIERS: tuple[int, ...] = (20, 200)

PUBMED_LABEL_ALIASES: dict[str, str] = {
    "background": "0",
    "objective": "1",
    "methods": "2",
    "results": "3",
    "conclusions": "4",
}


@dataclass(frozen=True, slots=True)
class GepaSetBuildResult:
    output_dir: Path
    manifest: dict[str, Any]
    centroids_df: pd.DataFrame
    predictions_df: pd.DataFrame
    contrastive_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame

# -------------- rest unchanged so output still readable

def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def dataset_spec_by_name(name: str) -> DatasetSpec:
    for ds in DATASETS:
        if ds.dataset_name == name:
            return ds
    known = ", ".join(d.dataset_name for d in DATASETS)
    raise ValueError(f"Unknown dataset {name!r}. Known: {known}")

def resolve_focus_label(raw: str | None, *, labels_in_data: list[str]) -> str | None:
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip()
    if s in labels_in_data:
        return s
    alias = PUBMED_LABEL_ALIASES.get(s.lower())
    if alias is not None and alias in labels_in_data:
        return alias
    raise ValueError(f"focus_label {raw!r} not in dataset labels {labels_in_data!r}")


def collect_golden_test_sample_ids(
    dataset_name: str,
    *,
    tiers: tuple[int, ...] = GOLDEN_TEST_TIERS,
    root: Path | None = None,
) -> frozenset[str]:
    ids: set[str] = set()
    for tier in tiers:
        try:
            df = load_processed_tier(
                dataset_name=dataset_name,
                split_name="test",
                tier_size=int(tier),
                root=root,
            )
        except FileNotFoundError:
            logger.warning("Missing golden test tier_{} for {}", tier, dataset_name)
            continue
        ids.update(df[SCHEMA.sample_id].astype(str).tolist())
    return frozenset(ids)


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def embed_texts(texts: list[str], *, model_name: str) -> np.ndarray:
    embedder = FastEmbedder(model_name=model_name)
    return embedder.embed([str(t) for t in texts])


def centroid_samples_per_label(
    df: pd.DataFrame,
    *,
    labels: list[str],
    n_clusters: int,
    embedding_model: str,
    seed: int,
    focus_label: str | None = None,
) -> pd.DataFrame:
    """
    Per label (or only ``focus_label``), K-means on embeddings and pick the row closest
    to each cluster centroid.
    """
    label_list = [focus_label] if focus_label is not None else list(labels)
    rows: list[dict[str, Any]] = []

    for lab in label_list:
        sub = df[df[SCHEMA.true_label].astype(str) == str(lab)]
        if sub.empty:
            logger.warning("No rows for label {} in centroid step", lab)
            continue
        texts = sub[SCHEMA.text].astype(str).tolist()
        emb = embed_texts(texts, model_name=embedding_model)
        n = len(sub)
        k = max(1, min(int(n_clusters), n))
        if k == 1:
            idxs = [0]
        else:
            km = KMeans(n_clusters=k, random_state=int(seed), n_init=10)
            labels_km = km.fit_predict(emb)
            centers = km.cluster_centers_
            idxs = []
            for c in range(k):
                mask = labels_km == c
                cluster_idx = np.where(mask)[0]
                if cluster_idx.size == 0:
                    continue
                vecs = emb[cluster_idx]
                dists = np.linalg.norm(vecs - centers[c], axis=1)
                idxs.append(int(cluster_idx[int(np.argmin(dists))]))

        sub_reset = sub.reset_index(drop=True)
        for cluster_id, local_i in enumerate(idxs):
            row = sub_reset.iloc[local_i]
            rows.append(
                {
                    SCHEMA.sample_id: str(row[SCHEMA.sample_id]),
                    SCHEMA.dataset_name: str(row[SCHEMA.dataset_name]),
                    SCHEMA.text: str(row[SCHEMA.text]),
                    SCHEMA.true_label: str(row[SCHEMA.true_label]),
                    "cluster_id": int(cluster_id),
                    "selection_role": "centroid",
                }
            )

    out = pd.DataFrame(rows)
    return stable_sort_for_determinism(out)


def expand_prediction_pool(
    embedding_pool: pd.DataFrame,
    *,
    label_ids: list[str],
    n_centroids_per_label: int,
    prediction_size: int,
    embedding_model: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    K-means per label, expand each cluster to ``samples_per_centroid`` rows, dedupe,
    stratified fill/trim to ``prediction_size``.
    """
    n_centroids = int(n_centroids_per_label) * len(label_ids)
    if prediction_size % n_centroids != 0:
        raise ValueError(f"prediction_size {prediction_size} not divisible by n_centroids {n_centroids}")
    samples_per_centroid = prediction_size // n_centroids

    centroid_rows: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    seen: set[str] = set()

    for lab in label_ids:
        sub = embedding_pool[embedding_pool[SCHEMA.true_label].astype(str) == str(lab)]
        if sub.empty:
            logger.warning("No pool rows for label {} in expand step", lab)
            continue
        sub_reset = sub.reset_index(drop=True)
        texts = sub_reset[SCHEMA.text].astype(str).tolist()
        emb = embed_texts(texts, model_name=embedding_model)
        n = len(sub_reset)
        k = max(1, min(int(n_centroids_per_label), n))

        if k == 1:
            cluster_labels = np.zeros(n, dtype=int)
            centers = emb.mean(axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=k, random_state=int(seed), n_init=10)
            cluster_labels = km.fit_predict(emb)
            centers = km.cluster_centers_

        for cluster_id in range(k):
            mask = cluster_labels == cluster_id
            cluster_idx = np.where(mask)[0]
            if cluster_idx.size == 0:
                continue
            vecs = emb[cluster_idx]
            center = centers[cluster_id] if k > 1 else centers[0]
            dists = np.linalg.norm(vecs - center, axis=1)
            order = cluster_idx[np.argsort(dists)]
            rep_local = int(order[0])
            rep_row = sub_reset.iloc[rep_local]
            centroid_rows.append(
                {
                    SCHEMA.sample_id: str(rep_row[SCHEMA.sample_id]),
                    SCHEMA.dataset_name: str(rep_row[SCHEMA.dataset_name]),
                    SCHEMA.text: str(rep_row[SCHEMA.text]),
                    SCHEMA.true_label: str(rep_row[SCHEMA.true_label]),
                    "cluster_id": int(cluster_id),
                    "label_id": str(lab),
                    "selection_role": "centroid",
                }
            )
            take = 0
            for local_i in order:
                sid = str(sub_reset.iloc[int(local_i)][SCHEMA.sample_id])
                if sid in seen:
                    continue
                seen.add(sid)
                selected_ids.append(sid)
                take += 1
                if take >= samples_per_centroid:
                    break

    centroids_df = stable_sort_for_determinism(pd.DataFrame(centroid_rows))
    if len(selected_ids) >= prediction_size:
        pool_sel = embedding_pool[embedding_pool[SCHEMA.sample_id].astype(str).isin(selected_ids)].copy()
        prediction_pool = stratified_subset_df(
            pool_sel,
            n_total=prediction_size,
            labels=label_ids,
            seed=int(seed),
        )
    else:
        pool_sel = embedding_pool[embedding_pool[SCHEMA.sample_id].astype(str).isin(selected_ids)].copy()
        need = prediction_size - len(pool_sel)
        remainder = embedding_pool[~embedding_pool[SCHEMA.sample_id].astype(str).isin(seen)].copy()
        if need > 0 and not remainder.empty:
            extra = stratified_subset_df(remainder, n_total=need, labels=label_ids, seed=int(seed) + 1)
            prediction_pool = stable_sort_for_determinism(pd.concat([pool_sel, extra], ignore_index=True))
        else:
            prediction_pool = pool_sel

    if len(prediction_pool) != prediction_size:
        logger.warning(
            "Prediction pool size {} != target {} (pool may be too small)",
            len(prediction_pool),
            prediction_size,
        )
    return centroids_df, prediction_pool


def contrastive_neighbors_for_errors(
    errors_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    *,
    embedding_model: str,
    focus_label: str | None = None,
) -> pd.DataFrame:
    """
    For each misclassified row, find the nearest embedding neighbor among rows with a
    different ``true_label`` (cross-class lexical twin).
    """
    if errors_df.empty:
        return pd.DataFrame()

    err = errors_df.copy()
    if focus_label is not None:
        err = err[err[SCHEMA.true_label].astype(str) == str(focus_label)]
    if err.empty:
        return pd.DataFrame()

    pool = pool_df.copy()
    pool_emb = _l2_normalize_rows(embed_texts(pool[SCHEMA.text].astype(str).tolist(), model_name=embedding_model))
    err_emb = _l2_normalize_rows(embed_texts(err[SCHEMA.text].astype(str).tolist(), model_name=embedding_model))

    pool_labels = pool[SCHEMA.true_label].astype(str).to_numpy()
    pool_ids = pool[SCHEMA.sample_id].astype(str).to_numpy()

    records: list[dict[str, Any]] = []
    for i, err_row in err.reset_index(drop=True).iterrows():
        true_lab = str(err_row[SCHEMA.true_label])
        mask = pool_labels != true_lab
        if not bool(mask.any()):
            continue
        cand_emb = pool_emb[mask]
        sims = cand_emb @ err_emb[i]
        best_local = int(np.argmax(sims))
        global_idx = int(np.where(mask)[0][best_local])
        contrast_row = pool.iloc[global_idx]
        records.append(
            {
                "anchor_sample_id": str(err_row[SCHEMA.sample_id]),
                "anchor_true_label": true_lab,
                "anchor_text": str(err_row[SCHEMA.text]),
                "pred_label": str(err_row.get("pred_label", "")),
                "contrast_sample_id": str(contrast_row[SCHEMA.sample_id]),
                "contrast_true_label": str(contrast_row[SCHEMA.true_label]),
                "contrast_text": str(contrast_row[SCHEMA.text]),
                "cosine_similarity": float(sims[best_local]),
                "selection_role": "contrastive_edge",
            }
        )

    return pd.DataFrame(records)


def build_operating_frame(
    embedding_pool: pd.DataFrame,
    centroids_df: pd.DataFrame,
    centroid_predictions: pd.DataFrame,
    contrastive_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rows we keep for GEPA train/val: centroid samples plus contrast partners from edges.
    Predictions exist only for centroids (contrast rows have null pred / not correct).
    """
    centroid_ids = set(centroids_df[SCHEMA.sample_id].astype(str).tolist())
    operating_ids = set(centroid_ids)
    if not contrastive_df.empty:
        operating_ids.update(contrastive_df["anchor_sample_id"].astype(str).tolist())
        operating_ids.update(contrastive_df["contrast_sample_id"].astype(str).tolist())

    base = embedding_pool[embedding_pool[SCHEMA.sample_id].astype(str).isin(operating_ids)].copy()
    pred_cols = [SCHEMA.sample_id, "pred_label", "correct"]
    preds = centroid_predictions[pred_cols].drop_duplicates(subset=[SCHEMA.sample_id])
    merged = base.merge(preds, on=SCHEMA.sample_id, how="left")
    is_centroid = merged[SCHEMA.sample_id].astype(str).isin(centroid_ids)
    merged.loc[~is_centroid, "pred_label"] = None
    merged["correct"] = is_centroid & merged["correct"].astype("boolean").fillna(False)
    return stable_sort_for_determinism(merged)


def _allowed_labels_from_card(dataset_name: str, *, root: Path | None) -> list[str]:
    pr = processed_root(root)
    card_path = default_card_path(processed_root_dir=pr, dataset_name=dataset_name)
    if card_path.exists():
        return [str(x) for x in read_card_json(card_path).labels]
    raise FileNotFoundError(f"Missing dataset card: {card_path}")


async def _run_baseline_predictions(
    df: pd.DataFrame,
    *,
    model_id: str,
    allowed_labels: list[str],
    batch_size: int,
    max_concurrency: int,
) -> pd.DataFrame:
    params = GoogleGenaiBatchParams(
        model_id=model_id,
        prompt_id="baseline_v1",
        few_shot=None,
        batch_size=int(batch_size),
        max_concurrency=int(max_concurrency),
        thinking_level="off",
        include_thoughts=False,
    )
    predictor = GoogleGenaiBatchPredictor(params=params)
    texts = df[SCHEMA.text].astype(str).tolist()
    preds = await predictor.apredict(texts, allowed_labels=allowed_labels)

    pred_labels = [normalize_label(p.pred_label, allowed_labels=allowed_labels) if p.pred_label else None for p in preds]
    out = df.reset_index(drop=True).copy()
    out["pred_label"] = pred_labels
    out["correct"] = out[SCHEMA.true_label].astype(str) == out["pred_label"].astype(str)
    return out


def stratified_subset_df(
    df: pd.DataFrame,
    *,
    n_total: int,
    labels: list[str],
    seed: int,
) -> pd.DataFrame:
    """Uniform stratified sample of ``n_total`` rows (by ``true_label``)."""
    if n_total <= 0 or df.empty:
        return df.iloc[0:0].copy()
    if len(df) <= n_total:
        return df.copy()
    ids = _stratified_sample_ids(df, n_total=n_total, labels=labels, seed=seed)
    out = df[df[SCHEMA.sample_id].astype(str).isin(ids)].copy()
    return stable_sort_for_determinism(out)


def _stratified_sample_ids(
    candidates: pd.DataFrame,
    *,
    n_total: int,
    labels: list[str],
    seed: int,
) -> list[str]:
    if n_total <= 0 or candidates.empty:
        return []
    rng = np.random.default_rng(int(seed))
    per_label = n_total // len(labels)
    remainder = n_total % len(labels)
    targets = {lab: per_label + (1 if i < remainder else 0) for i, lab in enumerate(labels)}

    picked: list[str] = []
    for lab in labels:
        need = targets[lab]
        pool = candidates[candidates[SCHEMA.true_label].astype(str) == str(lab)]
        if pool.empty or need <= 0:
            continue
        ids = pool[SCHEMA.sample_id].astype(str).tolist()
        if len(ids) <= need:
            picked.extend(ids)
        else:
            choice = rng.choice(ids, size=need, replace=False).tolist()
            picked.extend([str(x) for x in choice])
    return picked


def build_val_holdout(
    predictions_df: pd.DataFrame,
    contrastive_df: pd.DataFrame,
    *,
    labels: list[str],
    n_total: int,
    contrastive_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, frozenset[str]]:
    """
    Stratified val set: uniform across labels; ``contrastive_fraction`` from contrastive
    pool (anchor + contrast ids), remainder from correctly predicted rows.
    """
    n_contrastive = int(round(int(n_total) * float(contrastive_fraction)))
    n_easy = int(n_total) - n_contrastive

    contrast_ids: set[str] = set()
    if not contrastive_df.empty:
        contrast_ids.update(contrastive_df["anchor_sample_id"].astype(str).tolist())
        contrast_ids.update(contrastive_df["contrast_sample_id"].astype(str).tolist())

    contrast_pool = predictions_df[predictions_df[SCHEMA.sample_id].astype(str).isin(contrast_ids)]
    easy_pool = predictions_df[predictions_df["correct"] == True]  # noqa: E712

    contrast_ids_out = _stratified_sample_ids(contrast_pool, n_total=n_contrastive, labels=labels, seed=seed)
    easy_pool = easy_pool[~easy_pool[SCHEMA.sample_id].astype(str).isin(contrast_ids_out)]
    easy_ids_out = _stratified_sample_ids(
        easy_pool,
        n_total=n_easy,
        labels=labels,
        seed=seed + 1,
    )

    val_ids = list(dict.fromkeys(contrast_ids_out + easy_ids_out))
    if len(val_ids) < n_total:
        logger.warning(
            "Val holdout undersized: requested {} got {} (contrastive={}, easy={})",
            n_total,
            len(val_ids),
            len(contrast_ids_out),
            len(easy_ids_out),
        )

    val_df = predictions_df[predictions_df[SCHEMA.sample_id].astype(str).isin(val_ids)].copy()
    val_df["val_bucket"] = val_df[SCHEMA.sample_id].astype(str).apply(
        lambda sid: "contrastive" if sid in contrast_ids_out else "easy"
    )
    return val_df, frozenset(val_ids)


def build_train_from_contrastive(
    operating_df: pd.DataFrame,
    contrastive_df: pd.DataFrame,
    *,
    val_ids: frozenset[str],
) -> pd.DataFrame:
    train_ids: set[str] = set()
    if not contrastive_df.empty:
        train_ids.update(contrastive_df["anchor_sample_id"].astype(str).tolist())
        train_ids.update(contrastive_df["contrast_sample_id"].astype(str).tolist())
    train_ids -= set(val_ids)
    train_df = operating_df[operating_df[SCHEMA.sample_id].astype(str).isin(train_ids)].copy()
    train_df["selection_role"] = "gepa_train_contrastive"
    return stable_sort_for_determinism(train_df)


def build_contrastive_pair_table(contrastive_df: pd.DataFrame) -> pd.DataFrame:
    if contrastive_df.empty:
        return pd.DataFrame(
            columns=["pair_id", "anchor_sample_id", "contrast_sample_id", "anchor_true_label"],
        )
    records: list[dict[str, str]] = []
    for row in contrastive_df.itertuples(index=False):
        aid = str(getattr(row, "anchor_sample_id"))
        cid = str(getattr(row, "contrast_sample_id"))
        records.append(
            {
                "pair_id": f"{aid}|{cid}",
                "anchor_sample_id": aid,
                "contrast_sample_id": cid,
                "anchor_true_label": str(getattr(row, "anchor_true_label")),
            }
        )
    return pd.DataFrame(records)


def _pairs_excluding_sample_ids(
    pairs_df: pd.DataFrame,
    exclude_sample_ids: frozenset[str] | set[str] | None,
) -> pd.DataFrame:
    """Drop pairs whose anchor or contrast endpoint is already reserved."""
    if not exclude_sample_ids:
        return pairs_df
    excl = frozenset(exclude_sample_ids)
    if not excl:
        return pairs_df
    anchor = pairs_df["anchor_sample_id"].astype(str)
    contrast = pairs_df["contrast_sample_id"].astype(str)
    keep = ~anchor.isin(excl) & ~contrast.isin(excl)
    return pairs_df.loc[keep].copy()


def _sample_pair_ids_stratified(
    pairs_df: pd.DataFrame,
    *,
    n_pairs: int,
    label_ids: list[str],
    seed: int,
    exclude_pair_ids: frozenset[str] | None = None,
    exclude_sample_ids: frozenset[str] | set[str] | None = None,
) -> list[str]:
    from src.data_selection.label_utils import per_label_quotas

    if n_pairs <= 0 or pairs_df.empty:
        return []
    excl = exclude_pair_ids or frozenset()
    available = pairs_df[~pairs_df["pair_id"].astype(str).isin(excl)].copy()
    available = _pairs_excluding_sample_ids(available, exclude_sample_ids)
    if available.empty:
        return []

    targets = per_label_quotas(n_pairs, label_ids)
    rng = np.random.default_rng(int(seed))
    picked: list[str] = []
    for lab in label_ids:
        need = int(targets.get(lab, 0))
        pool = available[available["anchor_true_label"].astype(str) == str(lab)]
        ids = pool["pair_id"].astype(str).tolist()
        if not ids or need <= 0:
            continue
        if len(ids) <= need:
            picked.extend(ids)
        else:
            choice = rng.choice(ids, size=need, replace=False).tolist()
            picked.extend([str(x) for x in choice])

    if len(picked) < n_pairs:
        remaining = available[~available["pair_id"].astype(str).isin(picked)]
        extra_need = n_pairs - len(picked)
        extra_ids = remaining["pair_id"].astype(str).tolist()
        if extra_ids and extra_need > 0:
            take = min(extra_need, len(extra_ids))
            extra = rng.choice(extra_ids, size=take, replace=False).tolist()
            picked.extend([str(x) for x in extra])
    return picked[:n_pairs]


def _row_ids_for_pairs(pairs_df: pd.DataFrame, pair_ids: list[str]) -> set[str]:
    sub = pairs_df[pairs_df["pair_id"].astype(str).isin(pair_ids)]
    out: set[str] = set()
    out.update(sub["anchor_sample_id"].astype(str).tolist())
    out.update(sub["contrast_sample_id"].astype(str).tolist())
    return out


def _gepa_labeled_frame(
    predictions_df: pd.DataFrame,
    contrastive_df: pd.DataFrame,
    *,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Union of LLM-predicted pool rows plus contrast partners from edges.

    Contrast neighbors are drawn from the 5k embedding pool, so they are usually absent
    from ``predictions.parquet`` (500 rows). Without this merge, pair selection drops
    every contrast half and GEPA sets undershoot ``train_total`` / ``val_total``.
    """
    base = predictions_df.copy()
    present = set(base[SCHEMA.sample_id].astype(str))
    extras: list[dict[str, Any]] = []
    if not contrastive_df.empty:
        for row in contrastive_df.itertuples(index=False):
            cid = str(getattr(row, "contrast_sample_id"))
            if cid in present:
                continue
            text = getattr(row, "contrast_text", None)
            if text is None or not str(text).strip():
                continue
            extras.append(
                {
                    SCHEMA.sample_id: cid,
                    SCHEMA.dataset_name: str(dataset_name),
                    SCHEMA.text: str(text),
                    SCHEMA.true_label: str(getattr(row, "contrast_true_label")),
                    "pred_label": None,
                    "correct": False,
                }
            )
            present.add(cid)
    if extras:
        base = pd.concat([base, pd.DataFrame(extras)], ignore_index=True)
    base = base.drop_duplicates(subset=[SCHEMA.sample_id], keep="first")
    return stable_sort_for_determinism(base)


def _sample_error_anchor_ids(
    predictions_df: pd.DataFrame,
    *,
    n: int,
    label_ids: list[str],
    seed: int,
    exclude_ids: set[str],
) -> list[str]:
    """Single misclassified anchors (fallback when pair budget is not fillable)."""
    if n <= 0:
        return []
    pool = predictions_df[
        (predictions_df["correct"] == False)  # noqa: E712
        & ~predictions_df[SCHEMA.sample_id].astype(str).isin(exclude_ids)
    ].copy()
    return _stratified_sample_ids(pool, n_total=n, labels=label_ids, seed=seed)


def _label_histogram(df: pd.DataFrame, label_ids: list[str]) -> dict[str, int]:
    counts = df[SCHEMA.true_label].astype(str).value_counts().to_dict()
    return {lab: int(counts.get(lab, 0)) for lab in label_ids}


def build_gepa_train_val_from_huge_prediction(
    huge_prediction_dir: Path,
    *,
    dataset_name: str,
    train_total: int = 50,
    train_easy_fraction: float = 0.2,
    val_total: int = 70,
    seed: int = 42,
    processed_root_path: Path | None = None,
) -> GepaSetBuildResult:
    """
    Phase B: build ``gepa_train`` / ``gepa_val`` from huge-prediction artifacts (no LLM).
    Train: 80% hard (contrastive pairs), 20% easy. Val: ~50/50 with whole pairs only.
    """
    from src.data_selection.label_utils import load_dataset_context

    out_dir = Path(huge_prediction_dir)
    ctx = load_dataset_context(dataset_name, processed_root_path=processed_root_path)
    manifest_path = out_dir / "manifest.json"
    base_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        base_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    predictions_df = pd.read_parquet(out_dir / "predictions.parquet")
    contrastive_df = pd.read_parquet(out_dir / "contrastive_edges.parquet")
    centroids_df = pd.read_parquet(out_dir / "centroid_samples.parquet")
    pairs_df = build_contrastive_pair_table(contrastive_df)
    gepa_df = _gepa_labeled_frame(
        predictions_df,
        contrastive_df,
        dataset_name=ctx.dataset_name,
    )
    pool_ids = set(predictions_df[SCHEMA.sample_id].astype(str))
    contrast_enriched = int(len(gepa_df)) - int(len(predictions_df))

    train_easy_n = int(round(train_total * float(train_easy_fraction)))
    train_hard_rows = train_total - train_easy_n
    train_n_pairs = train_hard_rows // 2

    val_easy_target = val_total // 2
    val_hard_rows = val_total - val_easy_target
    if val_hard_rows % 2 == 1:
        val_hard_rows += 1
        val_easy_target = val_total - val_hard_rows
    val_n_pairs = val_hard_rows // 2

    train_pair_ids = _sample_pair_ids_stratified(
        pairs_df,
        n_pairs=train_n_pairs,
        label_ids=ctx.label_ids,
        seed=int(seed),
    )
    train_row_ids = _row_ids_for_pairs(pairs_df, train_pair_ids)

    val_pair_ids = _sample_pair_ids_stratified(
        pairs_df,
        n_pairs=val_n_pairs,
        label_ids=ctx.label_ids,
        seed=int(seed) + 100,
        exclude_pair_ids=frozenset(train_pair_ids),
        exclude_sample_ids=frozenset(train_row_ids),
    )
    val_row_ids_from_pairs = _row_ids_for_pairs(pairs_df, val_pair_ids)

    easy_pool = predictions_df[
        (predictions_df["correct"] == True)  # noqa: E712
        & ~predictions_df[SCHEMA.sample_id].astype(str).isin(train_row_ids | val_row_ids_from_pairs)
    ].copy()

    train_easy_ids = _stratified_sample_ids(
        easy_pool,
        n_total=train_easy_n,
        labels=ctx.label_ids,
        seed=int(seed) + 1,
    )
    easy_pool = easy_pool[~easy_pool[SCHEMA.sample_id].astype(str).isin(train_easy_ids)]
    val_easy_ids = _stratified_sample_ids(
        easy_pool,
        n_total=val_easy_target,
        labels=ctx.label_ids,
        seed=int(seed) + 2,
    )

    train_ids: set[str] = set(train_easy_ids) | train_row_ids
    val_ids: set[str] = set(val_easy_ids) | val_row_ids_from_pairs
    overlap_resolved = 0
    if train_ids & val_ids:
        overlap_resolved = len(train_ids & val_ids)
        train_ids = swap_train_overlap_with_val(
            train_ids,
            val_ids,
            gepa_df,
            ctx.label_ids,
            seed=int(seed),
        )

    def _fill_hard_singles(
        target_ids: set[str],
        *,
        want_rows: int,
        seed_offset: int,
    ) -> set[str]:
        need = want_rows - len(target_ids)
        if need <= 0:
            return target_ids
        exclude = set(target_ids)
        if seed_offset == 200:
            exclude |= val_ids
        else:
            exclude |= train_ids
        extra = _sample_error_anchor_ids(
            predictions_df,
            n=need,
            label_ids=ctx.label_ids,
            seed=int(seed) + seed_offset,
            exclude_ids=exclude,
        )
        return target_ids | set(extra)

    train_ids = _fill_hard_singles(train_ids, want_rows=train_total, seed_offset=200)
    val_ids = _fill_hard_singles(val_ids, want_rows=val_total, seed_offset=300)

    if train_ids & val_ids:
        overlap_resolved += len(train_ids & val_ids)
        train_ids = swap_train_overlap_with_val(
            train_ids,
            val_ids,
            gepa_df,
            ctx.label_ids,
            seed=int(seed) + 11,
        )

    final_overlap = train_ids & val_ids
    if final_overlap:
        raise RuntimeError(
            f"Train/val overlap unresolved after swap: {len(final_overlap)} ids "
            f"(e.g. {sorted(final_overlap)[:3]})"
        )

    def _bucket(sid: str, *, easy_ids: set[str], pair_row_ids: set[str]) -> str:
        if sid in easy_ids:
            return "easy"
        if sid in pair_row_ids:
            return "hard_pair"
        return "hard_error"

    train_df = gepa_df[gepa_df[SCHEMA.sample_id].astype(str).isin(train_ids)].copy()
    train_df["gepa_bucket"] = train_df[SCHEMA.sample_id].astype(str).apply(
        lambda sid: _bucket(sid, easy_ids=set(train_easy_ids), pair_row_ids=train_row_ids)
    )
    val_df = gepa_df[gepa_df[SCHEMA.sample_id].astype(str).isin(val_ids)].copy()
    val_df["gepa_bucket"] = val_df[SCHEMA.sample_id].astype(str).apply(
        lambda sid: (
            "easy"
            if sid in val_easy_ids
            else ("contrastive_pair" if sid in val_row_ids_from_pairs else "hard_error")
        )
    )

    def _top_up_easy(
        frame: pd.DataFrame,
        assigned: set[str],
        *,
        want: int,
        seed_offset: int,
    ) -> pd.DataFrame:
        need = want - len(frame)
        if need <= 0:
            return frame
        pool = predictions_df[
            (predictions_df["correct"] == True)  # noqa: E712
            & ~predictions_df[SCHEMA.sample_id].astype(str).isin(assigned)
        ]
        extra_ids = _stratified_sample_ids(
            pool,
            n_total=need,
            labels=ctx.label_ids,
            seed=int(seed) + seed_offset,
        )
        extra = gepa_df[gepa_df[SCHEMA.sample_id].astype(str).isin(extra_ids)].copy()
        extra["gepa_bucket"] = "easy"
        return stable_sort_for_determinism(pd.concat([frame, extra], ignore_index=True))

    all_assigned = set(train_df[SCHEMA.sample_id].astype(str)) | set(val_df[SCHEMA.sample_id].astype(str))
    train_df = _top_up_easy(train_df, all_assigned, want=train_total, seed_offset=400)
    all_assigned = set(train_df[SCHEMA.sample_id].astype(str)) | set(val_df[SCHEMA.sample_id].astype(str))
    val_df = _top_up_easy(val_df, all_assigned, want=val_total, seed_offset=500)

    train_df = stable_sort_for_determinism(train_df)
    val_df = stable_sort_for_determinism(val_df)

    if len(train_df) != train_total:
        logger.warning("Train rows {} != target {}", len(train_df), train_total)
    if len(val_df) != val_total:
        logger.warning("Val rows {} != target {}", len(val_df), val_total)

    train_df.to_parquet(out_dir / "gepa_train.parquet", index=False)
    val_df.to_parquet(out_dir / "gepa_val.parquet", index=False)

    gepa_manifest: dict[str, Any] = {
        **base_manifest,
        "gepa_train_rows": int(len(train_df)),
        "gepa_val_rows": int(len(val_df)),
        "train_easy_rows": int((train_df["gepa_bucket"] == "easy").sum()),
        "train_hard_rows": int(train_df["gepa_bucket"].isin(("hard_pair", "hard_error")).sum()),
        "val_easy_rows": int((val_df["gepa_bucket"] == "easy").sum()),
        "val_contrastive_rows": int((val_df["gepa_bucket"] == "contrastive_pair").sum()),
        "train_hard_error_rows": int((train_df["gepa_bucket"] == "hard_error").sum()),
        "val_hard_error_rows": int((val_df["gepa_bucket"] == "hard_error").sum()),
        "contrast_rows_enriched_from_edges": contrast_enriched,
        "prediction_pool_row_ids": int(len(pool_ids)),
        "train_pairs_selected": len(train_pair_ids),
        "val_pairs_selected": len(val_pair_ids),
        "train_label_histogram": _label_histogram(train_df, ctx.label_ids),
        "val_label_histogram": _label_histogram(val_df, ctx.label_ids),
        "train_val_overlap": 0,
        "train_val_overlap_resolved_via_swap": int(overlap_resolved),
    }
    (out_dir / "manifest.json").write_text(json.dumps(gepa_manifest, indent=2), encoding="utf-8")

    return GepaSetBuildResult(
        output_dir=out_dir,
        manifest=gepa_manifest,
        centroids_df=centroids_df,
        predictions_df=predictions_df,
        contrastive_df=contrastive_df,
        train_df=train_df,
        val_df=val_df,
    )


async def build_gepa_optimizer_sets(
    *,
    dataset_name: str,
    train_parquet: Path,
    output_dir: Path,
    model_id: str = "gemma-4-31b-it",
    batch_size: int = 5,
    max_concurrency: int = 5,
    n_clusters: int = 35,
    val_total: int = 70,
    val_contrastive_fraction: float = 0.7,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    seed: int = 42,
    focus_label: str | None = None,
    pool_size: int = 5000,
    max_embedding_pool_rows: int | None = None,
    skip_predict: bool = False,
    processed_root_path: Path | None = None,
) -> GepaSetBuildResult:
    source_df = pd.read_parquet(train_parquet)
    validate_processed_samples_df(source_df)
    source_df = stable_sort_for_determinism(source_df)
    rows_loaded = int(len(source_df))

    labels = _allowed_labels_from_card(dataset_name, root=processed_root_path)
    focus = resolve_focus_label(focus_label, labels_in_data=labels)

    golden_ids = collect_golden_test_sample_ids(dataset_name, root=processed_root_path)
    before = len(source_df)
    embedding_pool = source_df[~source_df[SCHEMA.sample_id].astype(str).isin(golden_ids)].copy()
    rows_after_golden = int(len(embedding_pool))
    logger.info(
        "Excluded {} golden test rows (tiers {}), {} rows remain for embedding pool",
        before - rows_after_golden,
        GOLDEN_TEST_TIERS,
        rows_after_golden,
    )

    pool_target = int(pool_size) if int(pool_size) > 0 else None
    if pool_target is not None and rows_after_golden > pool_target:
        embedding_pool = stratified_subset_df(
            embedding_pool,
            n_total=pool_target,
            labels=labels,
            seed=int(seed),
        )
        logger.info(
            "Embedding pool (stratified): {} rows (target {}, from {} after golden exclusion)",
            len(embedding_pool),
            pool_target,
            rows_after_golden,
        )
    elif pool_target is not None:
        logger.info(
            "Embedding pool: all {} rows (at or below stratified target {})",
            rows_after_golden,
            pool_target,
        )

    if max_embedding_pool_rows is not None:
        embedding_pool = embedding_pool.head(int(max_embedding_pool_rows)).copy()
        logger.info("Capped embedding pool to max_embedding_pool_rows={}", max_embedding_pool_rows)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Centroid selection (k={}) focus_label={}", n_clusters, focus)
    centroids_df = centroid_samples_per_label(
        embedding_pool,
        labels=labels,
        n_clusters=int(n_clusters),
        embedding_model=embedding_model,
        seed=int(seed),
        focus_label=focus,
    )
    centroids_df.to_parquet(out_dir / "centroid_samples.parquet", index=False)

    centroid_pred_path = out_dir / "centroid_predictions.parquet"
    if skip_predict and centroid_pred_path.exists():
        centroid_predictions = pd.read_parquet(centroid_pred_path)
        logger.info("Loaded cached centroid predictions from {}", centroid_pred_path)
    else:
        logger.info(
            "Running baseline {} on {} centroid rows only (batch_size={})",
            model_id,
            len(centroids_df),
            batch_size,
        )
        centroid_predictions = await _run_baseline_predictions(
            centroids_df,
            model_id=model_id,
            allowed_labels=labels,
            batch_size=int(batch_size),
            max_concurrency=int(max_concurrency),
        )
        centroid_predictions.to_parquet(centroid_pred_path, index=False)

    errors_df = centroid_predictions[centroid_predictions["correct"] == False].copy()  # noqa: E712
    logger.info("Centroid baseline errors: {} / {}", len(errors_df), len(centroid_predictions))

    contrastive_df = contrastive_neighbors_for_errors(
        errors_df,
        embedding_pool,
        embedding_model=embedding_model,
        focus_label=focus,
    )
    contrastive_df.to_parquet(out_dir / "contrastive_edges.parquet", index=False)
    logger.info("Contrastive edges: {}", len(contrastive_df))

    operating_df = build_operating_frame(
        embedding_pool,
        centroids_df,
        centroid_predictions,
        contrastive_df,
    )
    operating_df.to_parquet(out_dir / "operating_samples.parquet", index=False)
    # Legacy filename: operating set (not full-pool predictions).
    operating_df.to_parquet(out_dir / "baseline_predictions.parquet", index=False)
    logger.info(
        "Operating set: {} rows (centroids={}, with LLM preds on centroids only)",
        len(operating_df),
        len(centroids_df),
    )

    val_df, val_ids = build_val_holdout(
        operating_df,
        contrastive_df,
        labels=labels,
        n_total=int(val_total),
        contrastive_fraction=float(val_contrastive_fraction),
        seed=int(seed),
    )
    train_out_df = build_train_from_contrastive(operating_df, contrastive_df, val_ids=val_ids)

    overlap = set(val_ids) & set(train_out_df[SCHEMA.sample_id].astype(str).tolist())
    if overlap:
        raise RuntimeError(f"Train/val overlap detected: {len(overlap)} ids")

    val_df.to_parquet(out_dir / "gepa_val.parquet", index=False)
    train_out_df.to_parquet(out_dir / "gepa_train.parquet", index=False)

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "train_parquet": str(train_parquet),
        "model_id": model_id,
        "batch_size": int(batch_size),
        "n_clusters": int(n_clusters),
        "val_total": int(val_total),
        "val_contrastive_fraction": float(val_contrastive_fraction),
        "embedding_model": embedding_model,
        "seed": int(seed),
        "focus_label": focus,
        "golden_test_tiers_excluded": list(GOLDEN_TEST_TIERS),
        "golden_test_ids_excluded_count": len(golden_ids),
        "rows_loaded": rows_loaded,
        "rows_after_golden_exclusion": rows_after_golden,
        "embedding_pool_size_target": pool_target,
        "embedding_pool_rows": int(len(embedding_pool)),
        "centroid_rows": int(len(centroids_df)),
        "centroid_prediction_rows": int(len(centroid_predictions)),
        "operating_rows": int(len(operating_df)),
        "error_rows": int(len(errors_df)),
        "contrastive_edge_rows": int(len(contrastive_df)),
        "gepa_train_rows": int(len(train_out_df)),
        "gepa_val_rows": int(len(val_df)),
        "train_val_overlap": 0,
        "llm_ran_on": "centroids_only",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return GepaSetBuildResult(
        output_dir=out_dir,
        manifest=manifest,
        centroids_df=centroids_df,
        predictions_df=operating_df,
        contrastive_df=contrastive_df,
        train_df=train_out_df,
        val_df=val_df,
    )


def run_build_gepa_optimizer_sets_sync(**kwargs: Any) -> GepaSetBuildResult:
    return asyncio.run(build_gepa_optimizer_sets(**kwargs))