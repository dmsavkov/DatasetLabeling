from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from prosocial.constants import FASTEMBED_MODEL
from prosocial.prompting import is_unanimous_annotations
from src.data import now_stamp, save_json

try:
    from fastembed import TextEmbedding
except Exception:
    TextEmbedding = None


def embed_texts(texts: list[str], *, model_name: str = FASTEMBED_MODEL) -> np.ndarray:
    if TextEmbedding is None:
        raise RuntimeError("fastembed is required for embedding-based experiments")

    embedder = TextEmbedding(model_name=model_name)
    vectors = [np.asarray(v, dtype=np.float32) for v in iter(embedder.embed(texts))]
    if len(vectors) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: expected {len(texts)}, got {len(vectors)}")
    return np.vstack(vectors) if vectors else np.zeros((0, 384), dtype=np.float32)


def build_representative_dataset(
    *,
    pool_df: pd.DataFrame,
    labels: list[str],
    results_dir: Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = pool_df.copy()

    unanimous_pairs = [
        is_unanimous_annotations(values)
        for values in df.get("safety_annotations", pd.Series([[] for _ in range(len(df))])).tolist()
    ]
    df["is_unanimous_annotations"] = [pair[0] for pair in unanimous_pairs]
    df["unanimous_annotation_label"] = [pair[1] for pair in unanimous_pairs]
    unanimous_df = df[df["is_unanimous_annotations"]].copy().reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    embedding_chunks: list[np.ndarray] = []
    row_offset = 0
    kmeans_meta: dict[str, Any] = {}

    for label in labels:
        stratum_df = unanimous_df[unanimous_df["safety_label"] == label].copy().reset_index(drop=True)
        if len(stratum_df) == 0:
            kmeans_meta[label] = {"status": "empty", "k": 0, "size": 0}
            continue

        vectors = embed_texts(stratum_df["context"].astype(str).tolist(), model_name=FASTEMBED_MODEL)
        k = min(20, len(stratum_df))
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        cluster_id = km.fit_predict(vectors)
        centers = km.cluster_centers_[cluster_id]
        distances = np.linalg.norm(vectors - centers, axis=1)

        stratum_df["stratum_label"] = label
        stratum_df["cluster_id"] = cluster_id.astype(int)
        stratum_df["cluster_k"] = int(k)
        stratum_df["cluster_distance"] = distances.astype(float)
        stratum_df["embedding_row_id"] = np.arange(row_offset, row_offset + len(stratum_df), dtype=int)
        stratum_df["representative_rank_in_cluster"] = (
            stratum_df.groupby("cluster_id")["cluster_distance"].rank(method="dense", ascending=True).astype(int)
        )
        stratum_df["is_cluster_representative"] = stratum_df["representative_rank_in_cluster"] == 1

        parts.append(stratum_df)
        embedding_chunks.append(vectors)
        row_offset += len(stratum_df)

        kmeans_meta[label] = {
            "status": "ok",
            "k": int(k),
            "size": int(len(stratum_df)),
            "cluster_counts": {
                str(int(cid)): int(count)
                for cid, count in stratum_df["cluster_id"].value_counts().sort_index().to_dict().items()
            },
        }

    if parts:
        representative_df = pd.concat(parts, ignore_index=True)
        embedding_matrix = np.vstack(embedding_chunks).astype(np.float32)
    else:
        representative_df = unanimous_df.head(0).copy()
        representative_df["stratum_label"] = pd.Series(dtype="string")
        representative_df["cluster_id"] = pd.Series(dtype="int64")
        representative_df["cluster_k"] = pd.Series(dtype="int64")
        representative_df["cluster_distance"] = pd.Series(dtype="float64")
        representative_df["embedding_row_id"] = pd.Series(dtype="int64")
        representative_df["representative_rank_in_cluster"] = pd.Series(dtype="int64")
        representative_df["is_cluster_representative"] = pd.Series(dtype="bool")
        embedding_matrix = np.zeros((0, 384), dtype=np.float32)

    top100_df = (
        representative_df[representative_df["is_cluster_representative"]]
        .sort_values(["stratum_label", "cluster_id", "cluster_distance"], kind="stable")
        .head(100)
        .reset_index(drop=True)
    )

    stamp = now_stamp()
    dataset_path = results_dir / f"representative_clustered_dataset_{stamp}.csv"
    top100_path = results_dir / f"representative_top100_dataset_{stamp}.csv"
    embedding_path = results_dir / f"representative_embeddings_{stamp}.npy"
    meta_path = results_dir / f"representative_meta_{stamp}.json"

    representative_df.to_csv(dataset_path, index=False)
    top100_df.to_csv(top100_path, index=False)
    np.save(embedding_path, embedding_matrix)

    meta = {
        "fastembed_model": FASTEMBED_MODEL,
        "unanimous_total": int(len(unanimous_df)),
        "representative_total": int(len(representative_df)),
        "top100_total": int(len(top100_df)),
        "kmeans": kmeans_meta,
        "artifacts": {
            "representative_dataset_csv": str(dataset_path),
            "top100_dataset_csv": str(top100_path),
            "embedding_npy": str(embedding_path),
        },
    }
    save_json(meta_path, meta)
    return representative_df, top100_df, meta


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def build_retrieval_map(
    *,
    top100_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[int, list[dict[str, Any]]]:
    if len(top100_df) == 0:
        return {}

    top100_vec = normalize_rows(embed_texts(top100_df["context"].astype(str).tolist(), model_name=FASTEMBED_MODEL))
    test_vec = normalize_rows(embed_texts(test_df["context"].astype(str).tolist(), model_name=FASTEMBED_MODEL))
    sim = test_vec @ top100_vec.T

    retrieval: dict[int, list[dict[str, Any]]] = {}
    for row_idx, row in enumerate(test_df.to_dict(orient="records")):
        source_index = int(row["source_index"])
        top_ids = np.argsort(-sim[row_idx])[:3]
        retrieval[source_index] = [
            {
                "source_index": int(top100_df.iloc[int(idx)]["source_index"]),
                "context": str(top100_df.iloc[int(idx)]["context"]),
                "label": str(top100_df.iloc[int(idx)]["safety_label"]),
            }
            for idx in top_ids.tolist()
        ]

    return retrieval
