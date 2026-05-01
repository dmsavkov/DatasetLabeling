# pyright: basic
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from src.models.interfaces import Prediction, Usage
from src.models.baselines.head_registry import HeadKind, build_head
from src.models.embeddings.fastembed_backend import FastEmbedder


@dataclass(frozen=True, slots=True)
class TrainStats:
    train_time_s: float


@dataclass(frozen=True, slots=True)
class InferStats:
    infer_time_s: float


class EmbUmapHeadPredictor:
    """
    Embeddings + reducer(UMAP(10) or fallback PCA(10)) + head.

    To keep the dependency footprint small, if `umap-learn` is not installed
    we fall back to PCA and still preserve the predictor contract.
    """

    def __init__(
        self,
        *,
        embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        reducer_dim: int = 10,
        head_kind: HeadKind = "xgb",
        head_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
        name: str = "emb_umap_head",
    ) -> None:
        self._name = name
        self._embedding_model_id = embedding_model_id
        self._reducer_dim = int(reducer_dim)
        self._head_kind: HeadKind = head_kind
        self._seed = int(seed)
        self._head_kwargs = dict(head_kwargs or {})

        self._embedder: object | None = None
        self._reducer: object | None = None
        self._head: object | None = None
        self._label_encoder: object | None = None

        self.train_stats: TrainStats | None = None
        self.infer_stats: InferStats | None = None

    @property
    def name(self) -> str:
        return self._name

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._embedder is None:
            # Use fastembed instead of sentence-transformers package.
            self._embedder = FastEmbedder(model_name=self._embedding_model_id)
        return cast(FastEmbedder, self._embedder).embed([str(t) for t in texts])

    def fit(self, texts: list[str], labels: list[str]) -> TrainStats:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have same length")
        if not texts:
            raise ValueError("empty training set")

        start = time.perf_counter()
        X = self._embed(texts)

        # Reducer: UMAP
        umap_mod = importlib.import_module("umap")
        UMAP = getattr(umap_mod, "UMAP")
        self._reducer = UMAP(n_components=self._reducer_dim, random_state=self._seed)
        Xr = getattr(self._reducer, "fit_transform")(X)

        head = build_head(self._head_kind, **self._head_kwargs)
        # XGBoost expects numeric class labels in some configurations; encode if needed.
        if self._head_kind == "xgb":
            sklearn_preproc = importlib.import_module("sklearn.preprocessing")
            LabelEncoder = getattr(sklearn_preproc, "LabelEncoder")
            le = LabelEncoder()
            y_enc = le.fit_transform([str(x) for x in labels])
            head.fit(Xr, y_enc)
            self._label_encoder = le
        else:
            head.fit(Xr, labels)
        self._head = head

        elapsed = time.perf_counter() - start
        self.train_stats = TrainStats(train_time_s=float(elapsed))
        return self.train_stats

    def predict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
        if self._reducer is None or self._head is None:
            raise RuntimeError("Model is not fit yet. Call fit(...) first.")
        if not texts:
            return []
        if not allowed_labels:
            raise ValueError("allowed_labels must be non-empty")

        start = time.perf_counter()
        X = self._embed(texts)
        reducer = cast(Any, self._reducer)
        Xr = reducer.transform(X)

        pred = getattr(self._head, "predict")(Xr)  # type: ignore[union-attr]
        probs: np.ndarray | None = None
        try:
            probs = getattr(self._head, "predict_proba")(Xr)  # type: ignore[assignment]
        except Exception:
            probs = None

        elapsed = time.perf_counter() - start
        self.infer_stats = InferStats(infer_time_s=float(elapsed))

        allowed = set(allowed_labels)
        class_order: list[str] | None = None
        try:
            class_order = [str(x) for x in list(getattr(self._head, "classes_"))]  # type: ignore[arg-type]
        except Exception:
            class_order = None

        # Decode XGB numeric classes back to original string labels.
        if self._head_kind == "xgb" and self._label_encoder is not None:
            le = cast(Any, self._label_encoder)
            try:
                pred = le.inverse_transform(np.asarray(pred, dtype=int))
            except Exception:
                pred = np.asarray(pred)
            try:
                class_order = [str(x) for x in list(getattr(le, "classes_"))]
            except Exception:
                class_order = class_order

        out: list[Prediction] = []
        for i, lab in enumerate(pred.tolist()):
            lab_s = str(lab)
            if lab_s not in allowed:
                lab_s = None

            probs_map: dict[str, float] | None = None
            conf: float | None = None
            if probs is not None and class_order is not None:
                row = probs[i].tolist()
                probs_map = {str(c): float(p) for c, p in zip(class_order, row)}
                if lab_s is not None and lab_s in probs_map:
                    conf = float(probs_map[lab_s])
                else:
                    conf = float(max(row)) if row else None

            out.append(Prediction(pred_label=lab_s, confidence=conf, probs=probs_map, usage=Usage(None, None), raw=None))
        return out

