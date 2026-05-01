# pyright: basic
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.models.interfaces import Prediction, Usage


@dataclass(frozen=True, slots=True)
class TrainStats:
    train_time_s: float


@dataclass(frozen=True, slots=True)
class InferStats:
    infer_time_s: float


class TfidfXgbPredictor:
    """
    TF-IDF + boosting classifier.

    This repo previously planned “TF-IDF + XGB”, but we intentionally avoid a
    hard `xgboost` dependency. We approximate the same idea using
    `sklearn.ensemble.HistGradientBoostingClassifier`.
    """

    def __init__(
        self,
        *,
        min_df: float = 1,
        max_df: float = 1.0,
        max_features: int = 200_000,
        ngram_range: tuple[int, int] = (1, 2),
        n_estimators: int = 300,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        name: str = "tfidf_xgb",
        reducer: Literal["hist_gb"] = "hist_gb",
    ) -> None:
        self._name = name
        self._params = {
            "min_df": min_df,
            "max_df": max_df,
            "max_features": int(max_features),
            "ngram_range": ngram_range,
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            "max_depth": int(max_depth),
            "reducer": reducer,
        }
        self._pipeline: object | None = None
        self.train_stats: TrainStats | None = None
        self.infer_stats: InferStats | None = None
        self._labels: list[str] | None = None

    @property
    def name(self) -> str:
        return self._name

    def fit(self, texts: list[str], labels: list[str]) -> TrainStats:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have same length")
        if not texts:
            raise ValueError("empty training set")

        start = time.perf_counter()
        sklearn_pipeline = importlib.import_module("sklearn.pipeline")
        sklearn_text = importlib.import_module("sklearn.feature_extraction.text")
        sklearn_ensemble = importlib.import_module("sklearn.ensemble")
        sklearn_preproc = importlib.import_module("sklearn.preprocessing")

        Pipeline = getattr(sklearn_pipeline, "Pipeline")
        TfidfVectorizer = getattr(sklearn_text, "TfidfVectorizer")
        HistGradientBoostingClassifier = getattr(sklearn_ensemble, "HistGradientBoostingClassifier")

        pipe = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=tuple(self._params["ngram_range"]),
                        # sklearn interprets `min_df` / `max_df` as absolute counts
                        # when provided as ints, and as fractions when in (0,1].
                        min_df=int(self._params["min_df"])
                        if isinstance(self._params["min_df"], float) and float(self._params["min_df"]).is_integer()
                        else self._params["min_df"],
                        max_df=float(self._params["max_df"])
                        if isinstance(self._params["max_df"], float) and float(self._params["max_df"]).is_integer()
                        else self._params["max_df"],
                        max_features=self._params["max_features"],
                        # Default sklearn token_pattern requires 2+ chars; that makes toy
                        # tests fail (e.g. single-letter tokens). Accept 1+ word chars.
                        token_pattern=r"(?u)\b\w+\b",
                    ),
                ),
                (
                    "to_dense",
                    sklearn_preproc.FunctionTransformer(lambda x: x.toarray(), accept_sparse=True),
                ),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=self._params["max_depth"],
                        learning_rate=self._params["learning_rate"],
                        max_iter=self._params["n_estimators"],
                    ),
                ),
            ]
        )

        pipe.fit(texts, labels)
        elapsed = time.perf_counter() - start
        self._pipeline = pipe
        self._labels = sorted(set(labels))
        self.train_stats = TrainStats(train_time_s=float(elapsed))
        return self.train_stats

    def predict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
        if self._pipeline is None:
            raise RuntimeError("Model is not fit yet. Call fit(...) first.")
        if not texts:
            return []
        if not allowed_labels:
            raise ValueError("allowed_labels must be non-empty")

        start = time.perf_counter()
        pipe = self._pipeline

        pred = getattr(pipe, "predict")(texts)
        probs: np.ndarray | None = None
        try:
            probs = getattr(pipe, "predict_proba")(texts)  # type: ignore[assignment]
        except Exception:
            probs = None

        elapsed = time.perf_counter() - start
        self.infer_stats = InferStats(infer_time_s=float(elapsed))

        allowed = set(allowed_labels)
        out: list[Prediction] = []

        class_order: list[str] | None = None
        try:
            clf = getattr(pipe, "named_steps", {}).get("clf")
            if clf is not None and hasattr(clf, "classes_"):
                class_order = [str(x) for x in list(getattr(clf, "classes_"))]
        except Exception:
            class_order = None

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

            out.append(
                Prediction(
                    pred_label=lab_s,
                    confidence=conf,
                    probs=probs_map,
                    usage=Usage(None, None),
                    raw=None,
                )
            )
        return out

