# pyright: basic
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass

import numpy as np

from ..interfaces import Prediction, Usage


@dataclass(frozen=True, slots=True)
class TrainStats:
    train_time_s: float


@dataclass(frozen=True, slots=True)
class InferStats:
    infer_time_s: float


class SklearnTfidfLogRegPredictor:
    """
    Simple, fast baseline classifier.

    Uses LogisticRegression so we can expose `predict_proba` for confidence/probs.
    """

    def __init__(self, *, name: str = "sklearn_tfidf_logreg") -> None:
        self._name = name
        # Avoid static import issues in strict type-checker setups.
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
        sklearn_linear = importlib.import_module("sklearn.linear_model")

        Pipeline = getattr(sklearn_pipeline, "Pipeline")
        TfidfVectorizer = getattr(sklearn_text, "TfidfVectorizer")
        LogisticRegression = getattr(sklearn_linear, "LogisticRegression")

        pipe = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=200_000)),
                ("clf", LogisticRegression(max_iter=1000, n_jobs=None)),
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

        # sklearn stores class order in classifier
        class_order: list[str] | None = None
        try:
            named_steps = getattr(pipe, "named_steps", None)
            clf = named_steps["clf"] if isinstance(named_steps, dict) and "clf" in named_steps else None
            class_order = list(getattr(clf, "classes_")) if clf is not None else None
        except Exception:
            class_order = None

        allowed = set(allowed_labels)
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

            out.append(
                Prediction(
                    pred_label=lab_s,
                    confidence=conf,
                    probs=probs_map,
                    usage=Usage(in_tokens=None, out_tokens=None),
                    raw=None,
                )
            )
        return out

