# pyright: basic
from __future__ import annotations

import importlib
import time

from src.models.interfaces import Prediction, Usage


class SklearnTfidfSvmPredictor:
    """
    TF-IDF + LinearSVC baseline (no probabilities by default).
    """

    def __init__(self, *, name: str = "sklearn_tfidf_svm") -> None:
        self._name = name
        self._pipeline: object | None = None
        self.train_time_s: float | None = None
        self.infer_time_s: float | None = None

    @property
    def name(self) -> str:
        return self._name

    def fit(self, texts: list[str], labels: list[str]) -> None:
        start = time.perf_counter()

        sklearn_pipeline = importlib.import_module("sklearn.pipeline")
        sklearn_text = importlib.import_module("sklearn.feature_extraction.text")
        sklearn_svm = importlib.import_module("sklearn.svm")

        Pipeline = getattr(sklearn_pipeline, "Pipeline")
        TfidfVectorizer = getattr(sklearn_text, "TfidfVectorizer")
        LinearSVC = getattr(sklearn_svm, "LinearSVC")

        pipe = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=200_000)),
                ("clf", LinearSVC()),
            ]
        )
        pipe.fit(texts, labels)
        self._pipeline = pipe
        self.train_time_s = float(time.perf_counter() - start)

    def predict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
        if self._pipeline is None:
            raise RuntimeError("Model is not fit yet. Call fit(...) first.")
        start = time.perf_counter()
        pred = getattr(self._pipeline, "predict")(texts)
        self.infer_time_s = float(time.perf_counter() - start)

        allowed = set(allowed_labels)
        out: list[Prediction] = []
        for lab in pred.tolist():
            lab_s = str(lab)
            if lab_s not in allowed:
                lab_s = None
            out.append(Prediction(pred_label=lab_s, confidence=None, probs=None, reason=None, usage=Usage(None, None), raw=None))
        return out

