# pyright: basic
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal, Protocol


class _SklearnLikeClassifier(Protocol):
    def fit(self, X, y): ...  # noqa: ANN001, D401
    def predict(self, X): ...  # noqa: ANN001, D401


HeadKind = Literal["xgb", "logreg", "knn"]


@dataclass(frozen=True, slots=True)
class XgbHeadParams:
    n_estimators: int = 300
    learning_rate: float = 0.1
    max_depth: int = 3
    n_jobs: int | None = None


@dataclass(frozen=True, slots=True)
class LogRegHeadParams:
    max_iter: int = 2000


@dataclass(frozen=True, slots=True)
class KnnHeadParams:
    n_neighbors: int = 5


def build_head(kind: HeadKind, **kwargs: Any) -> _SklearnLikeClassifier:
    if kind == "xgb":
        xgb_mod = importlib.import_module("xgboost")
        XGBClassifier = getattr(xgb_mod, "XGBClassifier")
        return XGBClassifier(**kwargs, eval_metric="mlogloss")

    if kind == "knn":
        sklearn_neighbors = importlib.import_module("sklearn.neighbors")
        KNeighborsClassifier = getattr(sklearn_neighbors, "KNeighborsClassifier")
        return KNeighborsClassifier(**kwargs)

    sklearn_linear = importlib.import_module("sklearn.linear_model")
    LogisticRegression = getattr(sklearn_linear, "LogisticRegression")
    return LogisticRegression(**kwargs)

