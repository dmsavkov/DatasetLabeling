# pyright: basic
from __future__ import annotations

import numpy as np
import pytest

from src.experiments.config import EmbUmapHeadParams, ExperimentConfig, SetFitParams, TfidfXgbParams
from src.experiments.run import build_predictor
from src.eval.harness import evaluate_predictor_on_df
from src.models.baselines.emb_umap_head import EmbUmapHeadPredictor
from src.models.baselines.setfit import SetFitPredictor
from src.models.baselines.tfidf_xgb import TfidfXgbPredictor
from src.models.interfaces import Prediction


def test_tfidf_xgb_predictor_toy() -> None:
    p = TfidfXgbPredictor()
    texts = ["aa aa", "aa bb", "bb bb", "bb aa"]
    labels = ["L1", "L2", "L2", "L1"]
    _ = p.fit(texts, labels)
    preds = p.predict(["aa aa", "bb bb"], allowed_labels=["L1", "L2"])
    assert len(preds) == 2
    assert isinstance(preds[0], Prediction)


def test_emb_umap_head_predictor_toy_with_fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.models.embeddings import fastembed_backend

    class FakeFastEmbedder:
        def __init__(self, model_name: str) -> None:
            self.model_name: str = model_name

        def embed(self, texts: list[str]):
            # Deterministic 4d embeddings based on text length.
            arr = []
            for t in texts:
                x = float(len(t))
                arr.append([x, x + 1.0, x + 2.0, x + 3.0])
            return np.asarray(arr, dtype=float)

    monkeypatch.setattr(fastembed_backend, "FastEmbedder", FakeFastEmbedder)

    p = EmbUmapHeadPredictor(reducer_dim=2, head_kind="logreg", head_kwargs={"max_iter": 2000})
    texts = ["aa", "aaa", "bbbb", "cc", "dddd", "e"]
    labels = ["A", "A", "B", "A", "B", "B"]
    _ = p.fit(texts, labels)
    out = p.predict(["aa", "bbbb"], allowed_labels=["A", "B"])
    assert len(out) == 2
    assert out[0].pred_label in {"A", "B", None}


def test_setfit_predictor_toy_with_fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid heavy SetFit training in unit tests by monkeypatching the SetFit API.
    import setfit as setfit_mod
    import datasets as datasets_mod

    class FakeModel:
        def __init__(self, labels: list[str]) -> None:
            self.labels = labels

        def predict(self, texts: list[str]):
            # Mimic real SetFit behavior: can be list[str] or np.ndarray.
            return np.asarray([self.labels[0] for _ in texts], dtype=object)

    class FakeSetFitModel:
        @staticmethod
        def from_pretrained(_mid: str, labels: list[str] | None = None):
            return FakeModel(labels or [])

    class FakeTrainingArguments:
        def __init__(self, **_kw: object) -> None:
            pass

    class FakeTrainer:
        def __init__(self, **_kw: object) -> None:
            pass

        def train(self) -> None:
            return None

    class FakeDataset:
        @staticmethod
        def from_dict(_d: dict[str, list[str]]):
            return object()

    monkeypatch.setattr(setfit_mod, "SetFitModel", FakeSetFitModel)
    monkeypatch.setattr(setfit_mod, "Trainer", FakeTrainer)
    monkeypatch.setattr(setfit_mod, "TrainingArguments", FakeTrainingArguments)
    monkeypatch.setattr(datasets_mod, "Dataset", FakeDataset)

    p = SetFitPredictor(max_steps=2, epochs=1)
    texts = ["one", "two", "three", "four", "five"]
    labels = ["A", "A", "B", "B", "B"]
    _ = p.fit(texts, labels)
    out = p.predict(["one", "three"], allowed_labels=["A", "B"])
    assert len(out) == 2
    assert out[0].pred_label in {"A", "B", None}


def test_build_predictor_for_classical_kinds() -> None:
    base = {"name": "t", "seed": 1, "train_data": "dummy", "test_data": "dummy", "output_dir": "dummy"}

    cfg_tfidf = ExperimentConfig.model_validate({**base, "model": {"kind": "tfidf_xgb", "params": TfidfXgbParams().model_dump()}})
    p1 = build_predictor(cfg_tfidf, train_df=None)
    assert isinstance(p1, TfidfXgbPredictor)

    cfg_emb = ExperimentConfig.model_validate(
        {**base, "model": {"kind": "emb_umap_head", "params": EmbUmapHeadParams(reducer_dim=2, head_kind="knn", head_kwargs={"n_neighbors": 3}).model_dump()}}
    )
    p2 = build_predictor(cfg_emb, train_df=None)
    assert isinstance(p2, EmbUmapHeadPredictor)

    cfg_setfit = ExperimentConfig.model_validate({**base, "model": {"kind": "setfit", "params": SetFitParams(max_steps=1, epochs=1).model_dump()}})
    p3 = build_predictor(cfg_setfit, train_df=None)
    assert isinstance(p3, SetFitPredictor)


def test_classical_predictors_execute_end_to_end_on_toy_df(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import pandas as pd

    # Embeddings predictor needs deterministic embeddings.
    from src.models.embeddings import fastembed_backend

    class FakeFastEmbedder:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed(self, texts: list[str]):
            arr = []
            for t in texts:
                x = float(len(t))
                arr.append([x, x + 1.0, x + 2.0, x + 3.0])
            return np.asarray(arr, dtype=float)

    monkeypatch.setattr(fastembed_backend, "FastEmbedder", FakeFastEmbedder)

    # SetFit predictor: avoid heavy training; keep predict output array-like.
    import setfit as setfit_mod
    import datasets as datasets_mod

    class FakeModel:
        def __init__(self, labels: list[str]) -> None:
            self.labels = labels

        def predict(self, texts: list[str]):
            return np.asarray([self.labels[0] for _ in texts], dtype=object)

    class FakeSetFitModel:
        @staticmethod
        def from_pretrained(_mid: str, labels: list[str] | None = None):
            return FakeModel(labels or [])

    class FakeTrainingArguments:
        def __init__(self, **_kw: object) -> None:
            pass

    class FakeTrainer:
        def __init__(self, **_kw: object) -> None:
            pass

        def train(self) -> None:
            return None

    class FakeDataset:
        @staticmethod
        def from_dict(_d: dict[str, list[str]]):
            return object()

    monkeypatch.setattr(setfit_mod, "SetFitModel", FakeSetFitModel)
    monkeypatch.setattr(setfit_mod, "Trainer", FakeTrainer)
    monkeypatch.setattr(setfit_mod, "TrainingArguments", FakeTrainingArguments)
    monkeypatch.setattr(datasets_mod, "Dataset", FakeDataset)

    df = pd.DataFrame({"text": ["aa", "bbb", "cccc", "dd"], "true_label": ["A", "B", "A", "B"]})
    allowed = ["A", "B"]

    predictors = [
        ("tfidf_xgb", TfidfXgbPredictor()),
        ("emb_umap_head", EmbUmapHeadPredictor(reducer_dim=2, head_kind="logreg", head_kwargs={"max_iter": 2000})),
        ("setfit", SetFitPredictor(max_steps=None, epochs=1)),
    ]

    for name, p in predictors:
        _ = p.fit(df["text"].tolist(), df["true_label"].tolist())
        res = evaluate_predictor_on_df(
            p,
            df=df,
            allowed_labels=allowed,
            dataset_name="toy",
            split_name="test",
            tier_size=len(df),
            output_dir=tmp_path / f"out_{name}",
        )
        assert len(res.predictions_df) == len(df)

