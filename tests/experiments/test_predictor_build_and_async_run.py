from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.datasets.io import processed_root, save_processed_tier
from src.datasets.schema import SCHEMA
from src.eval.harness import aevaluate_predictor_on_df
from src.experiments.config import ExperimentConfig
from src.experiments.run import arun_experiment, build_predictor, run_experiment
from src.models.baselines.sklearn_svm import SklearnTfidfSvmPredictor
from src.models.interfaces import Prediction
from src.models.llm.openai_compat_chat_batch import OpenAICompatChatBatchPredictor


def test_build_predictor_sklearn_svm() -> None:
    cfg = ExperimentConfig(
        name="t",
        seed=1,
        train_data="dummy",
        test_data="dummy",
        output_dir="dummy",
        model={"kind": "sklearn_svm", "params": {}},
    )
    p = build_predictor(cfg, train_df=None)
    assert isinstance(p, SklearnTfidfSvmPredictor)


def test_build_predictor_google_calls_unified_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    seen: list[str] = []

    def fake_get_google_backend(model_id: str):
        seen.append(model_id)
        return type("B", (), {"kind": "openai_compat_chat", "client": MagicMock()})()

    monkeypatch.setattr("src.experiments.run.get_google_openai_chat_backend", fake_get_google_backend)

    cfg = ExperimentConfig(
        name="t",
        seed=1,
        train_data="dummy",
        test_data="dummy",
        output_dir="dummy",
        model={
            "kind": "google_openai_chat",
            "params": {
                "model_id": "gemini-3.1-flash-lite-preview",
                "prompt_id": "baseline_v1",
                "batch_size": 5,
                "max_concurrency": 4,
                "temperature": 0.0,
                "max_tokens": 100,
                "retries": 1,
            },
        },
    )
    p = build_predictor(cfg, train_df=None)
    assert isinstance(p, OpenAICompatChatBatchPredictor)
    assert seen == ["gemini-3.1-flash-lite-preview"]


def test_build_predictor_committee_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_get_llm_backend(model_id: str):
        seen.append(model_id)
        return type("B", (), {"kind": "openai_compat_chat", "client": MagicMock()})()

    monkeypatch.setattr("src.experiments.run.get_llm_backend", fake_get_llm_backend)

    cfg = ExperimentConfig(
        name="t",
        seed=1,
        train_data="dummy",
        test_data="dummy",
        output_dir="dummy",
        model={
            "kind": "committee_llm",
            "params": {
                "member_model_ids": ["gemma-3-4b-it", "Qwen/Qwen2.5-1.5B-Instruct"],
                "prompt_id": "baseline_v1",
                "batch_size": 5,
                "max_concurrency": 2,
                "temperature": 0.0,
                "max_tokens": None,
                "retries": 1,
            },
        },
    )
    p = build_predictor(cfg, train_df=None)
    assert p.name == "committee_llm"
    assert seen == ["gemma-3-4b-it", "Qwen/Qwen2.5-1.5B-Instruct"]

def test_build_predictor_unknown_google_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = ExperimentConfig(
        name="t",
        seed=1,
        train_data="dummy",
        test_data="dummy",
        output_dir="dummy",
        model={
            "kind": "google_openai_chat",
            "params": {"model_id": "not-in-registry", "prompt_id": "baseline_v1", "batch_size": 5, "max_concurrency": 4},
        },
    )
    with pytest.raises(ValueError, match="Unknown Google model_id"):
        _ = build_predictor(cfg)


def test_aevaluate_calls_apredict(tmp_path: Path) -> None:
    calls: list[int] = []

    class ToyAsyncPred:
        name = "toy_async"

        async def apredict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
            calls.append(len(texts))
            return [Prediction(pred_label=allowed_labels[0]) for _ in texts]

    df = pd.DataFrame(
        {
            SCHEMA.sample_id: ["a", "b"],
            SCHEMA.dataset_name: ["toy", "toy"],
            SCHEMA.text: ["x", "y"],
            SCHEMA.true_label: ["z", "z"],
            SCHEMA.meta_json: ["{}", "{}"],
        }
    )

    async def _go():
        return await aevaluate_predictor_on_df(
            ToyAsyncPred(),
            df=df,
            allowed_labels=["z"],
            dataset_name="toy",
            split_name="exp",
            tier_size=2,
            output_dir=tmp_path / "ev_out",
        )

    out = asyncio.run(_go())
    assert calls == [2]
    assert len(out.predictions_df) == 2
    assert (tmp_path / "ev_out" / "predictions.csv").exists()


async def _run_arun_with_fake_predictor(cfg_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    class FakeLLM:
        name = "fake_llm"

        async def apredict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
            return [Prediction(pred_label=allowed_labels[0]) for _ in texts]

    monkeypatch.setattr("src.experiments.run.build_predictor", lambda _cfg, **_kw: FakeLLM())
    return await arun_experiment(cfg_path)


def test_arun_experiment_async_llm_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train_df = pd.DataFrame(
        {
            SCHEMA.sample_id: [f"tr_{i}" for i in range(10)],
            SCHEMA.dataset_name: ["toy"] * 10,
            SCHEMA.text: [f"x{i}" for i in range(10)],
            SCHEMA.true_label: ["a"] * 10,
            SCHEMA.meta_json: ["{}"] * 10,
        }
    )
    test_df = pd.DataFrame(
        {
            SCHEMA.sample_id: [f"te_{i}" for i in range(4)],
            SCHEMA.dataset_name: ["toy"] * 4,
            SCHEMA.text: [f"y{i}" for i in range(4)],
            SCHEMA.true_label: ["a"] * 4,
            SCHEMA.meta_json: ["{}"] * 4,
        }
    )
    _ = save_processed_tier(
        train_df,
        dataset_name="toy",
        split_name="train_seed",
        tier_size=10,
        seed=1,
        builder="test",
        origin={},
        root=tmp_path,
    )
    _ = save_processed_tier(
        test_df,
        dataset_name="toy",
        split_name="test",
        tier_size=200,
        seed=1,
        builder="test",
        origin={},
        root=tmp_path,
    )

    pr = processed_root(tmp_path)
    cfg_dict = {
        "name": "toy_gemini_branch",
        "seed": 1,
        "train_data": str(pr / "toy" / "train_seed" / "tier_10" / "samples.parquet"),
        "test_data": str(pr / "toy" / "test" / "tier_200" / "samples.parquet"),
        "output_dir": str(tmp_path / "out_async"),
        "model": {
            "kind": "google_openai_chat",
            "params": {
                "model_id": "gemini-3.1-flash-lite-preview",
                "batch_size": 2,
                "max_concurrency": 3,
                "temperature": 0.0,
                "max_tokens": 50,
                "retries": 0,
            },
        },
    }
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(json.dumps(cfg_dict), encoding="utf-8")

    result = asyncio.run(_run_arun_with_fake_predictor(cfg_path, monkeypatch))
    assert (Path(result["output_dir"]) / "predictions.csv").exists()
    assert result["report"]["predictor_name"] == "fake_llm"


def test_run_experiment_sync_entrypoint_llm_fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_experiment uses asyncio.run once at the top; LLM path completes via arun_experiment."""

    train_df = pd.DataFrame(
        {
            SCHEMA.sample_id: [f"tr_{i}" for i in range(6)],
            SCHEMA.dataset_name: ["toy"] * 6,
            SCHEMA.text: ["x"] * 6,
            SCHEMA.true_label: ["b"] * 6,
            SCHEMA.meta_json: ["{}"] * 6,
        }
    )
    test_df = pd.DataFrame(
        {
            SCHEMA.sample_id: [f"te_{i}" for i in range(3)],
            SCHEMA.dataset_name: ["toy"] * 3,
            SCHEMA.text: ["y"] * 3,
            SCHEMA.true_label: ["b"] * 3,
            SCHEMA.meta_json: ["{}"] * 3,
        }
    )
    _ = save_processed_tier(
        train_df,
        dataset_name="toy",
        split_name="train_seed",
        tier_size=10,
        seed=1,
        builder="test",
        origin={},
        root=tmp_path,
    )
    _ = save_processed_tier(
        test_df,
        dataset_name="toy",
        split_name="test",
        tier_size=200,
        seed=1,
        builder="test",
        origin={},
        root=tmp_path,
    )

    pr = processed_root(tmp_path)
    cfg_dict = {
        "name": "sync_llm_fake",
        "seed": 1,
        "train_data": str(pr / "toy" / "train_seed" / "tier_10" / "samples.parquet"),
        "test_data": str(pr / "toy" / "test" / "tier_200" / "samples.parquet"),
        "output_dir": str(tmp_path / "out_sync"),
        "model": {
            "kind": "google_openai_chat",
            "params": {
                "model_id": "gemini-3.1-flash-lite-preview",
                "batch_size": 5,
                "max_concurrency": 2,
                "temperature": 0.0,
                "max_tokens": 50,
                "retries": 0,
            },
        },
    }
    cfg_path = tmp_path / "exp2.json"
    cfg_path.write_text(json.dumps(cfg_dict), encoding="utf-8")

    class FakeLLM:
        name = "fake"

        async def apredict(self, texts: list[str], *, allowed_labels: list[str]) -> list[Prediction]:
            return [Prediction(pred_label=allowed_labels[0]) for _ in texts]

    monkeypatch.setattr("src.experiments.run.build_predictor", lambda _cfg, **_kw: FakeLLM())

    res = run_experiment(cfg_path)
    assert Path(res["output_dir"]).exists()
