# pyright: basic
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.experiments.config import ExperimentConfig
from src.experiments.run import build_predictor
from src.models.interfaces import Usage, split_call_usage_across_rows
from src.models.llm.google_genai_batch import GoogleGenaiBatchParams, GoogleGenaiBatchPredictor


class _FakeUsage:
    prompt_token_count = 10
    candidates_token_count = 5
    thoughts_token_count = 3

    def model_dump(self, *, mode: str | None = None) -> dict[str, object]:
        return {
            "prompt_token_count": self.prompt_token_count,
            "candidates_token_count": self.candidates_token_count,
            "thoughts_token_count": self.thoughts_token_count,
        }


def _fake_response(*, answer_json: str, with_thought: bool) -> SimpleNamespace:
    parts = []
    if with_thought:
        parts.append(SimpleNamespace(text="step one", thought=True, thought_signature=None))
    parts.append(SimpleNamespace(text=answer_json, thought=False, thought_signature=None))
    content = SimpleNamespace(parts=parts)
    cand = SimpleNamespace(content=content)
    return SimpleNamespace(
        candidates=[cand],
        usage_metadata=_FakeUsage(),
        model_version="fake-model",
        response_id="fake-id",
        prompt_feedback=None,
    )


def test_google_genai_batch_apredict_parses_usage_and_thought(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")

    answer = '[{"id": "0", "label": "alpha"}, {"id": "1", "label": "beta"}]'

    class FakeModels:
        async def generate_content(self, **_kw: object) -> SimpleNamespace:
            return _fake_response(answer_json=answer, with_thought=True)

    class FakeAio:
        models = FakeModels()

    class FakeRoot:
        aio = FakeAio()

    monkeypatch.setattr("src.models.llm.google_genai_batch.genai.Client", lambda **kw: FakeRoot())

    p = GoogleGenaiBatchPredictor(
        params=GoogleGenaiBatchParams(
            model_id="gemini-test",
            batch_size=2,
            max_concurrency=2,
            retries=1,
            thinking_level="high",
            include_thoughts=True,
        ),
    )
    out = asyncio.run(p.apredict(["hello", "world"], allowed_labels=["alpha", "beta"]))
    assert len(out) == 2
    assert out[0].pred_label == "alpha"
    assert out[1].pred_label == "beta"
    assert out[0].usage is not None
    assert out[0].usage.in_tokens == 5  # 10 prompt tokens split across 2 rows
    assert out[0].usage.out_tokens == 4  # 8 = 5 + 3 thoughts, split across 2 rows
    assert out[1].usage is not None
    assert out[1].usage.in_tokens == 5
    assert out[1].usage.out_tokens == 4
    assert out[0].reason is not None
    assert isinstance(out[0].raw, dict)
    assert out[0].raw.get("usage_metadata")


def test_split_call_usage_across_rows_remainder_goes_to_first_rows() -> None:
    u = Usage(in_tokens=10, out_tokens=5)
    parts = split_call_usage_across_rows(u, 3)
    assert [p.in_tokens for p in parts] == [4, 3, 3]
    assert [p.out_tokens for p in parts] == [2, 2, 1]
    assert sum(p.in_tokens or 0 for p in parts) == 10
    assert sum(p.out_tokens or 0 for p in parts) == 5


def test_build_predictor_google_genai_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setattr(
        "src.models.llm.google_genai_batch.genai.Client",
        lambda **kw: SimpleNamespace(aio=SimpleNamespace(models=MagicMock())),
    )

    cfg = ExperimentConfig(
        name="t",
        seed=1,
        train_data="dummy",
        test_data="dummy",
        output_dir="dummy",
        model={
            "kind": "google_genai_chat",
            "params": {
                "model_id": "gemini-3.1-flash-lite-preview",
                "prompt_id": "baseline_v1",
                "batch_size": 3,
                "max_concurrency": 2,
                "thinking_level": "low",
                "include_thoughts": True,
                "retries": 2,
            },
        },
    )
    pred = build_predictor(cfg, train_df=None)
    assert isinstance(pred, GoogleGenaiBatchPredictor)
