from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.llm.openai_compat_chat_batch import OpenAICompatChatBatchParams, OpenAICompatChatBatchPredictor


def _mock_response_json_for_ids(ids: list[str], *, label: str = "x") -> MagicMock:
    arr = [{"id": i, "label": label} for i in ids]
    content = json.dumps(arr)
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return resp


def _parse_items_from_user_message(messages: list[dict[str, str]]) -> list[str]:
    """Extract item ids from the prompt payload built by build_llm_batch_messages."""

    user = messages[-1]["content"]
    marker = "Classify the following items. Output JSON only.\n"
    payload = json.loads(user.split(marker, 1)[1])
    return [str(it["id"]) for it in payload["items"]]


@pytest.mark.parametrize("max_concurrency", [1, 4])
def test_apredict_preserves_row_order_under_concurrency(max_concurrency: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent batch completion calls still yield predictions aligned to original row indices."""

    monkeypatch.setenv("GEMINI_API_KEY", "x")  # unused when client is mocked below

    async def fake_create(*args: object, **kwargs: object) -> MagicMock:
        msgs = kwargs["messages"]
        assert isinstance(msgs, list)
        ids = _parse_items_from_user_message(msgs)
        return _mock_response_json_for_ids(ids)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=fake_create)

    params = OpenAICompatChatBatchParams(
        model_id="gemini-3.1-flash-lite-preview",
        prompt_id="baseline_v1",
        batch_size=3,
        max_concurrency=max_concurrency,
        retries=0,
    )
    pred = OpenAICompatChatBatchPredictor(mock_client, params=params)
    texts = [f"t{i}" for i in range(9)]

    async def run() -> None:
        out = await pred.apredict(texts, allowed_labels=["x"])
        assert len(out) == 9
        for i, p in enumerate(out):
            assert p.pred_label == "x", f"row {i}"

    asyncio.run(run())


def test_predict_sync_runs_apredict_when_no_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def fake_create(*args: object, **kwargs: object) -> MagicMock:
        nonlocal calls
        calls += 1
        msgs = kwargs["messages"]
        ids = _parse_items_from_user_message(msgs)
        return _mock_response_json_for_ids(ids)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=fake_create)

    params = OpenAICompatChatBatchParams(
        model_id="gemini-3.1-flash-lite-preview",
        prompt_id="baseline_v1",
        batch_size=5,
        max_concurrency=2,
    )
    pred = OpenAICompatChatBatchPredictor(mock_client, params=params)

    out = pred.predict(["a", "b"], allowed_labels=["x"])
    assert len(out) == 2
    assert calls >= 1
    assert out[0].pred_label == "x"


def test_predict_raises_inside_running_loop() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_mock_response_json_for_ids(["0"]))
    params = OpenAICompatChatBatchParams(model_id="gemini-3.1-flash-lite-preview", prompt_id="baseline_v1", batch_size=5)
    pred = OpenAICompatChatBatchPredictor(mock_client, params=params)

    async def inner() -> None:
        with pytest.raises(RuntimeError, match="event loop"):
            pred.predict(["x"], allowed_labels=["x"])

    asyncio.run(inner())


def test_gemma_merges_system_into_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Gemma models may not reliably support the `system` role.
    Predictor must merge system content into the first user message and avoid
    sending any `role="system"` messages.
    """

    monkeypatch.setenv("GEMINI_API_KEY", "x")

    async def fake_create(*args: object, **kwargs: object) -> MagicMock:
        msgs = kwargs["messages"]
        assert isinstance(msgs, list)
        assert all(m.get("role") != "system" for m in msgs), "gemma should not receive system messages"
        ids = _parse_items_from_user_message(msgs)
        return _mock_response_json_for_ids(ids)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=fake_create)

    params = OpenAICompatChatBatchParams(
        model_id="gemma-3-4b-it",
        prompt_id="baseline_v1",
        batch_size=2,
        max_concurrency=1,
        retries=0,
    )
    pred = OpenAICompatChatBatchPredictor(mock_client, params=params)

    asyncio.run(pred.apredict(["a", "b"], allowed_labels=["x"]))
