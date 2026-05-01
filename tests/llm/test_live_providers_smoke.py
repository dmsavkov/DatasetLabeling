# pyright: basic
from __future__ import annotations

import asyncio
import os

import pytest

from src.models.clients.dispatch import get_google_openai_chat_backend, get_llm_backend
from src.utils.env import load_dotenv_if_present

load_dotenv_if_present()


pytestmark = pytest.mark.llm


def _enabled() -> bool:
    return os.environ.get("RUN_LLM_TESTS", "").strip() == "1"


@pytest.mark.skipif(not _enabled(), reason="Set RUN_LLM_TESTS=1 to enable live provider tests")
def test_live_google_chat_smoke() -> None:
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        pytest.skip("No GEMINI_API_KEY/GOOGLE_API_KEY set")

    b = get_google_openai_chat_backend("gemma-3-4b-it")
    client = b.client

    async def run() -> None:
        resp = await client.chat.completions.create(
            model="gemma-3-4b-it",
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=5,
            temperature=0.0,
        )
        assert resp.choices
        assert resp.choices[0].message.content

    asyncio.run(run())


@pytest.mark.skipif(not _enabled(), reason="Set RUN_LLM_TESTS=1 to enable live provider tests")
def test_live_hf_chat_smoke() -> None:
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        pytest.skip("No HF token set")

    model_id = os.environ.get("HF_LIVE_MODEL_ID", "").strip() or "Qwen/Qwen2.5-1.5B-Instruct"
    b = get_llm_backend(model_id)
    if b.kind != "hf_inference_textgen":
        pytest.skip("Not an HF backend in this environment")
    client = b.client

    async def run() -> None:
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                max_tokens=10,
                temperature=0.0,
            )
            assert resp.choices
        except Exception as e:
            # Provider availability is user-account dependent; treat unsupported model as skip.
            pytest.skip(f"HF live model not available: {e!r}")

    asyncio.run(run())

