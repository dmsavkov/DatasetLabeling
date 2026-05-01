from __future__ import annotations

import pytest
from openai import AsyncOpenAI
from huggingface_hub import AsyncInferenceClient

from src.models.clients.dispatch import get_llm_backend


def test_get_async_llm_client_unknown_id() -> None:
    # Unknown ids are treated as HF hub models, but require a token.
    import os
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)
    with pytest.raises(ValueError, match="Hugging Face token"):
        _ = get_llm_backend("unknown-model-not-registered")


def test_default_registry_lists_gemini_and_gemma() -> None:
    # Dispatch knows Google models explicitly.
    b1 = get_llm_backend("gemini-3.1-flash-lite-preview")
    assert b1.kind == "openai_compat_chat"
    b2 = get_llm_backend("gemma-3-4b-it")
    assert b2.kind == "openai_compat_chat"


def test_get_async_llm_client_returns_openai_instance_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-registry")
    b = get_llm_backend("gemini-3.1-flash-lite-preview")
    assert b.kind == "openai_compat_chat"
    assert isinstance(b.client, AsyncOpenAI)


def test_get_async_llm_client_hf_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="Hugging Face token"):
        _ = get_llm_backend("Qwen/Qwen2.5-1.5B-Instruct")


def test_get_async_llm_client_hf_works_with_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    b = get_llm_backend("Qwen/Qwen2.5-1.5B-Instruct")
    assert b.kind == "hf_inference_textgen"
    assert isinstance(b.client, AsyncInferenceClient)
