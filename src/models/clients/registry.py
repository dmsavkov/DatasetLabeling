# pyright: basic
"""
Unified LLM client resolution: exact `model_id` → async HTTP client.

Routing is table-driven (`LLMClientRegistry`). Callers use only `get_async_llm_client`;
there is no separate “OpenAI vs Hugging Face” API surface—unknown ids fail fast.

New providers or families are added by registering additional `(model_id, factory)` pairs.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

from openai import AsyncOpenAI

from src.utils.env import load_dotenv_if_present

load_dotenv_if_present()

_GOOGLE_GENAI_OPENAI_BASE_URL: Final[str] = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)

_HF_INFERENCE_OPENAI_BASE_URL: Final[str] = "https://router.huggingface.co/v1"

_DEFAULT_MAX_RETRIES: Final[int] = 20


def _google_genai_api_key() -> str:
    for env in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(env)
        if v:
            return v
    raise ValueError(
        "No API key found for Google Generative AI. Set GEMINI_API_KEY or GOOGLE_API_KEY "
        "(OPENAI_API_KEY is accepted if it holds the same key)."
    )


def _make_google_openai_compat_client() -> AsyncOpenAI:
    """Factory for models using Google’s OpenAI-compatible Generative Language endpoint."""

    return AsyncOpenAI(
        api_key=_google_genai_api_key(),
        base_url=_GOOGLE_GENAI_OPENAI_BASE_URL,
        max_retries=_DEFAULT_MAX_RETRIES,
    )

def _hf_api_key() -> str:
    # Hugging Face standard env var is HUGGINGFACEHUB_API_TOKEN; HF_TOKEN is also common.
    for env in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        v = os.environ.get(env)
        if v:
            return v
    raise ValueError("No Hugging Face token found. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN.")


def _make_hf_inference_openai_compat_client() -> AsyncOpenAI:
    """
    Factory for Hugging Face Inference Providers OpenAI-compatible endpoint.

    See docs: use OpenAI SDK with `base_url="https://router.huggingface.co/v1"`.
    Model is selected per request via `model=...` (exact model id).
    """

    return AsyncOpenAI(
        api_key=_hf_api_key(),
        base_url=_HF_INFERENCE_OPENAI_BASE_URL,
        max_retries=_DEFAULT_MAX_RETRIES,
    )


@dataclass
class LLMClientRegistry:
    """
    Maps exact model id strings to zero-arg factories that build transport clients.

    Factories may share implementation when multiple ids use the same endpoint/auth;
    registration still lists each supported id explicitly.
    """

    _factories: dict[str, Callable[[], AsyncOpenAI]] = field(default_factory=dict)

    def register(self, model_id: str, factory: Callable[[], AsyncOpenAI]) -> None:
        mid = model_id.strip()
        if mid in self._factories:
            raise ValueError(f"Duplicate model_id registration: {mid!r}")
        self._factories[mid] = factory

    def get_client(self, model_id: str) -> AsyncOpenAI:
        mid = model_id.strip()
        if mid not in self._factories:
            raise ValueError(
                f"Unknown model_id {mid!r}. Registered ids: {sorted(self._factories)}"
            )
        return self._factories[mid]()

    def registered_ids(self) -> frozenset[str]:
        return frozenset(self._factories.keys())


default_registry = LLMClientRegistry()


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """
    Provider/prompting capability hints used to build request payloads robustly.

    Not all OpenAI-compatible chat endpoints support the same message roles or
    JSON-mode features; predictors use these hints to adapt.
    """

    supports_system_prompt: bool = True


_MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
    # Gemma IT via Google endpoint behaves like a single-user endpoint and
    # does not reliably honor system prompts.
    "gemma-3-4b-it": ModelCapabilities(supports_system_prompt=False),
    "gemma-3-27b-it": ModelCapabilities(supports_system_prompt=False),
    "gemma-4-31b-it": ModelCapabilities(supports_system_prompt=False),
}


def model_supports_system_prompt(model_id: str) -> bool:
    mid = model_id.strip()
    cap = _MODEL_CAPABILITIES.get(mid)
    return cap.supports_system_prompt if cap is not None else True


def _bootstrap_default_registry() -> None:
    _google = _make_google_openai_compat_client
    _hf = _make_hf_inference_openai_compat_client
    for mid in (
        "gemini-3.1-flash-lite-preview",
        "gemma-3-4b-it",
    ):
        default_registry.register(mid, _google)

    # HF Inference committee members (exact ids).
    for mid in (
        "Qwen/Qwen2.5-1.5B-Instruct",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
    ):
        default_registry.register(mid, _hf)


_bootstrap_default_registry()


def get_async_llm_client(model_id: str) -> AsyncOpenAI:
    """
    Return an async LLM client for the given exact registered `model_id`.

    This is the single entrypoint used by experiment code; routing is handled by
    `default_registry`. Lifecycle: typical HTTP usage does not require explicit close.
    """
    return default_registry.get_client(model_id)
