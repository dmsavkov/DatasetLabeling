# pyright: basic
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, Literal

from huggingface_hub import AsyncInferenceClient
from openai import AsyncOpenAI

from src.utils.env import load_dotenv_if_present

load_dotenv_if_present()


LLMBackendKind = Literal["openai_compat_chat", "hf_inference_textgen"]


@dataclass(frozen=True, slots=True)
class LLMBackend:
    kind: LLMBackendKind
    client: object


_GOOGLE_GENAI_OPENAI_BASE_URL: Final[str] = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _google_genai_api_key() -> str:
    for env in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(env)
        if v:
            return v
    raise ValueError(
        "No API key found for Google Generative AI. Set GEMINI_API_KEY or GOOGLE_API_KEY "
        "(OPENAI_API_KEY is accepted if it holds the same key)."
    )


def _hf_token() -> str:
    for env in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        v = os.environ.get(env)
        if v:
            return v
    raise ValueError("No Hugging Face token found. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN.")


def get_llm_backend(model_id: str) -> LLMBackend:
    """
    Backend-aware client dispatch.

    - Google models use OpenAI-compatible Chat Completions via `AsyncOpenAI`.
    - HF models use `AsyncInferenceClient` (text generation), which works even
      when a model is not exposed as a “chat model” in the router.
    """

    mid = model_id.strip()

    if mid in {"gemini-3.1-flash-lite-preview", "gemma-3-4b-it"}:
        client = AsyncOpenAI(api_key=_google_genai_api_key(), base_url=_GOOGLE_GENAI_OPENAI_BASE_URL, max_retries=20)
        return LLMBackend(kind="openai_compat_chat", client=client)

    # Default: treat other ids as HF Hub models.
    # Note: availability depends on token + provider settings; errors are surfaced at call-time.
    client = AsyncInferenceClient(model=mid, token=_hf_token())
    return LLMBackend(kind="hf_inference_textgen", client=client)


def get_google_openai_chat_backend(model_id: str) -> LLMBackend:
    """
    Strict dispatch for configs declared as `google_openai_chat`.
    """

    mid = model_id.strip()
    if mid not in {"gemini-3.1-flash-lite-preview", "gemma-3-4b-it"}:
        raise ValueError(f"Unknown Google model_id {mid!r}. Supported: gemini-3.1-flash-lite-preview, gemma-3-4b-it")
    return get_llm_backend(mid)

