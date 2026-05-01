# pyright: basic

from __future__ import annotations

from time import sleep
from types import SimpleNamespace
from typing import Any

import dspy
import requests


def request_json(method: str, url: str, payload: dict[str, Any] | None = None, *, timeout: int = 30, retries: int = 3) -> dict[str, Any]:
	last_error: Exception | None = None
	for attempt in range(1, retries + 1):
		try:
			response = requests.request(method, url, json=payload, timeout=timeout)
			response.raise_for_status()
			return response.json()
		except Exception as exc:
			last_error = exc
			if attempt < retries:
				sleep(2 * attempt)
	raise RuntimeError(f"Request failed: {last_error}")


def list_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
	tags = request_json("GET", f"{base_url}/api/tags", timeout=10, retries=2)
	return [m.get("name", "") for m in tags.get("models", []) if m.get("name")]


class ManualOllamaLM(dspy.LM):
	def __init__(
		self,
		model: str,
		*,
		base_url: str = "http://localhost:11434",
		temperature: float = 0.0,
		max_tokens: int = 256,
		timeout: int = 60,
		retries: int = 3,
	):
		super().__init__(model=model, model_type="chat", temperature=temperature, max_tokens=max_tokens, cache=True)
		self.chat_url = f"{base_url}/api/chat"
		self.timeout = timeout
		self.retries = retries

	def forward(self, prompt=None, messages=None, **kwargs):
		merged_kwargs = {**self.kwargs, **kwargs}
		chat_messages = messages or [{"role": "user", "content": prompt or ""}]

		options: dict[str, Any] = {}
		if merged_kwargs.get("temperature") is not None:
			options["temperature"] = merged_kwargs["temperature"]
		if merged_kwargs.get("max_tokens") is not None:
			options["num_predict"] = int(merged_kwargs["max_tokens"])

		payload = {
			"model": self.model,
			"messages": chat_messages,
			"stream": False,
			"options": options,
		}
		response_payload = request_json(
			"POST",
			self.chat_url,
			payload=payload,
			timeout=self.timeout,
			retries=self.retries,
		)

		content = str((response_payload.get("message") or {}).get("content", "")).strip()
		choice = SimpleNamespace(message=SimpleNamespace(content=content))
		return SimpleNamespace(
			choices=[choice],
			model=response_payload.get("model", self.model),
			usage=response_payload.get("usage", {}),
			_hidden_params={},
		)


def configure_dspy_ollama_manual(
	model: str = "phi3:3.8b",
	*,
	base_url: str = "http://localhost:11434",
) -> ManualOllamaLM:
	models = list_ollama_models(base_url=base_url)
	if model not in models:
		raise RuntimeError(f"Model '{model}' not found in Ollama models: {models[:10]}")

	lm = ManualOllamaLM(model=model, base_url=base_url, temperature=0.0, max_tokens=256)
	dspy.settings.configure(lm=lm)
	return lm
