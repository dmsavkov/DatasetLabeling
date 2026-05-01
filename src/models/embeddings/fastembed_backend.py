# pyright: basic
from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FastEmbedder:
    model_name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=float)
        fastembed_mod = importlib.import_module("fastembed")
        TextEmbedding = getattr(fastembed_mod, "TextEmbedding")
        embedder = TextEmbedding(model_name=self.model_name)
        # Returns iterable[np.ndarray]; convert to 2D array.
        vecs = list(embedder.embed(texts))
        return np.asarray(vecs, dtype=float)

