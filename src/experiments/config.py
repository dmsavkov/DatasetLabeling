# pyright: basic
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Union

import yaml
from pydantic import BaseModel, Field, model_validator

# What the user can specify in an experiment config
ThinkingLevelLiteral = Literal["off", "low", "high"]


class GoogleGenaiChatParams(BaseModel):
    """Google AI Studio / Gemini API via `google.genai` SDK (native thinking + usage_metadata)."""

    model_id: str
    prompt_id: str = "baseline_v1"
    batch_size: int = 5
    max_concurrency: int = 5
    temperature: float = 0.0
    max_tokens: int | None = None
    retries: int = 20
    thinking_level: ThinkingLevelLiteral = "off"
    include_thoughts: bool = False


class GoogleOpenAIChatParams(BaseModel):
    model_id: str
    prompt_id: str = "baseline_v1"
    batch_size: int = 5
    """Rows grouped into one HTTP completion request (prompt batching)."""

    max_concurrency: int = 5
    """Parallel completion calls; separate from batch_size (HTTP-level parallelism)."""

    temperature: float = 0.0
    max_tokens: int | None = None
    retries: int = 20


class SklearnSvmParams(BaseModel):
    pass


class SklearnLogRegParams(BaseModel):
    pass


class TfidfXgbParams(BaseModel):
    min_df: float = 1
    max_df: float = 1.0
    max_features: int = 200_000
    ngram_range: tuple[int, int] = (1, 2)
    # Gradient boosting knobs (approximation of "XGB" without the xgboost dependency).
    n_estimators: int = 300
    learning_rate: float = 0.1
    max_depth: int = 3


class EmbUmapHeadParams(BaseModel):
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    reducer_dim: int = 10
    head_kind: Literal["xgb", "logreg", "knn"] = "xgb"

    # Preferred: pass model kwargs directly to the head builder.
    head_kwargs: dict[str, object] = Field(default_factory=dict)

    # Backward-compat: older configs used explicit xgb/knn fields.
    xgb_n_estimators: int | None = Field(default=None, exclude=True)
    xgb_learning_rate: float | None = Field(default=None, exclude=True)
    xgb_max_depth: int | None = Field(default=None, exclude=True)
    knn_n_neighbors: int | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _fill_head_kwargs_from_legacy(self) -> "EmbUmapHeadParams":
        if self.head_kwargs:
            return self
        if self.head_kind == "xgb":
            self.head_kwargs = {
                "n_estimators": int(self.xgb_n_estimators or 300),
                "learning_rate": float(self.xgb_learning_rate or 0.1),
                "max_depth": int(self.xgb_max_depth or 3),
            }
        elif self.head_kind == "knn":
            self.head_kwargs = {"n_neighbors": int(self.knn_n_neighbors or 5)}
        else:
            self.head_kwargs = {"max_iter": 2000}
        return self


class SetFitParams(BaseModel):
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_steps: int | None = 2000
    epochs: int = 1


ModelKind = Literal[
    "google_genai_chat",
    "google_openai_chat",
    "committee_llm",
    "sklearn_svm",
    "sklearn_logreg",
    "tfidf_xgb",
    "emb_umap_head",
    "setfit",
]


class GoogleGenaiChatSpec(BaseModel):
    kind: Literal["google_genai_chat"]
    params: GoogleGenaiChatParams


class GoogleOpenAIChatSpec(BaseModel):
    kind: Literal["google_openai_chat"]
    params: GoogleOpenAIChatParams


class CommitteeLLMParams(BaseModel):
    member_model_ids: list[str]
    prompt_id: str = "baseline_v1"
    batch_size: int = 5
    max_concurrency: int = 5
    temperature: float = 0.0
    max_tokens: int | None = None
    retries: int = 20


class CommitteeLLMSpec(BaseModel):
    kind: Literal["committee_llm"]
    params: CommitteeLLMParams


class SklearnSvmSpec(BaseModel):
    kind: Literal["sklearn_svm"]
    params: SklearnSvmParams = Field(default_factory=SklearnSvmParams)


class SklearnLogRegSpec(BaseModel):
    kind: Literal["sklearn_logreg"]
    params: SklearnLogRegParams = Field(default_factory=SklearnLogRegParams)


class TfidfXgbSpec(BaseModel):
    kind: Literal["tfidf_xgb"]
    params: TfidfXgbParams = Field(default_factory=TfidfXgbParams)


class EmbUmapHeadSpec(BaseModel):
    kind: Literal["emb_umap_head"]
    params: EmbUmapHeadParams = Field(default_factory=EmbUmapHeadParams)


class SetFitSpec(BaseModel):
    kind: Literal["setfit"]
    params: SetFitParams = Field(default_factory=SetFitParams)


ModelSpec = Union[
    GoogleGenaiChatSpec,
    GoogleOpenAIChatSpec,
    CommitteeLLMSpec,
    SklearnSvmSpec,
    SklearnLogRegSpec,
    TfidfXgbSpec,
    EmbUmapHeadSpec,
    SetFitSpec,
]


class ExperimentConfig(BaseModel):
    name: str
    seed: int = 42
    train_data: str
    test_data: str
    output_dir: str
    model: ModelSpec = Field(..., discriminator="kind")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    elif p.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        raise ValueError("Config must be .yaml/.yml or .json")
    return ExperimentConfig.model_validate(payload)

