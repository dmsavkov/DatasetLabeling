# pyright: basic
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd
from loguru import logger
from openai import AsyncOpenAI
from huggingface_hub import AsyncInferenceClient

from src.datasets.cards import default_card_path, read_card_json
from src.datasets.io import processed_root as processed_root_fn
from src.datasets.schema import SCHEMA, validate_processed_samples_df
from src.eval.harness import aevaluate_predictor_on_df, evaluate_predictor_on_df
from src.models.baselines.sklearn_svm import SklearnTfidfSvmPredictor
from src.models.baselines.sklearn_tfidf import SklearnTfidfLogRegPredictor
from src.models.baselines.tfidf_xgb import TfidfXgbPredictor
from src.models.baselines.emb_umap_head import EmbUmapHeadPredictor
from src.models.baselines.setfit import SetFitPredictor
from src.models.clients.dispatch import get_google_openai_chat_backend, get_llm_backend
from src.models.ensemble.committee import CommitteeMember, CommitteePredictor
from src.models.llm.openai_compat_chat_batch import OpenAICompatChatBatchParams, OpenAICompatChatBatchPredictor
from src.models.llm.hf_inference_textgen_batch import (
    HFInferenceTextGenBatchParams,
    HFInferenceTextGenBatchPredictor,
)

from .config import (
    CommitteeLLMSpec,
    ExperimentConfig,
    GoogleOpenAIChatSpec,
    SklearnLogRegSpec,
    SklearnSvmSpec,
    TfidfXgbSpec,
    EmbUmapHeadSpec,
    SetFitSpec,
    load_experiment_config,
)


def _load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate_processed_samples_df(df)
    return df


def _allowed_labels_for_df(df: pd.DataFrame, *, processed_root: Path | None = None) -> list[str]:
    dataset_name = str(df[SCHEMA.dataset_name].iloc[0])
    pr = processed_root_fn(processed_root)
    cp = default_card_path(processed_root_dir=pr, dataset_name=dataset_name)
    if cp.exists():
        return list(read_card_json(cp).labels)
    return sorted(df[SCHEMA.true_label].astype(str).unique().tolist())


def _few_shot_from_train_df(train_df: pd.DataFrame, *, n: int = 10) -> list[tuple[str, str]]:
    df = train_df.reset_index(drop=True)
    take = min(int(n), len(df))
    out: list[tuple[str, str]] = []
    for i in range(take):
        out.append((str(df[SCHEMA.text].iloc[i]), str(df[SCHEMA.true_label].iloc[i])))
    return out


def build_predictor(cfg: ExperimentConfig, *, train_df: pd.DataFrame | None = None) -> Any:
    m = cfg.model
    if isinstance(m, SklearnSvmSpec):
        return SklearnTfidfSvmPredictor()
    if isinstance(m, SklearnLogRegSpec):
        return SklearnTfidfLogRegPredictor()
    if isinstance(m, GoogleOpenAIChatSpec):
        few_shot = _few_shot_from_train_df(train_df, n=10) if train_df is not None else None
        backend = get_google_openai_chat_backend(m.params.model_id)
        if backend.kind == "openai_compat_chat":
            client = cast(AsyncOpenAI, backend.client)
            params = OpenAICompatChatBatchParams(
                model_id=m.params.model_id,
                prompt_id=m.params.prompt_id,
                few_shot=few_shot,
                batch_size=m.params.batch_size,
                max_concurrency=m.params.max_concurrency,
                temperature=m.params.temperature,
                max_tokens=m.params.max_tokens,
                retries=m.params.retries,
            )
            return OpenAICompatChatBatchPredictor(client, params=params)
        raise RuntimeError("google_openai_chat backend dispatch must return openai_compat_chat")
    if isinstance(m, CommitteeLLMSpec):
        few_shot = _few_shot_from_train_df(train_df, n=10) if train_df is not None else None
        members: list[CommitteeMember] = []
        for mid in list(m.params.member_model_ids):
            backend = get_llm_backend(mid)
            if backend.kind == "openai_compat_chat":
                p = OpenAICompatChatBatchPredictor(
                    cast(AsyncOpenAI, backend.client),
                    params=OpenAICompatChatBatchParams(
                        model_id=mid,
                        prompt_id=m.params.prompt_id,
                        few_shot=few_shot,
                        batch_size=m.params.batch_size,
                        max_concurrency=m.params.max_concurrency,
                        temperature=m.params.temperature,
                        max_tokens=m.params.max_tokens,
                        retries=m.params.retries,
                    ),
                    name=f"member:{mid}",
                )
            else:
                p = HFInferenceTextGenBatchPredictor(
                    cast(AsyncInferenceClient, backend.client),
                    params=HFInferenceTextGenBatchParams(
                        model_id=mid,
                        prompt_id=m.params.prompt_id,
                        few_shot=few_shot,
                        batch_size=m.params.batch_size,
                        max_concurrency=m.params.max_concurrency,
                        temperature=m.params.temperature,
                        max_new_tokens=m.params.max_tokens,
                        retries=m.params.retries,
                    ),
                    name=f"member:{mid}",
                )
            members.append(CommitteeMember(name=mid, predictor=p))
        return CommitteePredictor(members, name="committee_llm")
    if isinstance(m, TfidfXgbSpec):
        p = m.params
        return TfidfXgbPredictor(
            min_df=p.min_df,
            max_df=p.max_df,
            max_features=p.max_features,
            ngram_range=p.ngram_range,
            n_estimators=p.n_estimators,
            learning_rate=p.learning_rate,
            max_depth=p.max_depth,
            name="tfidf_xgb",
        )
    if isinstance(m, EmbUmapHeadSpec):
        p = m.params
        return EmbUmapHeadPredictor(
            embedding_model_id=p.embedding_model_id,
            reducer_dim=p.reducer_dim,
            head_kind=p.head_kind,
            head_kwargs=p.head_kwargs,
            seed=cfg.seed,
            name="emb_umap_head",
        )
    if isinstance(m, SetFitSpec):
        p = m.params
        return SetFitPredictor(
            embedding_model_id=p.embedding_model_id,
            max_steps=p.max_steps,
            epochs=p.epochs,
            name="setfit",
        )
    raise ValueError(f"Unsupported model kind: {getattr(m, 'kind', None)}")


async def arun_experiment(config_path: str | Path) -> dict[str, Any]:
    cfg = load_experiment_config(config_path)
    logger.info("Experiment start: {} ({})", cfg.name, str(config_path))
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.resolved.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

    train_df = _load_parquet(cfg.train_data)
    test_df = _load_parquet(cfg.test_data)
    logger.info(
        "Loaded dataframes: train_rows={}, test_rows={}, dataset={}",
        len(train_df),
        len(test_df),
        str(test_df[SCHEMA.dataset_name].iloc[0]),
    )

    predictor = build_predictor(cfg, train_df=train_df)
    logger.info("Built predictor: {}", getattr(predictor, "name", type(predictor).__name__))

    train_time_s: float | None = None

    def fit_sync() -> None:
        if hasattr(predictor, "fit"):
            logger.info("Fitting predictor on {} rows", len(train_df))
            start = time.perf_counter()
            predictor.fit(train_df[SCHEMA.text].astype(str).tolist(), train_df[SCHEMA.true_label].astype(str).tolist())
            nonlocal train_time_s
            train_time_s = float(time.perf_counter() - start)

    await asyncio.to_thread(fit_sync)

    allowed_labels = _allowed_labels_for_df(test_df)
    logger.info("Allowed labels: {}", len(allowed_labels))

    if isinstance(cfg.model, (GoogleOpenAIChatSpec, CommitteeLLMSpec)):
        logger.info("Evaluating LLM predictor (async) on {} rows", len(test_df))
        res = await aevaluate_predictor_on_df(
            predictor,
            df=test_df,
            allowed_labels=allowed_labels,
            dataset_name=str(test_df[SCHEMA.dataset_name].iloc[0]),
            split_name="experiment",
            tier_size=int(len(test_df)),
            output_dir=out_dir,
        )
    else:

        def eval_sync():
            logger.info("Evaluating sync predictor on {} rows", len(test_df))
            return evaluate_predictor_on_df(
                predictor,
                df=test_df,
                allowed_labels=allowed_labels,
                dataset_name=str(test_df[SCHEMA.dataset_name].iloc[0]),
                split_name="experiment",
                tier_size=int(len(test_df)),
                output_dir=out_dir,
            )

        res = await asyncio.to_thread(eval_sync)

    logger.info("Experiment done: {} → {}", cfg.name, str(out_dir))
    # Attach train time to the report and persist it (harness writes report.json earlier).
    if train_time_s is not None:
        try:
            extras = res.report.get("extras")
            if isinstance(extras, dict):
                extras["train_time_s"] = float(train_time_s)
            else:
                res.report["extras"] = {"train_time_s": float(train_time_s)}
            (out_dir / "report.json").write_text(json.dumps(res.report, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to attach train_time_s to report.json: {}", repr(exc))

    return {"output_dir": str(out_dir), "report": res.report}


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    """
    Top-level sync entrypoint: starts an event loop once per experiment (no nested asyncio.run).
    """

    return asyncio.run(arun_experiment(config_path))
