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
from src.models.llm.google_genai_batch import GoogleGenaiBatchParams, GoogleGenaiBatchPredictor
from src.models.llm.openai_compat_chat_batch import OpenAICompatChatBatchParams, OpenAICompatChatBatchPredictor
from src.models.llm.hf_inference_textgen_batch import (
    HFInferenceTextGenBatchParams,
    HFInferenceTextGenBatchPredictor,
)
from src.models.llm.multilabel_confusion_probe import (
    MultilabelConfusionProbeParams as MLProbeRuntimeParams,
    MultilabelConfusionProbePredictor,
)
from src.models.llm.self_debate_batch import SelfDebateBatchParams as SDRRuntimeParams, SelfDebateBatchPredictor
from src.data_selection.label_utils import canonicalizer_for_dataset

from .config import (
    CommitteeLLMSpec,
    ExperimentConfig,
    GoogleGenaiChatSpec,
    GoogleOpenAIChatSpec,
    MultilabelConfusionProbeSpec,
    SelfDebateSpec,
    SklearnLogRegSpec,
    SklearnSvmSpec,
    TfidfXgbSpec,
    EmbUmapHeadSpec,
    SetFitSpec,
    load_experiment_config,
)
from .logging import write_full_metadata, write_predictions_json, write_run_manifest


def _load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate_processed_samples_df(df)
    return df


def _shuffle_llm_df(df: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """
    Break correlation between on-disk row order (often label-clustered) and API batch order.

    Uses cfg.seed so shuffle is reproducible. Classification metrics are unchanged; only the
    sequence of LLM requests changes (reduces spurious ``repeat last label'' batch effects).
    """

    return df.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)


def _allowed_labels_for_df(df: pd.DataFrame, *, processed_root: Path | None = None) -> list[str]:
    dataset_name = str(df[SCHEMA.dataset_name].iloc[0])
    pr = processed_root_fn(processed_root)
    cp = default_card_path(processed_root_dir=pr, dataset_name=dataset_name)
    if cp.exists():
        return list(read_card_json(cp).labels)
    return sorted(df[SCHEMA.true_label].astype(str).unique().tolist())


def _few_shot_from_train_df(train_df: pd.DataFrame, *, n: int | None = None) -> list[tuple[str, str]]:
    """Use all train_seed rows by default (tier_10 is already stratified at build time)."""
    df = train_df.reset_index(drop=True)
    take = len(df) if n is None else min(int(n), len(df))
    out: list[tuple[str, str]] = []
    for i in range(take):
        out.append((str(df[SCHEMA.text].iloc[i]), str(df[SCHEMA.true_label].iloc[i])))
    return out


def _is_llm_spec(cfg: ExperimentConfig) -> bool:
    return isinstance(
        cfg.model,
        (
            GoogleOpenAIChatSpec,
            CommitteeLLMSpec,
            GoogleGenaiChatSpec,
            MultilabelConfusionProbeSpec,
            SelfDebateSpec,
        ),
    )


def build_predictor(cfg: ExperimentConfig, *, train_df: pd.DataFrame | None = None) -> Any:
    m = cfg.model
    if isinstance(m, SklearnSvmSpec):
        return SklearnTfidfSvmPredictor()
    if isinstance(m, SklearnLogRegSpec):
        return SklearnTfidfLogRegPredictor()
    if isinstance(m, MultilabelConfusionProbeSpec):
        few_shot = _few_shot_from_train_df(train_df) if train_df is not None else None
        ds = str(train_df[SCHEMA.dataset_name].iloc[0]) if train_df is not None and len(train_df) else ""
        canon = canonicalizer_for_dataset(ds) if ds else None
        p = m.params
        return MultilabelConfusionProbePredictor(
            params=MLProbeRuntimeParams(
                model_id=p.model_id,
                batch_size=p.batch_size,
                max_concurrency=p.max_concurrency,
                temperature=p.temperature,
                max_tokens=p.max_tokens,
                retries=p.retries,
                thinking_level=p.thinking_level,
                include_thoughts=p.include_thoughts,
            ),
            few_shot=few_shot,
            label_normalizer=canon,
        )
    if isinstance(m, SelfDebateSpec):
        few_shot = _few_shot_from_train_df(train_df) if train_df is not None else None
        ds = str(train_df[SCHEMA.dataset_name].iloc[0]) if train_df is not None and len(train_df) else ""
        canon = canonicalizer_for_dataset(ds) if ds else None
        p = m.params
        return SelfDebateBatchPredictor(
            params=SDRRuntimeParams(
                model_id=p.model_id,
                batch_size=p.batch_size,
                max_concurrency=p.max_concurrency,
                temperature_a=p.temperature_a,
                pass_b_temperature=p.pass_b_temperature,
                max_tokens=p.max_tokens,
                retries=p.retries,
                thinking_level=p.thinking_level,
                include_thoughts=p.include_thoughts,
            ),
            few_shot=few_shot,
            label_normalizer=canon,
        )
    if isinstance(m, GoogleGenaiChatSpec):
        few_shot = _few_shot_from_train_df(train_df) if train_df is not None else None
        p = m.params
        return GoogleGenaiBatchPredictor(
            params=GoogleGenaiBatchParams(
                model_id=p.model_id,
                prompt_id=p.prompt_id,
                few_shot=few_shot,
                batch_size=p.batch_size,
                max_concurrency=p.max_concurrency,
                temperature=p.temperature,
                max_tokens=p.max_tokens,
                retries=p.retries,
                thinking_level=p.thinking_level,
                include_thoughts=p.include_thoughts,
                sequential_batches=p.sequential_batches,
            ),
        )
    if isinstance(m, GoogleOpenAIChatSpec):
        few_shot = _few_shot_from_train_df(train_df) if train_df is not None else None
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
        few_shot = _few_shot_from_train_df(train_df) if train_df is not None else None
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


async def arun_experiment(config_path: str | Path, *, experiment_slug: str | None = None) -> dict[str, Any]:
    t_start = time.perf_counter()
    cfg = load_experiment_config(config_path)
    logger.info("Experiment start: {} ({})", cfg.name, str(config_path))
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.resolved.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

    slug = experiment_slug or cfg.name.split("_")[0]
    if isinstance(cfg.model, (MultilabelConfusionProbeSpec, SelfDebateSpec)):
        slug = "multilabel_confusion_probe" if isinstance(cfg.model, MultilabelConfusionProbeSpec) else "self_debate"
    write_run_manifest(
        out_dir,
        experiment_slug=slug,
        config_path=str(config_path),
        cfg_payload=cfg.model_dump(mode="json"),
        extra={"llm_shuffle_seed": int(cfg.seed), "llm_shuffle_seed_test": int(cfg.seed) + 1},
    )

    train_df = _load_parquet(cfg.train_data)
    test_df = _load_parquet(cfg.test_data)
    logger.info(
        "Loaded dataframes: train_rows={}, test_rows={}, dataset={}",
        len(train_df),
        len(test_df),
        str(test_df[SCHEMA.dataset_name].iloc[0]),
    )

    if _is_llm_spec(cfg):
        train_df = _shuffle_llm_df(train_df, seed=cfg.seed)
        test_df = _shuffle_llm_df(test_df, seed=cfg.seed + 1)
        logger.info("LLM run: shuffled train/test row order (seeded) before predict")

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

    if _is_llm_spec(cfg):
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

    duration_s = time.perf_counter() - t_start
    write_full_metadata(out_dir, report=res.report, duration_seconds=duration_s)
    try:
        pred_records = res.predictions_df.to_dict(orient="records")
        write_predictions_json(out_dir, pred_records)
    except Exception as exc:
        logger.warning("Could not write full_predictions.json: {}", repr(exc))

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
