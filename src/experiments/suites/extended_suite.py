# pyright: basic
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from src.experiments.config import ExperimentConfig


TrainKind = Literal["seed10", "seed100", "tier5000"]
HeadKind = Literal["xgb", "logreg", "knn"]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    dataset_name: str
    folder_name: str


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(dataset_name="banking-10", folder_name="banking-10"),
    DatasetSpec(dataset_name="tweet_eval_irony", folder_name="tweet_eval_irony"),
    DatasetSpec(dataset_name="implicit_hate", folder_name="implicit_hate"),
    DatasetSpec(dataset_name="pubmed_20k_rct", folder_name="pubmed_20k_rct"),
)


def _train_path(ds: DatasetSpec, kind: TrainKind) -> str:
    base = f"data/processed/{ds.folder_name}"
    if kind == "seed10":
        return f"{base}/train_seed/tier_10/samples.parquet"
    if kind == "seed100":
        return f"{base}/train_seed/tier_100/samples.parquet"
    return f"{base}/train_seed/tier_5000/samples.parquet"


def _test_path(ds: DatasetSpec, *, tier: int = 200) -> str:
    return f"data/processed/{ds.folder_name}/test/tier_{int(tier)}/samples.parquet"


def _name_for(*parts: str) -> str:
    return "_".join([p for p in parts if p])


def build_extended_suite_configs() -> list[ExperimentConfig]:
    out: list[ExperimentConfig] = []

    for ds in DATASETS:
        test200 = _test_path(ds, tier=200)

        # LLM single-model (Gemma) uses few-shot from train_seed_10.
        out.append(
            ExperimentConfig.model_validate(
                {
                    "name": _name_for("gemma", ds.dataset_name, "test200"),
                    "seed": 42,
                    "train_data": _train_path(ds, "seed10"),
                    "test_data": test200,
                    "output_dir": "results/extended/placeholder",
                    "model": {
                        "kind": "google_openai_chat",
                        "params": {
                            "model_id": "gemma-3-4b-it",
                            "prompt_id": "baseline_v1",
                            "batch_size": 5,
                            "max_concurrency": 5,
                            "temperature": 0.0,
                            "max_tokens": None,
                            "retries": 20,
                        },
                    },
                }
            )
        )

        # Committee.
        out.append(
            ExperimentConfig.model_validate(
                {
                    "name": _name_for("committee", ds.dataset_name, "test200"),
                    "seed": 42,
                    "train_data": _train_path(ds, "seed10"),
                    "test_data": test200,
                    "output_dir": "results/extended/placeholder",
                    "model": {
                        "kind": "committee_llm",
                        "params": {
                            "member_model_ids": [
                                "Qwen/Qwen2.5-1.5B-Instruct",
                                "mistralai/Mixtral-8x22B-Instruct-v0.1",
                                "gemma-3-4b-it",
                            ],
                            "prompt_id": "baseline_v1",
                            "batch_size": 5,
                            "max_concurrency": 5,
                            "temperature": 0.0,
                            "max_tokens": None,
                            "retries": 20,
                        },
                    },
                }
            )
        )

        # Sync baselines on train_seed_100.
        out.append(
            ExperimentConfig.model_validate(
                {
                    "name": _name_for("svm", ds.dataset_name, "test200"),
                    "seed": 42,
                    "train_data": _train_path(ds, "seed100"),
                    "test_data": test200,
                    "output_dir": "results/extended/placeholder",
                    "model": {"kind": "sklearn_svm", "params": {}},
                }
            )
        )
        out.append(
            ExperimentConfig.model_validate(
                {
                    "name": _name_for("tfidf_xgb", ds.dataset_name, "test200"),
                    "seed": 42,
                    "train_data": _train_path(ds, "seed100"),
                    "test_data": test200,
                    "output_dir": "results/extended/placeholder",
                    "model": {"kind": "tfidf_xgb", "params": {}},
                }
            )
        )

        # Embeddings + UMAP + head: train sizes 100 and tier5000; heads xgb/logreg/knn.
        for train_kind in ("seed100", "tier5000"):
            for head in ("xgb", "logreg", "knn"):
                head_kwargs: dict[str, Any]
                if head == "xgb":
                    head_kwargs = {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 3}
                elif head == "knn":
                    head_kwargs = {"n_neighbors": 5}
                else:
                    head_kwargs = {"max_iter": 2000}
                out.append(
                    ExperimentConfig.model_validate(
                        {
                            "name": _name_for("emb_umap_head", head, ds.dataset_name, f"train{train_kind}", "test200"),
                            "seed": 42,
                            "train_data": _train_path(ds, train_kind),  # type: ignore[arg-type]
                            "test_data": test200,
                            "output_dir": "results/extended/placeholder",
                            "model": {
                                "kind": "emb_umap_head",
                                "params": {
                                    "embedding_model_id": "sentence-transformers/all-MiniLM-L6-v2",
                                    "reducer_dim": 10,
                                    "head_kind": head,
                                    "head_kwargs": head_kwargs,
                                },
                            },
                        }
                    )
                )

        # SetFit: only small seeds (10/100). Larger tiers are intentionally excluded
        # from the extended suite to keep SetFit runs lightweight and comparable.
        for train_kind in ("seed10", "seed100"):
            out.append(
                ExperimentConfig.model_validate(
                    {
                        "name": _name_for("setfit", ds.dataset_name, f"train{train_kind}", "test200"),
                        "seed": 42,
                        "train_data": _train_path(ds, train_kind),  # type: ignore[arg-type]
                        "test_data": test200,
                        "output_dir": "results/extended/placeholder",
                        "model": {
                            "kind": "setfit",
                            "params": {
                                "embedding_model_id": "sentence-transformers/all-MiniLM-L6-v2",
                                "epochs": 1,
                                "max_steps": None,
                            },
                        },
                    }
                )
            )

    return out


def extended_suite_filenames() -> list[str]:
    """
    Deterministic file names under `experiments/` or a custom config directory.
    """

    cfgs = build_extended_suite_configs()
    return [f"{c.name}.yaml" for c in cfgs]


def llm_suite_filenames() -> list[str]:
    cfgs = build_extended_suite_configs()
    keep = []
    for c in cfgs:
        k = c.model.kind
        if k in ("google_openai_chat", "committee_llm"):
            keep.append(f"{c.name}.yaml")
    return keep


def ml_suite_filenames() -> list[str]:
    cfgs = build_extended_suite_configs()
    keep = []
    for c in cfgs:
        k = c.model.kind
        if k not in ("google_openai_chat", "committee_llm"):
            keep.append(f"{c.name}.yaml")
    return keep


def write_extended_suite_yamls(config_dir: Path) -> list[Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Cleanup: older suite versions generated SetFit tier5000 configs; remove them
    # on regeneration to avoid accidental runs.
    for p in config_dir.glob("setfit_*_traintier5000_*.yaml"):
        try:
            p.unlink()
        except OSError:
            pass

    for cfg in build_extended_suite_configs():
        # Give each YAML a stable default output_dir; runner overrides per-run.
        payload: dict[str, Any] = cfg.model_dump(mode="json")
        payload["output_dir"] = f"results/extended/{cfg.name}"

        p = config_dir / f"{cfg.name}.yaml"
        p.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written.append(p)
    return written

