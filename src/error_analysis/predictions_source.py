# pyright: basic
"""Load per-sample predictions without reading oversized JSON blobs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

# Skip huge prediction exports unless caller raises the limit.
DEFAULT_MAX_PREDICTIONS_BYTES = 40 * 1024 * 1024


def _normalize_predictions_df(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    if "id" in df.columns and "sample_id" not in df.columns:
        rename["id"] = "sample_id"
    if "gold" in df.columns and "true_label" not in df.columns:
        rename["gold"] = "true_label"
    if "pred" in df.columns and "pred_label" not in df.columns:
        rename["pred"] = "pred_label"
    if rename:
        df = df.rename(columns=rename)
    if "sample_id" in df.columns:
        df = df.copy()
        df["sample_id"] = df["sample_id"].astype(str)
    return df


def _normalize_legacy_predictions_df(
    df: pd.DataFrame,
    *,
    dataset_name: str | None,
    model_filter: str | None,
) -> pd.DataFrame:
    out = df.copy()
    rename: dict[str, str] = {}
    if "true" in out.columns and "true_label" not in out.columns:
        rename["true"] = "true_label"
    if "pred" in out.columns and "pred_label" not in out.columns:
        rename["pred"] = "pred_label"
    if "source_index" in out.columns and "sample_id" not in out.columns:
        rename["source_index"] = "sample_id"
    if rename:
        out = out.rename(columns=rename)
    if "sample_id" not in out.columns and len(out):
        out["sample_id"] = out.index.astype(str)
    if dataset_name and "dataset_name" not in out.columns:
        out["dataset_name"] = dataset_name
    if model_filter and "model" in out.columns:
        out = out[out["model"].astype(str) == model_filter]
    return out


def predictions_available(run_dir: Path) -> bool:
    if (run_dir / "predictions.csv").is_file():
        return True
    jp = run_dir / "full_predictions.json"
    if not jp.is_file() or (run_dir / "full_predictions_skipped.txt").is_file():
        return False
    return jp.stat().st_size <= DEFAULT_MAX_PREDICTIONS_BYTES


def load_predictions_df(
    run_dir: Path,
    *,
    max_bytes: int = DEFAULT_MAX_PREDICTIONS_BYTES,
) -> pd.DataFrame | None:
    csv_path = run_dir / "predictions.csv"
    if csv_path.is_file():
        try:
            return _normalize_predictions_df(pd.read_csv(csv_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read {}: {}", csv_path, exc)
            return None

    json_path = run_dir / "full_predictions.json"
    if not json_path.is_file():
        return None
    if (run_dir / "full_predictions_skipped.txt").is_file():
        logger.debug("Predictions skipped for {}", run_dir)
        return None
    size = json_path.stat().st_size
    if size > max_bytes:
        logger.warning(
            "Skipping {} ({} bytes > max {}); use predictions.csv or raise max_bytes",
            json_path,
            size,
            max_bytes,
        )
        return None

    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse {}: {}", json_path, exc)
        return None

    if isinstance(raw, list):
        if not raw:
            return None
        return _normalize_predictions_df(pd.DataFrame(raw))
    if isinstance(raw, dict) and isinstance(raw.get("predictions"), list):
        return _normalize_predictions_df(pd.DataFrame(raw["predictions"]))
    logger.warning("Unexpected full_predictions.json shape at {}", json_path)
    return None


def load_predictions_from_path(
    path: Path,
    *,
    max_bytes: int = DEFAULT_MAX_PREDICTIONS_BYTES,
    dataset_name: str | None = None,
    model_filter: str | None = None,
) -> pd.DataFrame | None:
    """Load predictions from an explicit CSV path (legacy experiment layouts)."""
    if not path.is_file():
        return None
    if path.stat().st_size > max_bytes:
        logger.warning("Skipping large predictions file {}", path)
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read {}: {}", path, exc)
        return None
    df = _normalize_legacy_predictions_df(df, dataset_name=dataset_name, model_filter=model_filter)
    return _normalize_predictions_df(df)


def load_predictions_for_run(
    run_dir: Path | None,
    *,
    predictions_path: Path | None = None,
    max_bytes: int = DEFAULT_MAX_PREDICTIONS_BYTES,
    dataset_name: str | None = None,
    model_filter: str | None = None,
) -> pd.DataFrame | None:
    if predictions_path is not None:
        loaded = load_predictions_from_path(
            predictions_path,
            max_bytes=max_bytes,
            dataset_name=dataset_name,
            model_filter=model_filter,
        )
        if loaded is not None:
            return loaded
    if run_dir is not None:
        return load_predictions_df(run_dir, max_bytes=max_bytes)
    return None
