# pyright: basic
"""Load MIPRO-compiled programs saved with ``save(..., save_program=True)``."""

from __future__ import annotations

from pathlib import Path

import dspy
from dspy.primitives.module import Module
from loguru import logger

from src.experiments.suites.extended_suite import DATASETS


def resolve_compiled_program_dir(path: Path) -> Path:
    """
    Normalize a user path to the directory that contains ``program.pkl`` + ``metadata.json``.

    Accepts:
    - ``.../compiled_program`` (directory)
    - ``.../compiled_program/program.pkl`` (file)
    """
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        pkl = resolved / "program.pkl"
        if pkl.is_file():
            return resolved
        raise FileNotFoundError(
            f"Compiled program directory missing program.pkl: {resolved}. "
            "Pass the folder from run_gepa_mipro_optimize (e.g. .../compiled_program), not a .pkl alone "
            "unless it lives inside that folder."
        )
    if resolved.is_file() and resolved.name == "program.pkl":
        return resolved.parent
    raise FileNotFoundError(
        f"Not a compiled program path: {path}. Expected a directory with program.pkl and metadata.json."
    )


def load_compiled_program(path: Path, *, allow_pickle: bool = True) -> Module:
    """Load full module written by ``optimized.save(dir, save_program=True)``."""
    program_dir = resolve_compiled_program_dir(path)
    return dspy.load(str(program_dir), allow_pickle=allow_pickle)


def resolve_eval_parquet(
    *,
    repo_root: Path,
    dataset_name: str,
    eval_parquet: Path | None,
    eval_tier: int,
) -> Path:
    """Default test split parquet; tier 10 falls back to tier_20 when absent."""
    if eval_parquet is not None:
        p = eval_parquet.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Eval parquet not found: {p}")
        return p

    folder = _dataset_folder(dataset_name)
    base = repo_root / "data" / "processed" / folder / "test"
    preferred = base / f"tier_{int(eval_tier)}" / "samples.parquet"
    if preferred.is_file():
        return preferred

    if int(eval_tier) == 10:
        fallback = base / "tier_20" / "samples.parquet"
        if fallback.is_file():
            logger.warning(
                "test/tier_10 not found for {}; using {} (apply --eval-max-rows 10 for a 10-row eval)",
                dataset_name,
                fallback,
            )
            return fallback

    raise FileNotFoundError(
        f"No eval parquet for {dataset_name!r} at {preferred}. "
        f"Available under {base}: "
        f"{sorted(d.name for d in base.iterdir() if d.is_dir()) if base.is_dir() else 'none'}"
    )


def _dataset_folder(dataset_name: str) -> str:
    for ds in DATASETS:
        if ds.dataset_name == dataset_name:
            return ds.folder_name
    raise ValueError(f"Unknown dataset: {dataset_name!r}")
