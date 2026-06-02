# pyright: basic
from __future__ import annotations

from pathlib import Path

import pytest

from src.dspy_gepa.program_load import resolve_compiled_program_dir, resolve_eval_parquet


def test_resolve_program_dir_from_pkl_file(tmp_path: Path) -> None:
    prog_dir = tmp_path / "compiled_program"
    prog_dir.mkdir()
    pkl = prog_dir / "program.pkl"
    pkl.write_bytes(b"x")
    (prog_dir / "metadata.json").write_text("{}", encoding="utf-8")
    assert resolve_compiled_program_dir(pkl) == prog_dir.resolve()
    assert resolve_compiled_program_dir(prog_dir) == prog_dir.resolve()


def test_resolve_program_dir_missing_pkl(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="program.pkl"):
        resolve_compiled_program_dir(empty)


def test_resolve_eval_parquet_tier10_fallback() -> None:
    root = Path(__file__).resolve().parents[2]
    p = resolve_eval_parquet(
        repo_root=root,
        dataset_name="banking-10",
        eval_parquet=None,
        eval_tier=10,
    )
    assert p.name == "samples.parquet"
    assert "tier_20" in str(p)
