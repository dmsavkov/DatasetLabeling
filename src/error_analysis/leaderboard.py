# pyright: basic
"""Build a CSV leaderboard across every discovered run under ``results/``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.error_analysis.discover import discover_all_runs, discover_reasoning_spectrum_rows
from src.error_analysis.legacy_experiments import discover_legacy_experiments, legacy_rows_for_leaderboard
from src.error_analysis.leaderboard_plots import write_leaderboard_plots
from src.error_analysis.run_record import LEADERBOARD_COLUMNS, RunRecord, extract_run_record


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _merge_leaderboard_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Deduplicate by run_key; prefer harness rows over legacy when both exist."""
    by_key: dict[str, dict[str, object]] = {}
    harness_formats = {"harness_report", "metrics_json", "metadata_only"}
    for row in rows:
        key = str(row.get("run_key") or "")
        if not key:
            continue
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = row
            continue
        prev_fmt = str(prev.get("format") or "")
        new_fmt = str(row.get("format") or "")
        if prev_fmt in harness_formats and new_fmt not in harness_formats:
            continue
        if new_fmt in harness_formats and prev_fmt not in harness_formats:
            by_key[key] = row
            continue
        # Keep higher F1 when both legacy/duplicate
        prev_f1 = prev.get("f1_macro")
        new_f1 = row.get("f1_macro")
        if new_f1 is not None and (prev_f1 is None or float(new_f1) > float(prev_f1)):
            by_key[key] = row
    return list(by_key.values())


def build_leaderboard_df(results_root: Path | None = None) -> pd.DataFrame:
    root = (results_root or (_repo_root() / "results")).resolve()
    rows: list[dict[str, object]] = []

    for discovered in discover_all_runs(root):
        rec = extract_run_record(discovered)
        if rec is None:
            continue
        rows.append(rec.to_leaderboard_dict())

    rows.extend(legacy_rows_for_leaderboard(discover_legacy_experiments(root)))
    rows.extend(discover_reasoning_spectrum_rows(root))
    rows = _merge_leaderboard_rows(rows)

    if not rows:
        return pd.DataFrame(columns=list(LEADERBOARD_COLUMNS))

    df = pd.DataFrame(rows)
    present = [c for c in LEADERBOARD_COLUMNS if c in df.columns]
    df = df[present]
    sort_cols = [c for c in ("f1_macro", "accuracy", "saved_utc", "run_key") if c in df.columns]
    return df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), na_position="last")


def write_leaderboard(
    *,
    results_root: Path | None = None,
    csv_path: Path,
    plots_dir: Path | None = None,
    write_plots: bool = True,
) -> pd.DataFrame:
    df = build_leaderboard_df(results_root)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Wrote leaderboard {} ({} rows)", csv_path, len(df))
    if write_plots and plots_dir is not None and not df.empty:
        write_leaderboard_plots(df, plots_dir)
    return df
