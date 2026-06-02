# pyright: basic
"""Summary plots for the experiment leaderboard."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger


def _label_row(row: pd.Series) -> str:
    parts = [str(row.get("series") or ""), str(row.get("run_leaf") or "")]
    if row.get("dataset_name"):
        parts.append(str(row["dataset_name"]))
    return " / ".join(p for p in parts if p)[:80]


def _plot_top(
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    out_path: Path,
    ascending: bool,
    n: int = 15,
) -> None:
    if metric not in df.columns or df.empty:
        return
    sub = df.dropna(subset=[metric]).sort_values(metric, ascending=ascending, na_position="last").head(n)
    if sub.empty:
        return

    fig_h = max(4.0, 0.4 * len(sub) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    vals = sub[metric].astype(float).tolist()
    labels = [_label_row(row) for _, row in sub.iterrows()]
    color = "#2ca02c" if not ascending else "#d62728"
    ax.barh(range(len(sub)), vals, color=color, alpha=0.85)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.set_xlim(0, min(1.05, max(vals) + 0.05) if vals else 1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote {}", out_path)


def write_leaderboard_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    scored = df[df["has_predictions"] == True] if "has_predictions" in df.columns else df  # noqa: E712
    if scored.empty:
        scored = df
    if "f1_macro" not in scored.columns:
        return
    _plot_top(
        scored,
        metric="f1_macro",
        title="Top runs by macro F1 (with predictions when available)",
        out_path=plots_dir / "top15_f1_macro.png",
        ascending=False,
    )
    _plot_top(
        scored,
        metric="f1_macro",
        title="Lowest macro F1 (sample)",
        out_path=plots_dir / "bottom15_f1_macro.png",
        ascending=True,
    )
