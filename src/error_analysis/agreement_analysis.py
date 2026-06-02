# pyright: basic
"""Pairwise agreement (Cohen's kappa), McNemar, and per-class PR across runs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import cohen_kappa_score

from src.error_analysis.discover import DiscoveredRun, discover_all_runs
from src.error_analysis.labels import canonical_pred_value, dataset_name_from_frame
from src.error_analysis.leaderboard import build_leaderboard_df
from src.error_analysis.legacy_experiments import (
    discover_legacy_experiments,
    legacy_classification_reports,
    legacy_predictions_index,
)
from src.error_analysis.predictions_source import load_predictions_for_run
from src.error_analysis.run_record import extract_run_record


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:120]


def _mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    try:
        from scipy.stats import binomtest

        return float(binomtest(k, n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        pass
    if n > 64:
        return float("nan")
    prob = 0.0
    for i in range(0, k + 1):
        prob += math.comb(n, i) * (0.5**n)
    return float(min(1.0, 2.0 * prob))


def mcnemar_table(
    gold: pd.Series,
    pred_a: pd.Series,
    pred_b: pd.Series,
    *,
    dataset_name: str | None,
) -> tuple[dict[str, int], float]:
    def is_correct(pred: pd.Series) -> pd.Series:
        if dataset_name:

            def row_ok(idx: int) -> bool:
                g = gold.iloc[idx]
                p = pred.iloc[idx]
                cg = canonical_pred_value(g, dataset_name=dataset_name)
                cp = canonical_pred_value(p, dataset_name=dataset_name)
                return cg is not None and cp is not None and cg == cp

            return pd.Series([row_ok(i) for i in range(len(gold))], index=gold.index)

        return pred.astype(str) == gold.astype(str)

    a_ok = is_correct(pred_a)
    b_ok = is_correct(pred_b)
    both_ok = int((a_ok & b_ok).sum())
    a_only = int((a_ok & ~b_ok).sum())
    b_only = int((~a_ok & b_ok).sum())
    both_wrong = int((~a_ok & ~b_ok).sum())
    return (
        {"both_correct": both_ok, "a_only": a_only, "b_only": b_only, "both_wrong": both_wrong},
        _mcnemar_exact_p(a_only, b_only),
    )


def cohen_kappa_matrix(preds_wide: pd.DataFrame) -> pd.DataFrame:
    names = list(preds_wide.columns)
    n = len(names)
    mat = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
            elif j < i:
                mat[i, j] = mat[j, i]
            else:
                mat[i, j] = float(cohen_kappa_score(preds_wide.iloc[:, i].tolist(), preds_wide.iloc[:, j].tolist()))
    return pd.DataFrame(mat, index=names, columns=names)


def _heatmap(mat: pd.DataFrame, *, out_path: Path, title: str) -> None:
    if mat.empty:
        return
    n = mat.shape[0]
    fig_w = max(10.0, 0.55 * n + 4.0)
    fig_h = max(8.0, 0.55 * n + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.to_numpy(), cmap="viridis", vmin=-0.2, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cohen's kappa", rotation=90)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_title(title)
    data = mat.to_numpy()
    for i in range(n):
        for j in range(n):
            v = data[i, j]
            txt = "NA" if np.isnan(v) else f"{v:.2f}"
            color = "white" if (not np.isnan(v) and v < 0.4) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


@dataclass(frozen=True, slots=True)
class AgreementGroup:
    group_key: str
    dataset_name: str
    tier_size: int | None
    run_keys: tuple[str, ...]


def _group_key(dataset_name: str, tier_size: object) -> str:
    ts = tier_size if tier_size is not None else "any"
    return f"{dataset_name}__tier{ts}"


def build_agreement_groups(
    leaderboard_df: pd.DataFrame,
    *,
    min_runs: int = 2,
    max_runs_per_group: int = 25,
) -> list[AgreementGroup]:
    if leaderboard_df.empty:
        return []
    sub = leaderboard_df[leaderboard_df["has_predictions"] == True].copy()  # noqa: E712
    if sub.empty:
        return []
    groups: list[AgreementGroup] = []
    for (ds, tier), g in sub.groupby(["dataset_name", "tier_size"], dropna=False):
        if not isinstance(ds, str) or not ds:
            continue
        g2 = g.sort_values("f1_macro", ascending=False, na_position="last").head(max_runs_per_group)
        keys = tuple(g2["run_key"].astype(str).tolist())
        if len(keys) < min_runs:
            continue
        groups.append(
            AgreementGroup(
                group_key=_group_key(ds, tier),
                dataset_name=ds,
                tier_size=int(tier) if tier is not None and not pd.isna(tier) else None,
                run_keys=keys,
            )
        )
    return groups


def _model_filter_for_run_key(run_key: str) -> str | None:
    if run_key.startswith("hf_llms_comparison/"):
        return run_key.rsplit("/", 1)[-1]
    return None


def _pred_series_from_df(df: pd.DataFrame, *, dataset_name: str) -> pd.Series | None:
    if "sample_id" not in df.columns or "pred_label" not in df.columns:
        return None
    s = df.set_index("sample_id")["pred_label"]
    if dataset_name:

        def canon(v: object) -> str | None:
            c = canonical_pred_value(v, dataset_name=dataset_name)
            return c if c is not None else None

        return s.apply(canon).dropna()
    return s.astype(str)


def load_preds_wide_for_group(
    group: AgreementGroup,
    *,
    key_to_dir: dict[str, Path],
    key_to_pred_path: dict[str, Path],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    series_by_run: dict[str, pd.Series] = {}
    gold: pd.Series | None = None
    kept: list[str] = []

    for run_key in group.run_keys:
        run_dir = key_to_dir.get(run_key)
        pred_path = key_to_pred_path.get(run_key)
        if run_dir is None and pred_path is None:
            continue
        df = load_predictions_for_run(
            run_dir,
            predictions_path=pred_path,
            dataset_name=group.dataset_name,
            model_filter=_model_filter_for_run_key(run_key),
        )
        if df is None or "sample_id" not in df.columns:
            continue
        if "true_label" in df.columns and gold is None:
            gold = df.set_index("sample_id")["true_label"]
        pred = _pred_series_from_df(df, dataset_name=group.dataset_name)
        if pred is None or pred.empty:
            continue
        short = run_key.rsplit("/", 1)[-1][:48]
        series_by_run[short] = pred
        kept.append(short)

    if not series_by_run or gold is None:
        return pd.DataFrame(), pd.Series(dtype=str), []

    preds_wide = pd.DataFrame(series_by_run).dropna(axis=0, how="any")
    gold = gold.loc[preds_wide.index].astype(str)
    return preds_wide, gold, kept


def _load_classification_report_dict(run_dir: Path) -> dict[str, Any] | None:
    import json

    for path in (
        run_dir / "full_classification_report.json",
        run_dir / "metrics.json",
        run_dir / "val_eval_macro_f1.json",
    ):
        if not path.is_file():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        cr = raw.get("classification_report") if path.name in {"metrics.json", "val_eval_macro_f1.json"} else raw
        if isinstance(cr, dict):
            return cr
    return None


def per_class_pr_from_reports(
    run_keys: tuple[str, ...],
    key_to_dir: dict[str, Path],
    *,
    legacy_cr: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    legacy_cr = legacy_cr or {}
    for run_key in run_keys:
        label = run_key.rsplit("/", 1)[-1][:48]
        cr = legacy_cr.get(run_key)
        if cr is None:
            run_dir = key_to_dir.get(run_key)
            if run_dir is not None:
                cr = _load_classification_report_dict(run_dir)
        if not isinstance(cr, dict):
            continue
        for class_name, stats in cr.items():
            if class_name in {"accuracy", "macro avg", "weighted avg", "micro avg"}:
                continue
            if not isinstance(stats, dict):
                continue
            for metric in ("precision", "recall"):
                v = stats.get(metric)
                if v is None:
                    continue
                rows.append({"run": label, "label": str(class_name), "metric": metric, "value": float(v)})
    return pd.DataFrame(rows)


def _boxplot_per_class(pr_df: pd.DataFrame, *, out_path: Path, title: str) -> None:
    if pr_df.empty:
        return
    labels = sorted(pr_df["label"].unique().tolist())
    fig, axes = plt.subplots(2, 1, figsize=(max(12, 0.8 * len(labels) + 6), 10), sharex=True)
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes, ["precision", "recall"], strict=True):
        sub = pr_df[pr_df["metric"] == metric]
        data = [sub[sub["label"] == lab]["value"].astype(float).tolist() for lab in labels]
        ax.boxplot(data, showfliers=False)
        for i, vals in enumerate(data, start=1):
            if vals:
                ax.scatter(rng.normal(loc=i, scale=0.06, size=len(vals)), vals, s=12, alpha=0.55)
        ax.set_ylabel(metric)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(range(1, len(labels) + 1))
    axes[-1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.suptitle(title, y=0.99)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_agreement_bundle(
    group: AgreementGroup,
    *,
    key_to_dir: dict[str, Path],
    key_to_pred_path: dict[str, Path],
    legacy_cr: dict[str, dict[str, Any]],
    out_dir: Path,
    mcnemar_top_k: int = 15,
) -> None:
    preds_wide, gold, kept = load_preds_wide_for_group(
        group, key_to_dir=key_to_dir, key_to_pred_path=key_to_pred_path
    )
    if preds_wide.empty or len(kept) < 2:
        logger.debug("Skipping agreement group {} (<2 comparable runs)", group.group_key)
        return

    suffix = _safe_filename(group.group_key)
    gdir = out_dir / suffix
    gdir.mkdir(parents=True, exist_ok=True)

    kappa = cohen_kappa_matrix(preds_wide)
    _heatmap(
        kappa,
        out_path=gdir / "cohen_kappa_heatmap.png",
        title=f"Cohen's kappa — {group.dataset_name} (tier={group.tier_size})",
    )
    kappa.to_csv(gdir / "cohen_kappa_matrix.csv")

    names = list(preds_wide.columns)
    pair_rows: list[dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            tbl, p = mcnemar_table(
                gold,
                preds_wide[names[i]],
                preds_wide[names[j]],
                dataset_name=group.dataset_name,
            )
            pair_rows.append(
                {
                    "model_a": names[i],
                    "model_b": names[j],
                    **tbl,
                    "disagreements": tbl["a_only"] + tbl["b_only"],
                    "p_value_exact": p,
                }
            )
    pairs_df = pd.DataFrame(pair_rows)
    if not pairs_df.empty and "disagreements" in pairs_df.columns:
        pairs_df = pairs_df.sort_values("disagreements", ascending=False)
    pairs_df.to_csv(gdir / "mcnemar_pairs.csv", index=False)

    pr_df = per_class_pr_from_reports(group.run_keys, key_to_dir, legacy_cr=legacy_cr)
    if not pr_df.empty:
        pr_df.to_csv(gdir / "per_class_pr_long.csv", index=False)
        _boxplot_per_class(
            pr_df,
            out_path=gdir / "per_class_pr_boxplot.png",
            title=f"Per-class PR — {group.dataset_name}",
        )

    summary_path = gdir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"group_key={group.group_key}",
                f"dataset_name={group.dataset_name}",
                f"tier_size={group.tier_size}",
                f"runs_with_predictions={len(kept)}",
                f"items_intersection={preds_wide.shape[0]}",
                "",
                "Top McNemar pairs (exact two-sided p):",
                pairs_df.head(mcnemar_top_k).to_string(index=False) if not pairs_df.empty else "(no pairs)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("Wrote agreement bundle {}", gdir)


def run_agreement_analysis(
    *,
    results_root: Path,
    out_dir: Path,
    max_groups: int | None = None,
    max_runs_per_group: int = 25,
) -> pd.DataFrame:
    root = results_root.resolve()
    lb = build_leaderboard_df(root)
    groups = build_agreement_groups(lb, max_runs_per_group=max_runs_per_group)
    if max_groups is not None:
        groups = groups[:max_groups]

    key_to_dir: dict[str, Path] = {}
    for d in discover_all_runs(root):
        key_to_dir[d.rel_dir] = d.run_dir

    legacy_records = discover_legacy_experiments(root)
    key_to_pred_path = legacy_predictions_index(legacy_records)
    legacy_cr = legacy_classification_reports(legacy_records)

    index_rows: list[dict[str, Any]] = []
    for g in groups:
        write_agreement_bundle(
            g,
            key_to_dir=key_to_dir,
            key_to_pred_path=key_to_pred_path,
            legacy_cr=legacy_cr,
            out_dir=out_dir,
        )
        index_rows.append(
            {
                "group_key": g.group_key,
                "dataset_name": g.dataset_name,
                "tier_size": g.tier_size,
                "n_runs": len(g.run_keys),
            }
        )

    index_df = pd.DataFrame(index_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(out_dir / "agreement_groups_index.csv", index=False)
    return index_df
