# pyright: basic
from __future__ import annotations

from collections import Counter
from typing import Literal

import pandas as pd
import warnings

from src.error_analysis.io import LoadedExperiment


def _dataset_name_from_exp(e: LoadedExperiment) -> str | None:
    if e.report and isinstance(e.report.get("dataset_name"), str):
        return e.report["dataset_name"]
    if e.predictions_df is not None and "dataset_name" in e.predictions_df.columns:
        vals = e.predictions_df["dataset_name"].dropna().astype(str).unique().tolist()
        if len(vals) == 1:
            return vals[0]
    return None


def assert_same_dataset(exps: list[LoadedExperiment]) -> str | None:
    names = [n for n in (_dataset_name_from_exp(e) for e in exps) if n]
    if not names:
        return None
    c = Counter(names)
    if len(c) == 1:
        return next(iter(c))
    # ambiguous: let caller decide; return None
    warnings.warn(f"Multiple datasets detected in experiments: {dict(c)}. Skipping comparison by default.")
    return None


JoinKind = Literal["inner", "outer"]


def build_comparison_df(exps: list[LoadedExperiment], *, join: JoinKind = "inner") -> pd.DataFrame:
    """
    Build a wide comparison DF joined on `sample_id`.

    Requires: `predictions_df` present for each experiment, with `sample_id`.
    Keeps shared columns once: sample_id, text, true_label, meta_json, dataset_name (if present).
    Adds per-experiment columns: pred_label__{exp_id}, confidence__{exp_id}, reason__{exp_id}, in_tokens__{exp_id}, out_tokens__{exp_id}.
    """
    usable = [e for e in exps if e.predictions_df is not None]
    if not usable:
        return pd.DataFrame()

    for e in usable:
        if e.predictions_df is None or "sample_id" not in e.predictions_df.columns:
            raise ValueError(f"Experiment {e.exp_id} has no `sample_id` column; cannot compare.")

    # dataset mismatch detection (warn + skip compare)
    ds = assert_same_dataset(usable)
    if ds is None:
        # fallback: check if any experiment has >1 dataset value in its df
        dsn = [_dataset_name_from_exp(e) for e in usable]
        if len({x for x in dsn if x}) > 1:
            warnings.warn("Dataset mismatch across experiments; comparison skipped (returning empty df).")
            return pd.DataFrame()

    df0 = usable[0].predictions_df
    assert df0 is not None
    base_cols = [c for c in ["sample_id", "dataset_name", "text", "true_label", "meta_json"] if c in df0.columns]

    def per_exp_df(e: LoadedExperiment, *, include_shared: bool) -> pd.DataFrame:
        df = e.predictions_df
        assert df is not None
        keep_shared = [c for c in base_cols if c in df.columns] if include_shared else ["sample_id"]
        per_cols: dict[str, str] = {}
        for c in ["pred_label", "confidence", "reason", "in_tokens", "out_tokens"]:
            if c in df.columns:
                per_cols[c] = f"{c}__{e.exp_id}"
        out = df[keep_shared + list(per_cols.keys())].copy()
        if per_cols:
            out = out.rename(columns=per_cols)  # pyright: ignore[reportCallIssue]
        return pd.DataFrame(out)

    merged = per_exp_df(usable[0], include_shared=True)
    base_ids = set(merged["sample_id"].astype(str))

    how = "inner" if join == "inner" else "outer"
    for e in usable[1:]:
        cur = per_exp_df(e, include_shared=False)
        cur_ids = set(cur["sample_id"].astype(str))
        overlap = len(base_ids & cur_ids)
        if overlap == 0:
            # no overlap => probably wrong directory mix
            warnings.warn(f"No sample_id overlap between experiments while joining; skipping comparison.")
            return pd.DataFrame()
        if join == "inner" and overlap < min(len(base_ids), len(cur_ids)):
            warnings.warn(
                f"Partial sample_id overlap when joining {e.exp_id}: overlap={overlap} base={len(base_ids)} cur={len(cur_ids)} (inner join)."
            )
        merged = merged.merge(cur, on="sample_id", how=how)
        base_ids = set(merged["sample_id"].astype(str))

    return merged


def disagreements(df: pd.DataFrame) -> pd.DataFrame:
    pred_cols = [c for c in df.columns if c.startswith("pred_label__")]
    if len(pred_cols) < 2:
        return df.iloc[0:0].copy()
    m = df[pred_cols].astype("string").fillna("")
    # row disagrees if >1 distinct non-empty pred among experiments
    distinct = m.apply(lambda r: len({x for x in r.tolist() if x}), axis=1)
    return df.loc[distinct >= 2].copy()


def multiwise_diff(df: pd.DataFrame, exp_ids: list[str]) -> pd.DataFrame:
    """
    Return rows where at least two pred_label__{exp_id} columns differ (ignoring NaN).
    Compares all provided experiment ids.
    """
    if not exp_ids or len(exp_ids) < 2:
        raise ValueError("Need at least two experiment ids for multiwise_diff.")
    pred_cols = [f"pred_label__{eid}" for eid in exp_ids]
    missing = [c for c in pred_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    keep = [c for c in ["sample_id", "dataset_name", "text", "true_label"] if c in df.columns] + pred_cols
    sub = df[keep].copy()
    # Compare as string (for consistency, also handles None/NaN by string)
    str_pred = sub[pred_cols].astype("string")
    # Row is a diff if at least two non-empty preds are non-equal (set length > 1)
    differing = str_pred.apply(lambda row: len({v for v in row if v != ""}) > 1, axis=1)
    return sub.loc[differing].copy()


def confusions(df: pd.DataFrame, exp_id: str, top_k: int = 20) -> pd.DataFrame:
    pred_col = f"pred_label__{exp_id}"
    if pred_col not in df.columns:
        raise ValueError(f"Missing prediction column: {pred_col}")
    if "true_label" not in df.columns:
        raise ValueError("Missing true_label column")
    s = df.assign(true=df["true_label"].astype("string"), pred=df[pred_col].astype("string")).groupby(["true", "pred"], dropna=False).size()
    c = s.to_frame("count").reset_index().sort_values("count", ascending=False)
    return c.head(int(top_k)).reset_index(drop=True)


def pairwise_agreement_matrix(df: pd.DataFrame) -> pd.DataFrame:
    pred_cols = [c for c in df.columns if c.startswith("pred_label__")]
    ids = [c.split("__", 1)[1] for c in pred_cols]
    if len(pred_cols) < 2:
        return pd.DataFrame()

    m = df[pred_cols].astype("string")
    out: dict[str, dict[str, float]] = {}
    for i, a in enumerate(pred_cols):
        row: dict[str, float] = {}
        for j, b in enumerate(pred_cols):
            if i == j:
                row[ids[j]] = 1.0
                continue
            eq = (m[a] == m[b]).fillna(False)
            row[ids[j]] = float(eq.mean())
        out[ids[i]] = row
    return pd.DataFrame(out).T

