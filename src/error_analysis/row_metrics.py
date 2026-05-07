from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class RowVoteMetricsConfig:
    pred_prefix: str = "pred_label__"
    empty_token: str = ""
    normalize_entropy: bool = True


def _shannon_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p)
    return float(h)


def add_row_vote_metrics(df: pd.DataFrame, *, cfg: RowVoteMetricsConfig = RowVoteMetricsConfig()) -> pd.DataFrame:
    """
    Adds per-row label-consistency metrics across columns that start with `cfg.pred_prefix`.

    Output columns added:
    - n_models: number of prediction columns
    - n_nonempty: predictions present (non-empty string)
    - n_unique: number of distinct non-empty labels
    - majority_label: most common predicted label ("" if none)
    - majority_frac: majority vote share in [0,1] (0 if none)
    - vote_margin: (majority - runner_up) / n_nonempty in [0,1] (0 if <2 non-empty)
    - entropy: Shannon entropy over vote distribution (optionally normalized to [0,1])
    - disagree_any: True if >=2 distinct non-empty labels
    """
    pred_cols = [c for c in df.columns if c.startswith(cfg.pred_prefix)]
    if not pred_cols:
        return df.copy()

    m = df[pred_cols].astype("string").fillna(cfg.empty_token)

    def per_row_metrics(row: pd.Series) -> dict[str, object]:
        vals = [v for v in row.tolist() if v and v != cfg.empty_token]
        n_models = int(len(row))
        n_nonempty = int(len(vals))
        if n_nonempty == 0:
            return {
                "n_models": n_models,
                "n_nonempty": 0,
                "n_unique": 0,
                "majority_label": cfg.empty_token,
                "majority_frac": 0.0,
                "vote_margin": 0.0,
                "entropy": 0.0,
                "disagree_any": False,
            }
        vc = pd.Series(vals, dtype="string").value_counts(dropna=False)
        labels = vc.index.tolist()
        counts = [int(x) for x in vc.tolist()]

        maj = counts[0]
        second = counts[1] if len(counts) > 1 else 0
        majority_label = str(labels[0]) if labels else cfg.empty_token
        majority_frac = float(maj / n_nonempty) if n_nonempty else 0.0
        vote_margin = float((maj - second) / n_nonempty) if n_nonempty else 0.0

        h = _shannon_entropy(counts)
        if cfg.normalize_entropy:
            # Normalize by log(k) where k is number of distinct labels present
            k = max(1, len(counts))
            denom = math.log(k) if k > 1 else 1.0
            h = float(h / denom) if denom else 0.0

        disagree_any = bool(len(counts) >= 2)
        return {
            "n_models": n_models,
            "n_nonempty": n_nonempty,
            "n_unique": int(len(counts)),
            "majority_label": majority_label,
            "majority_frac": majority_frac,
            "vote_margin": vote_margin,
            "entropy": float(h),
            "disagree_any": disagree_any,
        }

    metrics = m.apply(per_row_metrics, axis=1, result_type="expand")
    return df.copy().join(metrics)


def add_probs_confidence_from_probs_json(
    df: pd.DataFrame,
    *,
    exp_id: str,
    probs_col: str | None = None,
    pred_col: str | None = None,
    out_confidence_col: str | None = None,
) -> pd.DataFrame:
    """
    Best-effort: if an experiment stores per-class probs in a `probs__{exp_id}` column as JSON,
    derive `confidence__{exp_id}` as the probability of the predicted label.

    This is useful when confidence is blank but `probs` exists.
    """
    pc = probs_col or f"probs__{exp_id}"
    prc = pred_col or f"pred_label__{exp_id}"
    outc = out_confidence_col or f"confidence__{exp_id}"

    if pc not in df.columns or prc not in df.columns:
        return df.copy()
    if outc in df.columns and pd.to_numeric(df[outc], errors="coerce").notna().any():
        # already has usable confidence
        return df.copy()

    sub = df[[pc, prc]].copy()
    probs_raw = sub[pc]
    probs = probs_raw.apply(lambda x: None if pd.isna(x) else x)

    def parse_obj(x: object) -> dict[str, float] | None:
        if x is None:
            return None
        if isinstance(x, dict):
            return {str(k): float(v) for k, v in x.items() if v is not None}
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return None
            try:
                obj = pd.read_json(pd.Series([s]), typ="series").iloc[0]
                if isinstance(obj, dict):
                    return {str(k): float(v) for k, v in obj.items() if v is not None}
            except Exception:
                return None
        return None

    parsed = probs.apply(parse_obj)

    def pick_conf(row: pd.Series) -> float | None:
        d = row["__p"]
        if d is None:
            return None
        label = str(row["__pred"]) if row["__pred"] is not None else ""
        if not label:
            return None
        v = d.get(label)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    conf = pd.DataFrame({"__p": parsed, "__pred": sub[prc].astype("string")}).apply(pick_conf, axis=1)

    out = df.copy()
    out[outc] = conf
    return out


def add_model_correctness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each `pred_label__{exp_id}` column, adds `is_correct__{exp_id}` if `true_label` exists.
    """
    if "true_label" not in df.columns:
        return df.copy()
    pred_cols = [c for c in df.columns if c.startswith("pred_label__")]
    if not pred_cols:
        return df.copy()
    out = df.copy()
    true = out["true_label"].astype("string")
    for c in pred_cols:
        exp_id = c.split("__", 1)[1]
        out[f"is_correct__{exp_id}"] = out[c].astype("string").eq(true)
    return out


def calibration_bins(
    df: pd.DataFrame,
    *,
    exp_id: str,
    n_bins: int = 10,
    min_count_per_bin: int = 1,
) -> pd.DataFrame:
    """
    Compute a reliability table (confidence vs accuracy) for an experiment, if
    `confidence__{exp_id}` and `pred_label__{exp_id}` and `true_label` are present.

    Returns a DF with bins and: count, avg_conf, accuracy, abs_gap.
    """
    conf_col = f"confidence__{exp_id}"
    pred_col = f"pred_label__{exp_id}"
    if conf_col not in df.columns or pred_col not in df.columns or "true_label" not in df.columns:
        return pd.DataFrame()

    sub = df[[conf_col, pred_col, "true_label"]].copy()
    sub = sub.dropna(subset=[conf_col])
    if not len(sub):
        return pd.DataFrame()

    conf = pd.to_numeric(sub[conf_col], errors="coerce")
    sub = sub.assign(_conf=conf).dropna(subset=["_conf"])
    if not len(sub):
        return pd.DataFrame()

    sub = sub.assign(_is_correct=sub[pred_col].astype("string").eq(sub["true_label"].astype("string")))

    # clamp to [0,1] for binning
    sub["_conf"] = sub["_conf"].clip(lower=0.0, upper=1.0)
    # bins are [0,1] inclusive
    bins = pd.interval_range(start=0.0, end=1.0, periods=int(n_bins), closed="right")
    sub["_bin"] = pd.cut(sub["_conf"], bins=bins, include_lowest=True)

    g = (
        sub.groupby("_bin", dropna=False)
        .agg(count=("_conf", "size"), avg_conf=("_conf", "mean"), accuracy=("_is_correct", "mean"))
        .reset_index()
    )
    g["abs_gap"] = (g["avg_conf"] - g["accuracy"]).abs()
    g = g[g["count"] >= int(min_count_per_bin)].reset_index(drop=True)
    return g.rename(columns={"_bin": "bin"})


def expected_calibration_error(calib_df: pd.DataFrame) -> float | None:
    """
    Compute ECE from a calibration_bins() output.
    """
    if calib_df is None or calib_df.empty:
        return None
    if not {"count", "abs_gap"} <= set(calib_df.columns):
        return None
    total = float(calib_df["count"].sum())
    if total <= 0:
        return None
    return float((calib_df["count"] * calib_df["abs_gap"]).sum() / total)

