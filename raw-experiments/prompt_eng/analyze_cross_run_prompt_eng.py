# %% [markdown]
# ### Cross-run prompt_eng analysis: errors, disagreements, confidence vs correctness
# Reads latest `full_predictions.json` per experiment slug (or explicit paths), writes CSVs + plots under `results/raw/prompt_eng/_analysis/<stamp>/`.
#
# Run from repo root:
#   uv run python raw-experiments/prompt_eng/analyze_cross_run_prompt_eng.py
# Optional:
#   PROMPT_ENG_SLUGS=baseline_gemini31_pubmed_rct,semantic_entropy_vcsc
#   PROMPT_ENG_PRED_PATHS=D:/path/to/full_predictions.json;D:/path/other.json

# %%
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_ENG_ROOT = Path(os.getenv("PROMPT_ENG_RESULTS_ROOT", REPO_ROOT / "results" / "raw" / "prompt_eng"))

# Trailing comma required — ("foo") is a str, ("foo",) is a 1-tuple.
DEFAULT_SLUGS = (
    "baseline_gemini31_pubmed_rct",
    "semantic_entropy_vcsc",
    "semantic_entropy_vcsc_by_sample_ids",
    "chain_of_verification",
    "blind_dinco_multilabel",
    "harsh_top2_single_critique",
    "gemma_top2_gemini_boolean_match",
    "dinco_alllabel_confidence_gemini",
    "gepa_light_batch_opt",
    "rubric_confidence_gemini",
    "rubric_confidence_gemini_perfect16",
)


def _slugs_from_env() -> list[str]:
    raw = os.getenv("PROMPT_ENG_SLUGS")
    source = raw if raw is not None and raw.strip() else ",".join(DEFAULT_SLUGS)
    return [s.strip() for s in source.split(",") if s.strip()]

LABELS = ["background", "conclusions", "methods", "objective", "results"]
PLOT_SEED = 42


def _norm_label(x: object) -> str:
    s = str(x or "").strip().lower()
    if s in ("conclusion", "concl"):
        s = "conclusions"
    if s == "method":
        s = "methods"
    if s == "result":
        s = "results"
    return s


def gold_from(rec: dict) -> str | None:
    for k in ("gold", "true"):
        if k in rec and rec[k] is not None:
            v = _norm_label(rec[k])
            return v if v in LABELS else str(rec[k]).strip().lower()
    return None


def pred_from(rec: dict) -> str | None:
    crit = rec.get("critique")
    if isinstance(crit, dict) and crit.get("final_label") is not None:
        s = _norm_label(crit["final_label"])
        if s in LABELS or s in ("confusing", "error"):
            return s
        return str(crit["final_label"]).strip().lower()
    for k in ("pred_for_eval", "pred", "final_label"):
        if k not in rec:
            continue
        v = rec[k]
        if v is None:
            continue
        s = _norm_label(v)
        if s in LABELS or s in ("confusing", "error"):
            return s
        return str(v).strip().lower()
    return None


def _json_from_raw(rec: dict) -> dict | None:
    for key in ("raw", "parse_snippet"):
        blob = rec.get(key)
        if not blob or not isinstance(blob, str):
            continue
        try:
            d = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            return d
    return None


def confidence_0_1(rec: dict, *, pred: str | None, slug: str) -> tuple[float | None, str | None]:
    """Return (confidence in [0,1] or None, source label for plots)."""
    slug_l = slug.lower()

    jd = _json_from_raw(rec)
    if jd is not None and jd.get("confidence") is not None:
        try:
            return max(0.0, min(1.0, float(jd["confidence"]) / 100.0)), "reported_model_confidence"
        except (TypeError, ValueError):
            pass

    if "VCavg" in rec:
        try:
            return max(0.0, min(1.0, float(rec["VCavg"]) / 100.0)), "VCavg_semantic_entropy"
        except (TypeError, ValueError):
            pass

    if "coverage_confidence" in rec:
        try:
            return max(0.0, min(1.0, float(rec["coverage_confidence"]))), "coverage_confidence_cove"
        except (TypeError, ValueError):
            pass

    if "normalized_target_confidence" in rec:
        try:
            return max(0.0, min(1.0, float(rec["normalized_target_confidence"]))), "normalized_target_dinco"
        except (TypeError, ValueError):
            pass

    if "confidence_rubric_scaled" in rec:
        try:
            return max(0.0, min(1.0, float(rec["confidence_rubric_scaled"]))), "confidence_rubric_scaled"
        except (TypeError, ValueError):
            pass

    if "confidence_rubric" in rec:
        try:
            r = int(rec["confidence_rubric"])
            if 1 <= r <= 4:
                return max(0.0, min(1.0, r / 4.0)), "confidence_rubric_1_to_4"
        except (TypeError, ValueError):
            pass

    # All-label DiNCo: prefer softmax peak over the hand-tuned combined scalar (see README).
    if "softmax_max_prob" in rec:
        try:
            return max(0.0, min(1.0, float(rec["softmax_max_prob"]))), "softmax_max_prob"
        except (TypeError, ValueError):
            pass

    if "reliability_combined" in rec:
        try:
            return max(0.0, min(1.0, float(rec["reliability_combined"]))), "reliability_combined"
        except (TypeError, ValueError):
            pass

    if "harsh" in slug_l or "top2" in slug_l:
        p1 = rec.get("pass1")
        crit = rec.get("critique")
        if isinstance(p1, dict) and pred and pred in LABELS:
            tt = p1.get("top_two")
            if isinstance(tt, list):
                for it in tt:
                    if isinstance(it, dict) and _norm_label(it.get("label")) == pred:
                        try:
                            return max(0.0, min(1.0, float(it.get("probability", 0)))), "pass1_top2_probability"
                        except (TypeError, ValueError):
                            pass
                for it in tt:
                    if isinstance(it, dict):
                        try:
                            return max(0.0, min(1.0, float(it.get("probability", 0)))), "pass1_top1_probability_fallback"
                        except (TypeError, ValueError):
                            pass
        if isinstance(crit, dict) and crit.get("confidence") is not None:
            try:
                return max(0.0, min(1.0, float(crit["confidence"]) / 100.0)), "critique_confidence"
            except (TypeError, ValueError):
                pass

    traces = rec.get("traces")
    if isinstance(traces, list) and pred and pred in LABELS:
        confs = []
        for t in traces:
            if not isinstance(t, dict):
                continue
            if _norm_label(t.get("label")) == pred:
                try:
                    confs.append(float(t["confidence"]))
                except (TypeError, ValueError, KeyError):
                    continue
        if confs:
            return max(0.0, min(1.0, float(np.mean(confs)) / 100.0)), "mean_trace_conf_matching_pred"

    return None, None


def multilabel_score_fields(rec: dict, pred: str | None) -> dict[str, float | None]:
    """Per-label DiNCo fields for rows_long and alternate reliability plots."""
    out: dict[str, float | None] = {
        "raw_confidence_pred": None,
        "softmax_prob_pred": None,
        "softmax_max_prob": None,
        "softmax_margin_top1_top2": None,
        "label_entropy_bits": None,
        "label_entropy_normalized": None,
        "reliability_combined": None,
    }
    pred_n = _norm_label(pred) if pred else None
    raw_scores = rec.get("raw_scores")
    if isinstance(raw_scores, dict) and pred_n and pred_n in raw_scores:
        try:
            out["raw_confidence_pred"] = max(0.0, min(1.0, float(raw_scores[pred_n])))
        except (TypeError, ValueError):
            pass
    softmax_probs = rec.get("softmax_probs")
    if isinstance(softmax_probs, dict) and pred_n and pred_n in softmax_probs:
        try:
            out["softmax_prob_pred"] = max(0.0, min(1.0, float(softmax_probs[pred_n])))
        except (TypeError, ValueError):
            pass
    for key in (
        "softmax_max_prob",
        "softmax_margin_top1_top2",
        "label_entropy_bits",
        "label_entropy_normalized",
        "reliability_combined",
    ):
        if key not in rec:
            continue
        try:
            out[key] = float(rec[key])
        except (TypeError, ValueError):
            pass
    return out


def discover_latest_predictions(root: Path, slug: str) -> Path | None:
    d = root / slug
    if not d.is_dir():
        return None
    best: tuple[float, Path] | None = None
    for child in d.iterdir():
        if not child.is_dir():
            continue
        pred_file = child / "full_predictions.json"
        if not pred_file.is_file():
            continue
        mtime = pred_file.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, pred_file)
    return best[1] if best else None


def load_predictions_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _row_index(rec: dict, fallback: int) -> int:
    if rec.get("i") is not None:
        return int(rec["i"])
    sid = rec.get("sample_id")
    if isinstance(sid, str) and sid.strip():
        return fallback
    return fallback


def rows_to_dataframe(rows: list[dict], *, slug: str, run_label: str) -> pd.DataFrame:
    out: list[dict] = []
    for pos, rec in enumerate(rows):
        if not isinstance(rec, dict):
            continue
        g = gold_from(rec)
        p = pred_from(rec)
        if g is None:
            continue
        idx = _row_index(rec, pos)
        conf, csrc = confidence_0_1(rec, pred=p, slug=slug)
        in_eval = g in LABELS and p in LABELS
        correct = bool(in_eval and g == p) if in_eval else None
        row = {
            "run_label": run_label,
            "slug": slug,
            "i": idx,
            "sample_id": rec.get("sample_id"),
            "gold": g,
            "pred": p,
            "in_eval": in_eval,
            "correct": correct,
            "confidence": conf,
            "confidence_source": csrc,
        }
        row.update(multilabel_score_fields(rec, p))
        if rec.get("confidence_rubric") is not None:
            try:
                row["confidence_rubric"] = int(rec["confidence_rubric"])
            except (TypeError, ValueError):
                pass
        if rec.get("confidence_rubric_scaled") is not None:
            try:
                row["confidence_rubric_scaled"] = float(rec["confidence_rubric_scaled"])
            except (TypeError, ValueError):
                pass
        out.append(row)
    return pd.DataFrame(out)


def plot_confusion(pred: pd.Series, gold: pd.Series, title: str, out_path: Path) -> None:
    mask = pred.notna() & gold.notna()
    p = pred[mask].map(_norm_label)
    g = gold[mask].map(_norm_label)
    ok = p.isin(LABELS) & g.isin(LABELS)
    p, g = p[ok], g[ok]
    if len(p) == 0:
        return
    cm = confusion_matrix(g, p, labels=LABELS)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=LABELS, yticklabels=LABELS, ylabel="Gold", xlabel="Predicted", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_reliability(
    df: pd.DataFrame,
    slug: str,
    out_path: Path,
    *,
    x_col: str = "confidence",
    title: str | None = None,
) -> None:
    """Per-sample scatter: confidence (x) vs outcome (y). No probability binning."""
    if x_col not in df.columns:
        return
    sub = df[(df["slug"] == slug) & (df[x_col].notna()) & (df["correct"].notna())].copy()
    if sub.empty:
        return
    x = sub[x_col].astype(float).values
    y = sub["correct"].astype(float).values
    # Tiny vertical jitter so overlapping points remain visible
    rng = np.random.default_rng(PLOT_SEED)
    jitter = rng.uniform(-0.03, 0.03, size=len(y))
    y_plot = np.clip(y + jitter, -0.08, 1.08)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, label="perfect calibration")
    wrong = sub["correct"] == False
    right = sub["correct"] == True
    ax.scatter(
        x[wrong.values],
        y_plot[wrong.values],
        c="#e74c3c",
        alpha=0.75,
        s=36,
        edgecolors="white",
        linewidths=0.4,
        label=f"wrong (n={int(wrong.sum())})",
    )
    ax.scatter(
        x[right.values],
        y_plot[right.values],
        c="#2ecc71",
        alpha=0.75,
        s=36,
        edgecolors="white",
        linewidths=0.4,
        label=f"right (n={int(right.sum())})",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel(f"{x_col} (higher → more confident)")
    ax.set_ylabel("Correct (0 = wrong, 1 = right)")
    ax.set_title(title or f"Reliability (unbinned): {slug}\n({x_col})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_conf_vs_correct_box(
    df: pd.DataFrame,
    slug: str,
    out_path: Path,
    *,
    x_col: str = "confidence",
    title: str | None = None,
) -> None:
    """Per-sample confidence by outcome (box + jittered points, no binning)."""
    if x_col not in df.columns:
        return
    sub = df[(df["slug"] == slug) & (df[x_col].notna()) & (df["correct"].notna())].copy()
    if sub.empty or sub["correct"].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    positions = [1, 2]
    for pos, ok in zip(positions, (False, True)):
        vals = sub.loc[sub["correct"] == ok, x_col].astype(float).values
        if len(vals) == 0:
            continue
        bp = ax.boxplot(
            [vals],
            positions=[pos],
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            showmeans=True,
            meanline=True,
        )
        bp["boxes"][0].set_facecolor("#fadbd8" if not ok else "#d5f5e3")
        bp["boxes"][0].set_alpha(0.5)
        jitter_x = np.random.default_rng(PLOT_SEED + pos).uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter_x,
            vals,
            alpha=0.85,
            s=28,
            c="#c0392b" if not ok else "#27ae60",
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(["Wrong", "Right"])
    ax.set_ylabel(f"{x_col} (per sample)")
    ax.set_title(title or f"Confidence by outcome (unbinned)\n{slug} ({x_col})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_pairwise_disagreement(wide: pd.DataFrame, pred_cols: list[str], out_path: Path) -> None:
    if len(pred_cols) < 2:
        return
    mat = np.zeros((len(pred_cols), len(pred_cols)))
    for a, ca in enumerate(pred_cols):
        for b, cb in enumerate(pred_cols):
            if a == b:
                continue
            m = wide[ca].notna() & wide[cb].notna()
            if m.sum() == 0:
                continue
            mat[a, b] = float((wide.loc[m, ca] != wide.loc[m, cb]).mean())
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="Oranges", vmin=0, vmax=max(0.05, mat.max()))
    fig.colorbar(im, ax=ax, label="Fraction pred differs")
    ax.set_xticks(np.arange(len(pred_cols)))
    ax.set_yticks(np.arange(len(pred_cols)))
    ax.set_xticklabels(pred_cols, rotation=45, ha="right")
    ax.set_yticklabels(pred_cols)
    ax.set_title("Pairwise prediction disagreement (same index)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    root = Path(os.getenv("PROMPT_ENG_RESULTS_ROOT", DEFAULT_PROMPT_ENG_ROOT)).resolve()
    explicit = os.getenv("PROMPT_ENG_PRED_PATHS", "").strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = root / "_analysis" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, str, Path]] = []  # (slug, run_folder_name, predictions_path)
    if explicit:
        for part in explicit.replace(";", ",").split(","):
            p = Path(part.strip()).expanduser()
            if not p.is_file():
                continue
            # .../<slug>/<stamp>/full_predictions.json
            slug = p.parent.parent.name
            runs.append((slug, p.parent.name, p))
    else:
        slugs = _slugs_from_env()
        for slug in slugs:
            latest = discover_latest_predictions(root, slug)
            if latest is None:
                continue
            runs.append((slug, latest.parent.name, latest))

    if not runs:
        raise SystemExit(f"No runs found under {root}. Set PROMPT_ENG_PRED_PATHS or check slugs.")

    frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for slug, folder, pred_path in runs:
        rows = load_predictions_json(pred_path)
        run_label = f"{slug}::{folder}"
        df = rows_to_dataframe(rows, slug=slug, run_label=run_label)
        df["predictions_path"] = str(pred_path)
        frames.append(df)
        sub = df[df["in_eval"]]
        acc = float(accuracy_score(sub["gold"], sub["pred"])) if len(sub) else float("nan")
        n_conf = int(df["confidence"].notna().sum())
        summary_rows.append(
            {
                "slug": slug,
                "run_folder": folder,
                "n_rows": len(df),
                "n_eval_labels": len(sub),
                "accuracy_eval_labels": acc,
                "n_with_confidence": n_conf,
                "predictions_path": str(pred_path),
            }
        )
        plot_confusion(df["pred"], df["gold"], f"{slug}\n{folder}", out_dir / f"confusion__{slug}.png")

    long_df = pd.concat(frames, ignore_index=True)
    long_df.to_csv(out_dir / "rows_long.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_by_run.csv", index=False)

    # Wide merge: same index i only where gold matches across all included slugs on shared indices
    gold_by_run: dict[str, pd.Series] = {}
    for slug in long_df["slug"].unique():
        sub = long_df[long_df["slug"] == slug].drop_duplicates("i").set_index("i")["gold"]
        gold_by_run[str(slug)] = sub

    slugs_all = sorted(gold_by_run.keys(), key=str)
    common_idx = None
    for s in slugs_all:
        ix = gold_by_run[s].index
        common_idx = ix if common_idx is None else common_idx.intersection(ix)

    aligned_slugs: list[str] = []
    if common_idx is not None and len(common_idx) > 0:
        ref = "baseline_gemini31_pubmed_rct" if "baseline_gemini31_pubmed_rct" in gold_by_run else slugs_all[0]
        ref_g = gold_by_run[ref].reindex(common_idx)
        if ref_g.notna().all():
            for s in slugs_all:
                g = gold_by_run[s].reindex(common_idx)
                if g.notna().all() and (g == ref_g).all():
                    aligned_slugs.append(s)

    if aligned_slugs and common_idx is not None:
        base = pd.DataFrame({"i": common_idx, "gold": gold_by_run[aligned_slugs[0]].reindex(common_idx).values})
        for slug in aligned_slugs:
            d = long_df[long_df["slug"] == slug].drop_duplicates("i").set_index("i")
            base[f"pred__{slug}"] = d["pred"].reindex(common_idx).values
            base[f"conf__{slug}"] = d["confidence"].reindex(common_idx).values
            base[f"correct__{slug}"] = d["correct"].reindex(common_idx).values
        base.to_csv(out_dir / "wide_aligned_by_i.csv", index=False)

        pred_cols = [c for c in base.columns if c.startswith("pred__")]
        if len(pred_cols) >= 2:
            plot_pairwise_disagreement(base, pred_cols, out_dir / "pairwise_disagreement.png")
            mat = base[pred_cols].astype(str)
            n_unique = mat.nunique(axis=1)
            wide_out = base.copy()
            wide_out["n_distinct_preds"] = n_unique
            wide_out["any_disagreement"] = n_unique > 1
            wide_out.to_csv(out_dir / "wide_with_disagreement.csv", index=False)
            disagree_rows = []
            for _, row in wide_out.iterrows():
                preds = [row[c] for c in pred_cols if pd.notna(row.get(c)) and str(row[c]) not in ("nan", "None", "nan")]
                if len(set(preds)) <= 1:
                    continue
                disagree_rows.append(
                    {
                        "i": row["i"],
                        "gold": row["gold"],
                        "preds": "|".join(sorted(set(preds))),
                        "n_distinct": len(set(preds)),
                    }
                )
            pd.DataFrame(disagree_rows).to_csv(out_dir / "indices_where_runs_disagree.csv", index=False)

    # Per-slug reliability / confidence plots (primary uses `confidence` column)
    _ALT_RELIABILITY_COLS = (
        ("raw_confidence_pred", "model raw score on predicted label"),
        ("reliability_combined", "0.5·softmax_max + 0.5·(1−H/H_max)"),
        ("softmax_margin_top1_top2", "softmax top1 − top2"),
        ("label_entropy_normalized", "1 − H/H_max (certainty from entropy)"),
    )
    for slug in long_df["slug"].unique():
        sub_slug = long_df[long_df["slug"] == slug]
        plot_reliability(long_df, slug, out_dir / f"reliability__{slug}.png")
        plot_conf_vs_correct_box(long_df, slug, out_dir / f"conf_box__{slug}.png")
        for col, desc in _ALT_RELIABILITY_COLS:
            if not sub_slug[col].notna().any():
                continue
            plot_reliability(
                long_df,
                slug,
                out_dir / f"reliability__{slug}__{col}.png",
                x_col=col,
                title=f"Reliability: {slug}\n{col} — {desc}",
            )
            plot_conf_vs_correct_box(
                long_df,
                slug,
                out_dir / f"conf_box__{slug}__{col}.png",
                x_col=col,
                title=f"{slug}: {col}",
            )

    # Bar: accuracy
    fig, ax = plt.subplots(figsize=(8, 4))
    sdf = pd.DataFrame(summary_rows).sort_values("accuracy_eval_labels", ascending=False)
    ax.barh(sdf["slug"], sdf["accuracy_eval_labels"])
    ax.set_xlabel("Accuracy (gold vs pred, eval labels only)")
    ax.set_title("Prompt_eng runs — latest per slug")
    ax.set_xlim(0, 1)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_slug.png", dpi=140)
    plt.close(fig)

    readme = out_dir / "README.txt"
    readme.write_text(
        "Artifacts:\n"
        "- rows_long.csv: one row per (run, index) with gold, pred, correct, confidence, source.\n"
        "- summary_by_run.csv: accuracy and paths.\n"
        "- wide_aligned_by_i.csv: merged preds where gold sequences match across runs (same indices).\n"
        "- wide_with_disagreement.csv: adds n_distinct_preds when multiple slugs aligned.\n"
        "- indices_where_runs_disagree.csv: rows where models picked different labels.\n"
        "- confusion__*.png: confusion matrices.\n"
        "- reliability__*.png: primary confidence vs correct (per sample, no binning).\n"
        "- reliability__*__<metric>.png: alternate x-axes when all-label softmax fields exist.\n"
        "- conf_box__*.png / conf_box__*__<metric>.png: same metrics by wrong/right.\n"
        "- pairwise_disagreement.png: fraction of indices where preds differ (aligned runs only).\n"
        "- accuracy_by_slug.png: quick ranking.\n"
        "\n"
        "Primary confidence (column `confidence`):\n"
        "  - DiNCo all-label runs: softmax_max_prob (winner mass after softmax over 5 labels).\n"
        "  - Other runs: heuristic per schema (model JSON %, VCavg, CoVe coverage, top2 prob, etc.).\n"
        "\n"
        "DiNCo all-label extra columns in rows_long.csv:\n"
        "  - raw_scores / raw_confidence_pred: independent 0–1 scores per label (not a distribution).\n"
        "  - softmax_probs / softmax_prob_pred: softmax(raw_scores); pred = argmax; prob_pred == max_prob.\n"
        "  - softmax_margin_top1_top2: gap between top-2 softmax masses.\n"
        "  - label_entropy_bits, label_entropy_normalized: spread of softmax (normalized = 1 − H/log2(5)).\n"
        "  - reliability_combined: 0.5·softmax_max_prob + 0.5·label_entropy_normalized (legacy blend).\n"
        "\n"
        "Runs with different EVAL_N or different gold rows are only partially merged; check summary n_rows.\n",
        encoding="utf-8",
    )
    print(f"Wrote analysis to: {out_dir}")


if __name__ == "__main__":
    main()
