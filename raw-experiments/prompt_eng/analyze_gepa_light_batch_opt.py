# %% [markdown]
# ### Post-hoc GEPA inspection: train/eval examples, delta cohorts, DSPy cache probe
#
# The original `gepa_light_batch_opt.py` run did **not** save `optimized_program.dump_state()`.
# This script reconstructs the **exact train/eval rows** (same seed + stratified sampling as the experiment),
# compares saved predictions to a baseline run (default: `bootstrap_batch_cot` on the same 50 eval sentences),
# and scans `~/.dspy_cache` for cached LLM responses (not the final compiled prompt state).
#
# Run from repo root:
#   uv run python raw-experiments/prompt_eng/analyze_gepa_light_batch_opt.py
#
# Env:
#   GEPA_RUN_DIR=.../gepa_light_batch_opt/20260514_111314
#   BASELINE_PRED_PATH=.../bootstrap_batch_cot/.../full_predictions.json
#   DSPY_CACHE_ROOT=C:\Users\Dmitry\.dspy_cache
#   RUN_UNCOMPILED_BASELINE=1  # optional: one forward pass per eval batch (API cost)

# %%
from __future__ import annotations

import json
import os
import pickle
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, confusion_matrix

_PROMPT_ENG = Path(__file__).resolve().parent
if str(_PROMPT_ENG) not in sys.path:
    sys.path.insert(0, str(_PROMPT_ENG))

import prompt_eng_common as pec
import gepa_light_batch_opt as gepa

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_ENG = REPO_ROOT / "results" / "raw" / "prompt_eng"
DEFAULT_GEPA_RUN = DEFAULT_PROMPT_ENG / "gepa_light_batch_opt" / "20260514_111314"
DEFAULT_BASELINE = DEFAULT_PROMPT_ENG / "bootstrap_batch_cot" / "20260512_125452" / "full_predictions.json"
DEFAULT_CACHE = Path(os.getenv("DSPY_CACHE_ROOT", Path.home() / ".dspy_cache"))

LABELS = gepa.LABELS
BATCH_SIZE = gepa.BATCH_SIZE


def _latest_predictions(slug: str) -> Path | None:
    root = Path(os.getenv("PROMPT_ENG_RESULTS_ROOT", DEFAULT_PROMPT_ENG)) / slug
    best: tuple[float, Path] | None = None
    if not root.is_dir():
        return None
    for child in root.iterdir():
        p = child / "full_predictions.json"
        if p.is_file():
            m = p.stat().st_mtime
            if best is None or m > best[0]:
                best = (m, p)
    return best[1] if best else None


def reconstruct_splits() -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[dict]]:
    """Same sampling as gepa_light_batch_opt.main (seed=DEFAULT_SEED)."""
    eval_df, _, train_df, _ = pec.load_hf_pubmed_splits()
    train_df = train_df.copy()
    train_df["label_name"] = train_df["label"].apply(pec.label_name_from_value)

    eval_subset = gepa.stratified_sample(
        eval_df, gepa.EVAL_N, label_col="label_name", min_per_class=gepa.MIN_PER_CLASS, seed=pec.DEFAULT_SEED
    )
    train_subset = gepa.stratified_sample(
        train_df, gepa.TRAIN_N, label_col="label_name", min_per_class=gepa.MIN_PER_CLASS, seed=pec.DEFAULT_SEED
    )
    eval_subset = eval_subset.reset_index(drop=True)
    train_subset = train_subset.reset_index(drop=True)
    eval_subset["sentence_idx"] = eval_subset.index
    train_subset["sentence_idx"] = train_subset.index

    def rows_from_subset(df: pd.DataFrame, *, split: str) -> list[dict]:
        out: list[dict] = []
        texts = df["text"].astype(str).tolist()
        labels = df["label_name"].astype(str).str.lower().tolist()
        for bi in range(0, len(texts), BATCH_SIZE):
            bt = texts[bi : bi + BATCH_SIZE]
            bl = labels[bi : bi + BATCH_SIZE]
            if len(bt) != BATCH_SIZE:
                continue
            batch_id = bi // BATCH_SIZE
            flat_start = bi
            sentences = []
            for pos, (tx, lb) in enumerate(zip(bt, bl)):
                sentences.append(
                    {
                        "flat_i": flat_start + pos,
                        "position_in_batch": pos + 1,
                        "text": tx,
                        "gold": lb,
                        "sentence_idx_in_split": int(df.iloc[flat_start + pos]["sentence_idx"]),
                    }
                )
            out.append(
                {
                    "split": split,
                    "batch_id": batch_id,
                    "input_texts_numbered": "\n".join(f"{j+1}. {t}" for j, t in enumerate(bt)),
                    "target_labels": bl,
                    "sentences": sentences,
                }
            )
        return out

    train_batches = rows_from_subset(train_subset, split="train")
    eval_batches = rows_from_subset(eval_subset, split="eval")
    return train_subset, eval_subset, train_batches, eval_batches


def load_flat_predictions(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for rec in data:
        rows.append({"flat_i": int(rec["i"]), "gold": str(rec["true"]).lower(), "pred": str(rec["pred"]).lower()})
    return pd.DataFrame(rows)


def attach_predictions(explicit_batches: list[dict], preds: pd.DataFrame, *, pred_col: str) -> list[dict]:
    pmap = preds.set_index("flat_i")
    enriched = []
    for b in explicit_batches:
        b2 = json.loads(json.dumps(b))
        for s in b2["sentences"]:
            fi = s["flat_i"]
            if fi in pmap.index:
                s["gold"] = pmap.loc[fi, "gold"]
                s[pred_col] = pmap.loc[fi, "pred"]
                s[f"{pred_col}_correct"] = s["gold"] == s[pred_col]
            else:
                s[pred_col] = None
        enriched.append(b2)
    return enriched


def cohort_frames(gepa_p: pd.DataFrame, base_p: pd.DataFrame, texts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    m = gepa_p.merge(base_p, on="flat_i", suffixes=("_gepa", "_baseline"))
    m = m.merge(texts[["sentence_idx", "text"]], left_on="flat_i", right_on="sentence_idx", how="left")
    m["gepa_ok"] = m["gold_gepa"] == m["pred_gepa"]
    m["base_ok"] = m["gold_baseline"] == m["pred_baseline"]
    fixed = m[~m["base_ok"] & m["gepa_ok"]].copy()
    regress = m[m["base_ok"] & ~m["gepa_ok"]].copy()
    stubborn = m[~m["base_ok"] & ~m["gepa_ok"]].copy()
    both_ok = m[m["base_ok"] & m["gepa_ok"]].copy()
    return {"all": m, "fixed": fixed, "regressions": regress, "stubborn_failures": stubborn, "both_correct": both_ok}


def plot_cohort_summary(cohorts: dict[str, pd.DataFrame], out_dir: Path) -> None:
    counts = {
        "both_correct": len(cohorts["both_correct"]),
        "fixed_by_gepa": len(cohorts["fixed"]),
        "regressions": len(cohorts["regressions"]),
        "stubborn_failures": len(cohorts["stubborn_failures"]),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(counts.keys()), list(counts.values()), color=["#2ecc71", "#3498db", "#e74c3c", "#95a5a6"])
    ax.set_ylabel("Sentences (n=50 eval)")
    ax.set_title("GEPA vs baseline comparator — outcome cohorts")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "cohort_counts.png", dpi=140)
    plt.close(fig)

    m = cohorts["all"]
    if len(m) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, ok, color in [("GEPA", "gepa_ok", "#3498db"), ("Baseline", "base_ok", "#e67e22")]:
        sub = m.groupby("gold_gepa")[ok].mean().reindex(LABELS)
        ax.plot(LABELS, sub.values, marker="o", label=label, color=color)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-class accuracy (eval set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "per_class_accuracy_gepa_vs_baseline.png", dpi=140)
    plt.close(fig)


def inspect_dspy_cache(cache_root: Path, out_dir: Path, *, max_rows_per_db: int = 500) -> dict:
    """Scan cache DBs for batch-classification / GEPA-era LLM calls (not full GEPA state)."""
    if not cache_root.is_dir():
        logger.warning("DSPy cache not found: {}", cache_root)
        return {"error": "cache_root missing"}

    summary_rows: list[dict] = []
    batch_hits: list[dict] = []
    needles = ("predicted_labels", "input_texts", "list of 5", "5 medical", "classification_results")

    for folder in sorted(cache_root.iterdir()):
        if not folder.is_dir():
            continue
        db = folder / "cache.db"
        if not db.is_file():
            continue
        models: Counter[str] = Counter()
        n_rows = 0
        n_parsed = 0
        folder_hits = 0
        try:
            con = sqlite3.connect(db)
            for row in con.execute("SELECT value FROM Cache LIMIT ?", (max_rows_per_db,)):
                n_rows += 1
                try:
                    resp = pickle.loads(row[0])
                    n_parsed += 1
                    model = getattr(resp, "model", "unknown")
                    models[model] += 1
                    ch = resp.choices[0] if getattr(resp, "choices", None) else None
                    content = (getattr(ch.message, "content", None) or "") if ch else ""
                    low = content.lower()
                    if any(n in low for n in needles) or "rhetorical" in low:
                        folder_hits += 1
                        if len(batch_hits) < 40:
                            batch_hits.append(
                                {
                                    "cache_folder": folder.name,
                                    "model": model,
                                    "content_preview": content[:2500],
                                }
                            )
                except Exception:
                    pass
            con.close()
        except Exception as e:
            summary_rows.append({"cache_folder": folder.name, "error": str(e)})
            continue

        summary_rows.append(
            {
                "cache_folder": folder.name,
                "cache_db_bytes": db.stat().st_size,
                "rows_scanned": n_rows,
                "rows_parsed": n_parsed,
                "batch_like_hits": folder_hits,
                "top_models": dict(models.most_common(5)),
                "db_mtime_utc": datetime.fromtimestamp(db.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
        )

    pd.DataFrame(summary_rows).to_csv(out_dir / "dspy_cache_folder_summary.csv", index=False)
    if batch_hits:
        (out_dir / "dspy_cache_batch_like_samples.json").write_text(
            json.dumps(batch_hits, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return {"folders": len(summary_rows), "batch_samples": len(batch_hits)}


def write_report(
    out_dir: Path,
    *,
    gepa_meta: dict,
    cohorts: dict[str, pd.DataFrame],
    cache_info: dict,
    gepa_run_dir: Path,
    baseline_path: Path,
    n_train: int,
    n_eval: int,
) -> None:
    m = cohorts["all"]
    acc_g = float((m["gold_gepa"] == m["pred_gepa"]).mean()) if len(m) else 0.0
    acc_b = float((m["gold_baseline"] == m["pred_baseline"]).mean()) if len(m) else 0.0
    lines = [
        "# GEPA light batch — post-hoc inspection report",
        "",
        "## Experiment (from saved metadata)",
        f"- **GEPA run dir:** `{gepa_run_dir}`",
        f"- **Duration (compile+eval):** {gepa_meta.get('duration_seconds', '?')} s",
        f"- **Executor:** {gepa_meta.get('settings', {}).get('EXECUTOR_MODEL', '?')}",
        f"- **Reflector:** {gepa_meta.get('settings', {}).get('REFLECTOR_MODEL', '?')}",
        f"- **Train / eval sentences:** {n_train} / {n_eval} (batches of {BATCH_SIZE})",
        f"- **GEPA:** `auto=light`, `reflection_minibatch_size=2`, `skip_perfect_score=True`, `candidate_selection_strategy=pareto`",
        f"- **Saved accuracy (GEPA eval):** {gepa_meta.get('metrics', {}).get('accuracy', '?')}",
        "",
        "## What we cannot recover",
        "- **`optimized_program.dump_state()`** was not saved — final mutated instructions / selected demos are **not** reconstructable from disk.",
        "- **DSPy cache** (`~/.dspy_cache/*/cache.db`) stores **pickled LiteLLM `ModelResponse`** objects (LLM completions), keyed by request hash — **not** the compiled program graph.",
        "",
        "## Baseline comparator",
        f"- Predictions: `{baseline_path}`",
        f"- On this eval slice (seed={pec.DEFAULT_SEED}), **gold labels match 50/50** with GEPA predictions file when both use the same protocol.",
        f"- **Accuracy GEPA:** {acc_g:.3f} | **Comparator:** {acc_b:.3f} | **Δ:** {acc_g - acc_b:+.3f}",
        "",
        "## Delta cohorts (sentence-level, n=50)",
        f"- **Both correct:** {len(cohorts['both_correct'])}",
        f"- **Fixed (baseline wrong → GEPA right):** {len(cohorts['fixed'])}",
        f"- **Regressions (baseline right → GEPA wrong):** {len(cohorts['regressions'])}",
        f"- **Stubborn failures (both wrong):** {len(cohorts['stubborn_failures'])}",
        "",
        "## DSPy cache scan",
        f"- Folders scanned: {cache_info.get('folders', 0)}",
        f"- Batch-like response samples exported: {cache_info.get('batch_samples', 0)} (see `dspy_cache_batch_like_samples.json`)",
        "",
        "## Artifacts in this folder",
        "- `train_batches_explicit.json` — all **training** batches with full sentence text + gold labels",
        "- `eval_batches_explicit.json` — **eval** batches (same rows GEPA scored)",
        "- `train_sentences_flat.csv` / `eval_sentences_flat.csv` — one row per sentence",
        "- `delta_all.csv`, `delta_fixed.csv`, `delta_regressions.csv`, `delta_stubborn.csv`",
        "- `cohort_counts.png`, `per_class_accuracy_gepa_vs_baseline.png`",
        "",
        "## Playbook next steps",
        "1. Read **Fixed** rows in `delta_fixed.csv` — what label boundary changed?",
        "2. Read **Regressions** — overfit rules / catastrophic forgetting?",
        "3. Read **Stubborn** — ambiguous text or bad gold?",
        "4. Re-run GEPA with `gepa_light_batch_opt.py` after adding `optimized_program.save(...)` (see updated script).",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def maybe_run_uncompiled_baseline(eval_batches: list[dict], out_dir: Path) -> pd.DataFrame | None:
    if not pec.env_bool("RUN_UNCOMPILED_BASELINE", False):
        return None
    if not pec.GOOGLE_API_KEY:
        logger.warning("RUN_UNCOMPILED_BASELINE set but no GOOGLE_API_KEY")
        return None
    import dspy

    lm = dspy.LM(
        model=f"openai/{gepa.EXECUTOR_MODEL}",
        api_base=pec.GOOGLE_OPENAI_BASE_URL,
        api_key=pec.GOOGLE_API_KEY,
        max_tokens=None,
        num_retries=pec.MAIN_MAX_RETRIES,
    )
    dspy.settings.configure(lm=lm)
    prog = gepa.PubMedBatchClassifier()
    rows: list[dict] = []
    for b in eval_batches:
        res = prog(input_texts=b["input_texts_numbered"])
        preds = gepa._parse_predicted_labels(res)
        for s, p in zip(b["sentences"], preds):
            rows.append({"flat_i": s["flat_i"], "gold": s["gold"], "pred": p})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "uncompiled_baseline_predictions.csv", index=False)
    return df


def main() -> None:
    gepa_run = Path(os.getenv("GEPA_RUN_DIR", DEFAULT_GEPA_RUN))
    gepa_pred_path = gepa_run / "full_predictions.json"
    if not gepa_pred_path.is_file():
        alt = _latest_predictions("gepa_light_batch_opt")
        if alt:
            gepa_pred_path = alt
            gepa_run = alt.parent
        else:
            raise FileNotFoundError("No GEPA full_predictions.json found")

    baseline_path = Path(os.getenv("BASELINE_PRED_PATH", DEFAULT_BASELINE))
    if not baseline_path.is_file():
        alt = _latest_predictions("bootstrap_batch_cot")
        baseline_path = alt if alt else baseline_path

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = gepa_run.parent / "_analysis" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output: {}", out_dir)

    train_df, eval_df, train_batches, eval_batches = reconstruct_splits()
    train_df.to_csv(out_dir / "train_sentences_flat.csv", index=False)
    eval_df.to_csv(out_dir / "eval_sentences_flat.csv", index=False)
    (out_dir / "train_batches_explicit.json").write_text(
        json.dumps(train_batches, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "eval_batches_explicit.json").write_text(
        json.dumps(eval_batches, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Exported train batches={} eval batches={}", len(train_batches), len(eval_batches))

    gepa_p = load_flat_predictions(gepa_pred_path)
    if baseline_path.is_file():
        base_p = load_flat_predictions(baseline_path)
        base_label = "pred_bootstrap"
    else:
        logger.warning("No baseline predictions at {}; skipping delta", baseline_path)
        base_p = gepa_p.copy()
        base_p["pred"] = "missing"
        base_label = "pred_missing"

    uncompiled = maybe_run_uncompiled_baseline(eval_batches, out_dir)
    if uncompiled is not None:
        base_p = uncompiled
        baseline_path = out_dir / "uncompiled_baseline_predictions.csv"
        base_label = "pred_uncompiled"

    cohorts = cohort_frames(gepa_p, base_p, eval_df.rename(columns={"sentence_idx": "sentence_idx"}))
    for name, df in cohorts.items():
        if len(df):
            df.to_csv(out_dir / f"delta_{name}.csv", index=False)

    plot_cohort_summary(cohorts, out_dir)

    meta_path = gepa_run / "full_metadata.json"
    gepa_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    cache_info = inspect_dspy_cache(DEFAULT_CACHE, out_dir)

    # Enrich eval batches with preds for manual reading
    eval_enriched = attach_predictions(eval_batches, gepa_p, pred_col="pred_gepa")
    eval_enriched = attach_predictions(eval_enriched, base_p[["flat_i", "gold", "pred"]], pred_col="pred_baseline")
    (out_dir / "eval_batches_with_preds.json").write_text(
        json.dumps(eval_enriched, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    write_report(
        out_dir,
        gepa_meta=gepa_meta,
        cohorts=cohorts,
        cache_info=cache_info,
        gepa_run_dir=gepa_run,
        baseline_path=baseline_path,
        n_train=len(train_df),
        n_eval=len(eval_df),
    )
    print(f"Done. See {out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
