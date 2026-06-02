# uni-judge-research

Research codebase for **few-shot text classification with LLM judges**, hybrid ML guardrails, and systematic error analysis. The project is organized around **executed experiment campaigns** under `results/`, not around a single model or script.

**376 runs** are indexed in [`analysis/leaderboard.csv`](analysis/leaderboard.csv) (regenerate with `uv run python analysis/run_all.py`). Metrics below are macro F1 on held-out tiers unless noted.

---

## Research question

Can a small golden seed (10–100 labeled examples) plus a well-instrumented LLM judge match or beat traditional low-data baselines—and where do LLM, ML, and ensemble approaches **disagree** in ways that matter?

Primary benchmarks (tier-200 test slices unless noted):

| Dataset | Role | Best macro F1 (indexed) | Notes |
|---------|------|-------------------------|--------|
| `tweet_eval_irony` | Subjective / pragmatic | **0.990** | Extended LLM eval suite |
| `banking-10` | Business intent routing | **0.965** | 10-class support intents |
| `pubmed_20k_rct` | Structured scientific labels | **0.830** | Label-id vs name scoring sensitive |
| `implicit_hate` | Implicit toxicity / hate | **0.764** | Nuanced boundary cases |

Additional executed tracks: `ag_news`, `glue_sst2`, `reward_bench`, `prosocial_dialogue`, `social_bias_frames`, `lex_glue_scotus`, and others in the leaderboard.

---

## Experiment campaigns (what actually ran)

Results live in `results/<series>/…`. Each campaign uses a different artifact layout; all are merged by the analysis pipeline.

### Harness-backed eval (primary path)

Standard layout: `report.json`, `config.resolved.json`, `predictions.csv`, optional `run_manifest.json`.

| Series | Runs | Focus |
|--------|------|--------|
| `evaluate_google_llm` | 64 | Gemini / Gemma via Google GenAI; batch size, thinking level, seed sweeps |
| `extended_evaluation_ml` | 40 | SetFit, XGBoost, SVM, TF-IDF baselines on shared tiers |
| `extended_evaluation_llm` | 12 | OpenAI-compatible batch LLM eval (Gemini flash, unshuffled controls) |
| `prompt_eng` | 14 | Multilabel confusion probe, self-debate, targeted prompt variants |
| `baseline_performance` | 4 | Reference Google GenAI configs per benchmark |
| `gepa_mipro` | 3 | DSPy GEPA/MIPRO optimize + compiled-program eval |
| `raw/prompt_eng` | 97 | Early prompt-engineering notebooks ported to disk (many manifest-only / prelim) |

Entry points: `scripts/evaluate_google_llm.py`, `scripts/extended_evaluation_{llm,ml}.py`, `scripts/prompt_eng_evaluation.py`, `scripts/eval_gepa_compiled_program.py`.

### Legacy & exploratory campaigns

Flat or summary-driven layouts (see `.gitignore` allowlist). Duplicates vs harness runs are kept when `run_key` differs.

| Series | Runs | Artifacts |
|--------|------|-----------|
| `mvp4_results` | 43 | `mvp4_final_results.json` + `preds_*.csv` — ML vs LLM across 5 public datasets |
| `hf_llms_comparison` | 36 | `metrics_<stamp>.csv`, shared `predictions_long_*.csv` — open-weight HF models |
| `reasoning_spectrum` | 28 | Tier rows from `summary.json` — reasoning-depth ablation on confusing subsets |
| `prosocial_v3` | 19 | `exp_*_summary_*.json` + test-25 preds — safety / intervention routing |
| `debug_accuracy` | 13 | Hypothesis-driven scoring audits (`results.json`) on pubmed eval |
| `mvp_v1` | 3 | Early prosocial MVP (DSPy, XGBoost, Gemma transfer) |

---

## Analysis & leaderboard

All executed work is summarized under **`analysis/`**:

```bash
uv run python analysis/run_all.py          # leaderboard + agreement + reports
uv run python analysis/build_leaderboard.py
uv run python analysis/build_agreement.py
```

| Output | Contents |
|--------|----------|
| `analysis/leaderboard.csv` | ~20 columns per run: dataset, model, F1, tier, timing, paths, format |
| `analysis/plots/agreement/<dataset>__tier<N>/` | Cohen's κ, McNemar pairs, per-class PR |
| `analysis/plots/leaderboard/` | Top / bottom macro-F1 bars |
| `analysis/reports/` | Aggregated harness reports, discovery index |

Agreement groups runs with predictions by **dataset + tier** (up to 25 runs per group). Details: [`analysis/README.md`](analysis/README.md).

Implementation: `src/error_analysis/` (`discover.py`, `legacy_experiments.py`, `agreement_analysis.py`, …).

---

## Architecture (short)

```
data/processed/<dataset>/train_seed/tier_10/   ← golden seed
data/processed/<dataset>/test/tier_200/        ← eval slice
        ↓
src/experiments/run.py  →  LLM / ML predictors  →  results/<series>/<stamp>/<run>/
        ↓
analysis/  →  leaderboard, pairwise agreement, error reports
```

- **LLM judges**: Google GenAI (Gemini 3.1 Flash Lite, Gemma 4 31B), batch + thinking-level controls, JSON structured output.
- **ML guardrails**: SetFit, XGBoost/TF-IDF, embeddings — fast, complementary error profile.
- **Prompt engineering**: multilabel confusion probes, self-debate, GEPA-optimized programs.
- **Metrics**: macro F1, accuracy, per-class PR, confusion stats, token/time tracking.

---

## Setup

Requires **Python 3.10+** and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
# Create .env with GOOGLE_API_KEY (and other provider keys as needed)
```

Run a harness eval (example):

```bash
uv run python scripts/baseline_performance.py --run
uv run python scripts/evaluate_google_llm.py --help
```

Tests (no live LLM by default):

```bash
uv run pytest
```

---

## Repository map

| Path | Purpose |
|------|---------|
| `results/` | All experiment artifacts (metrics git-tracked; predictions local) |
| `analysis/` | Cross-run leaderboard and error analysis |
| `scripts/` | Campaign entry points |
| `src/experiments/` | Config, harness, suites |
| `src/eval/` | Metrics, artifacts, label canonicalization |
| `src/error_analysis/` | Discovery, legacy adapters, agreement stats |
| `raw-experiments/` | Notebooks and one-off explorations (source of many `results/raw/` runs) |
| `data/processed/` | Tiered parquet splits per benchmark |

---

## Conventions

- **New runs** go in `results/<topic>/<experiment>/<timestamp>/` — never overwrite prior timestamps.
- **Scored artifacts** committed per `.gitignore`: `report.json`, metadata, classification reports — not row-level predictions.
- **Leaderboard is the inventory** of executed work; if a run is missing, check its artifact names against `src/error_analysis/discover.py` and `legacy_experiments.py`.

---

## Status

Active research prototype—not a production API. Emphasis is on **empirical comparison across many small-data regimes**, disagreement analysis between model families, and cheap iteration before productizing a judge-as-a-service path.
