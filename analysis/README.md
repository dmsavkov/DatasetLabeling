# Experiment analysis

Unified error analysis and benchmarking over **all** runs under `results/`, regardless of layout (`report.json`, `run_manifest.json`, `full_metadata.json`, `metrics.json`, `full_predictions.json`, `predictions.csv`, etc.).

Past ad-hoc artifacts elsewhere in the repo are ignored; outputs live only under this folder.

## Quick start

From the project root:

```bash
uv run python analysis/run_all.py
```

Or step by step:

```bash
uv run python analysis/build_leaderboard.py
uv run python analysis/build_agreement.py
```

## Outputs

| Path | Description |
|------|-------------|
| `analysis/leaderboard.csv` | One row per run (~20 key columns: dataset, model, F1, timing, paths, …) |
| `analysis/plots/leaderboard/` | Top/bottom macro-F1 bar charts |
| `analysis/plots/agreement/<dataset>__tier<N>/` | Cohen's kappa heatmap, McNemar CSV, per-class PR boxplots |
| `analysis/reports/` | `reports_aggregated.csv`, `loaded_experiments.csv`, `discovery_index.csv` |

## Leaderboard columns

`run_key`, `series`, `campaign`, `suite`, `run_leaf`, `experiment_slug`, `predictor_name`, `dataset_name`, `tier_size`, `n_samples`, `model_kind`, `model_id`, `thinking_level`, `batch_size`, `phase`, `accuracy`, `f1_macro`, `duration_seconds`, `infer_time_s`, `has_predictions`, `predictions_source`, `saved_utc`, `run_dir`, `format`

## Agreement grouping

Runs with loadable predictions are grouped by **`dataset_name` + `tier_size`**. Within each group (up to 25 runs by F1):

- **Cohen's kappa** on predicted labels (pairwise)
- **McNemar** exact test on correctness vs gold
- **Per-class precision/recall** from `full_classification_report.json` or `metrics.json` when present

Large `full_predictions.json` files (>40 MB) are skipped unless `predictions.csv` exists.

## Legacy / non-harness experiments

Also indexed from flat or campaign-specific layouts (see ``.gitignore`` allowlist):

| Series | Source artifacts |
|--------|------------------|
| `mvp4_results` | `mvp4_final_results.json` + `preds_*.csv` |
| `mvp_v1` | `mvp_*_summary_*.json` + paired preds CSV |
| `prosocial_v3` | `exp_*_summary_*.json` + preds CSV |
| `hf_llms_comparison` | `metrics_<stamp>.csv` (+ shared `predictions_long_<stamp>.csv`) |
| `debug_accuracy` | `*/results.json` (+ optional `live_*.csv`) |
| `reasoning_spectrum` | tier rows from `summary.json` |

Duplicate rows vs harness runs are kept when ``run_key`` differs; harness rows win on key collision.

## Implementation

Core logic: `src/error_analysis/` (`discover.py`, `run_record.py`, `leaderboard.py`, `agreement_analysis.py`, …).
