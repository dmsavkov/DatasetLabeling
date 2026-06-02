# GEPA pipeline (huge prediction → optimizer sets → MIPRO → eval)

End-to-end prompt optimization for **extended-suite** datasets: `pubmed_20k_rct`, `banking-10`, `tweet_eval_irony`, `implicit_hate`.

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| Processed data | `data/processed/<dataset>/train_seed/tier_5000/samples.parquet` |
| Dataset card | `data/processed/<dataset>/dataset_card.json` |
| API keys | `GOOGLE_API_KEY` or `GEMINI_API_KEY` in `.env` |
| Test tiers | `test/tier_20` and `tier_200` used to **exclude** golden rows from mining |

## Pipeline order

```text
1. huge prediction   → data/huge_prediction_representatives/<dataset>/<stamp>/
2. GEPA sets         → data/gepa_optimizer_sets/<dataset>/<stamp>/
3. MIPRO optimize    → results/gepa_mipro/<dataset>/optimize/<stamp>/
4. Eval compiled     → results/gepa_mipro/<dataset>/eval/<stamp>/  (optional)
```

Each step reads the **previous stamp directory** you pass explicitly (or uses latest under `gepa_optimizer_sets` for step 3 only).

---

## Step 1 — Huge prediction

**Script:** `scripts/run_huge_prediction_representatives.py`

**Role:** Embed train pool → K-means centroids → expand to prediction pool → **Gemma batch LLM** on ~500 rows → contrastive edges on errors.

**Defaults (align with `gemma4_31b_think_high` eval):**

| Param | Default | Notes |
|-------|---------|--------|
| `--model-id` | `gemma-4-31b-it` | Executor |
| `--batch-size` | `10` | Must divide `prediction-size` |
| `--thinking-level` | `high` | |
| `--few-shot-tier` | `10` | `train_seed/tier_10`; use `--no-few-shot` to disable |
| `--pool-size` | `5000` | Embedding pool cap |
| `--seed` | `42` | Shuffle + few-shot |

**Suggested `prediction-size` / `n-centroids-per-label` by label count** (logged at startup):

| Labels | `prediction-size` | `n-centroids-per-label` | Why |
|--------|-------------------|-------------------------|-----|
| 2 (irony, hate) | 400 | 10 | 400 ÷ (10×2) = 20 samples/centroid |
| 5 (pubmed) | 500 | 20 | 500 ÷ 100 centroids = 5 |
| 10 (banking) | 400 | 10 | 500 fails divisibility with 200 centroids |

```powershell
# PubMed (defaults work as-is)
uv run python scripts/run_huge_prediction_representatives.py `
  --dataset pubmed_20k_rct

# Binary dataset — override grid
uv run python scripts/run_huge_prediction_representatives.py `
  --dataset implicit_hate `
  --prediction-size 400 `
  --n-centroids-per-label 10

# Banking (10 classes)
uv run python scripts/run_huge_prediction_representatives.py `
  --dataset banking-10 `
  --prediction-size 400 `
  --n-centroids-per-label 10
```

**Outputs:** `predictions.parquet`, `contrastive_edges.parquet`, `manifest.json`, …

**Considerations:**

- Cost: ~`prediction_size / batch_size` API batches (e.g. 50 batches × concurrency).
- Wrong divisibility → clear error with suggested params.
- Labels in prompts use **card ids** (`ctx.label_ids`), not display names only.

---

## Step 2 — Build GEPA train/val

**Script:** `scripts/build_gepa_optimizer_sets.py`

**Role:** Hard errors + contrast partners from edges → `gepa_train.parquet` / `gepa_val.parquet` (disjoint from test tiers 20/200).

| Param | Default |
|-------|---------|
| `--train-total` | 50 |
| `--val-total` | 70 |
| `--train-easy-frac` | 0.2 |
| `--seed` | 42 |

```powershell
uv run python scripts/build_gepa_optimizer_sets.py `
  --dataset implicit_hate `
  --huge-prediction-dir data/huge_prediction_representatives/implicit_hate/20260520_120000
```

Check `manifest.json` for `gepa_train_rows`, `gepa_val_rows`, `contrast_rows_enriched_from_edges`.

---

## Step 3 — MIPRO optimize

**Script:** `scripts/run_gepa_mipro_optimize.py`

**Role:** DSPy MIPROv2 on batch metric (per-sentence accuracy within batch). Saves `compiled_program/`.

| Param | Default |
|-------|---------|
| `--executor-model` | `gemma-4-31b-it` |
| `--reflector-model` | `gemini-3.1-flash-lite-preview` |
| `--batch-size` | 5 (GEPA batches; independent of huge-pred 10) |
| `--num-candidates` | 20 |
| `--num-shots` | 0 (`max_labeled_demos`) |
| `--max-bootstrapped-demos` | 3 |
| `--minibatch-size` | 4 |

```powershell
uv run python scripts/run_gepa_mipro_optimize.py `
  --dataset implicit_hate `
  --gepa-sets-dir data/gepa_optimizer_sets/implicit_hate/20260520_121500
```

**Outputs:** `results/gepa_mipro/<dataset>/optimize/<stamp>/compiled_program/`, `val_eval_macro_f1.json`

---

## Step 4 — Evaluate compiled program (holdout)

**Script:** `scripts/eval_gepa_compiled_program.py`

**Required:** `--dataset` must match the program’s training dataset (label space + canonicalization).

```powershell
# Pass compiled_program/ (directory), not program.pkl alone
uv run python scripts/eval_gepa_compiled_program.py `
  --dataset banking-10 `
  --program-dir results/gepa_mipro/banking-10/optimize/<stamp>/compiled_program

# Default: tier_10 eval path (falls back to tier_20) capped to 10 rows
# Full test_200: --eval-tier 200 --eval-max-rows 0
```

---

## Cross-dataset checklist

- [ ] `tier_5000` train parquet exists  
- [ ] `prediction_size % batch_size == 0`  
- [ ] `prediction_size % (n_centroids_per_label × n_labels) == 0`  
- [ ] Huge-pred `manifest.json` `overall_accuracy` sane before GEPA build  
- [ ] GEPA manifest row counts ≈ `--train-total` + `--val-total` (plus contrast enrichment)  
- [ ] Same `--dataset` string through steps 2–4  
- [ ] Eval uses **test** parquet, not train tiers used for mining  

## Cost / time tips

- Reduce `--prediction-size` or `--pool-size` for smoke tests.  
- `--max-embedding-pool-rows` caps embedding work.  
- `--prelim-only` on MIPRO runs one val batch forward.  
- Lower `--num-candidates` for cheaper optimize passes.
