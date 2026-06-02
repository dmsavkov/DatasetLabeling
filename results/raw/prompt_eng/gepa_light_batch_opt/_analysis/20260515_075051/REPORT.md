# GEPA light batch — post-hoc inspection report

## Experiment (from saved metadata)
- **GEPA run dir:** `D:\Coding\VSCFiles\IndependentProjects\AI\uni-judge-research\results\raw\prompt_eng\gepa_light_batch_opt\20260514_111314`
- **Duration (compile+eval):** 4246.683936399873 s
- **Executor:** gemma-4-31b-it
- **Reflector:** gemini-3.1-flash-lite-preview
- **Train / eval sentences:** 50 / 50 (batches of 5)
- **GEPA:** `auto=light`, `reflection_minibatch_size=2`, `skip_perfect_score=True`, `candidate_selection_strategy=pareto`
- **Saved accuracy (GEPA eval):** 0.74

## What we cannot recover
- **`optimized_program.dump_state()`** was not saved — final mutated instructions / selected demos are **not** reconstructable from disk.
- **DSPy cache** (`~/.dspy_cache/*/cache.db`) stores **pickled LiteLLM `ModelResponse`** objects (LLM completions), keyed by request hash — **not** the compiled program graph.

## Baseline comparator
- Predictions: `D:\Coding\VSCFiles\IndependentProjects\AI\uni-judge-research\results\raw\prompt_eng\bootstrap_batch_cot\20260512_125452\full_predictions.json`
- On this eval slice (seed=42), **gold labels match 50/50** with GEPA predictions file when both use the same protocol.
- **Accuracy GEPA:** 0.740 | **Comparator:** 0.720 | **Δ:** +0.020

## Delta cohorts (sentence-level, n=50)
- **Both correct:** 33
- **Fixed (baseline wrong → GEPA right):** 4
- **Regressions (baseline right → GEPA wrong):** 3
- **Stubborn failures (both wrong):** 10

## DSPy cache scan
- Folders scanned: 16
- Batch-like response samples exported: 40 (see `dspy_cache_batch_like_samples.json`)

## Artifacts in this folder
- `train_batches_explicit.json` — all **training** batches with full sentence text + gold labels
- `eval_batches_explicit.json` — **eval** batches (same rows GEPA scored)
- `train_sentences_flat.csv` / `eval_sentences_flat.csv` — one row per sentence
- `delta_all.csv`, `delta_fixed.csv`, `delta_regressions.csv`, `delta_stubborn.csv`
- `cohort_counts.png`, `per_class_accuracy_gepa_vs_baseline.png`

## Playbook next steps
1. Read **Fixed** rows in `delta_fixed.csv` — what label boundary changed?
2. Read **Regressions** — overfit rules / catastrophic forgetting?
3. Read **Stubborn** — ambiguous text or bad gold?
4. Re-run GEPA with `gepa_light_batch_opt.py` after adding `optimized_program.save(...)` (see updated script).