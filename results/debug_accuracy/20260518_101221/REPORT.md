# debug_accuracy (minimal)

## H0_scoring
**supported** — Saved eval CSV uses raw id==pred scoring
```json
{
  "report_accuracy": 0.83,
  "n": 200.0,
  "accuracy_raw_string": 0.83,
  "accuracy_canonical_names": 0.83,
  "accuracy_normalize_card_ids": 0.83,
  "accuracy_normalize_prompt_names": 0.0
}
```

## H_repro_saved
**supported** — Hardcoded test probes vs saved bs10 run (offline)
```json
{
  "n": 6,
  "accuracy_on_probes": 0.6666666666666666,
  "rows": [
    {
      "sample_id": "test_10035",
      "gold": "1",
      "pred_saved": "1",
      "correct_saved": true
    },
    {
      "sample_id": "test_20044",
      "gold": "2",
      "pred_saved": "2",
      "correct_saved": true
    },
    {
      "sample_id": "test_13953",
      "gold": "3",
      "pred_saved": "0",
      "correct_saved": false
    },
    {
      "sample_id": "test_14870",
      "gold": "4",
      "pred_saved": "1",
      "correct_saved": false
    },
    {
      "sample_id": "test_20520",
      "gold": "0",
      "pred_saved": "0",
      "correct_saved": true
    },
    {
      "sample_id": "test_10646",
      "gold": "1",
      "pred_saved": "1",
      "correct_saved": true
    }
  ]
}
```

## H4_convergence_83
**supported** — Full test200 eval accuracy ~83% (offline saved run)
```json
{
  "full_accuracy": 0.83,
  "bootstrap_mean": 0.82991,
  "bootstrap_p5": 0.785,
  "bootstrap_p95": 0.875
}
```

## H5_first_rows
**supported** — First 20 shuffled rows harder than rest (offline)
```json
{
  "first_20": 0.7,
  "rest": 0.8444444444444444,
  "delta": -0.1444444444444445
}
```

## H6_name_scoring_trap
**supported** — Scoring numeric preds with name allow-list gives ~0%
```json
{
  "n": 200.0,
  "accuracy_raw_string": 0.83,
  "accuracy_canonical_names": 0.83,
  "accuracy_normalize_card_ids": 0.83,
  "accuracy_normalize_prompt_names": 0.0
}
```

## H7_adhoc_scripts
**supported** — request_google.py / t.py are not eval-equivalent
```json
{
  "eval": "baseline_v1, card ids 0-4, 10-shot shuffle seed, thinking high, test/tier_200",
  "adhoc": "custom JSON, partial labels, no few-shot, wrong pool, no thinking"
}
```

## H1_different_samples
**supported** — Huge pool and test200 are disjoint sample sets
```json
{
  "pool_vs_test200_overlap": 0,
  "shared_sample_ids": 0,
  "huge_overall": 0.67,
  "huge_reweighted_to_test_label_dist": 0.75175,
  "huge_per_label": {
    "0": 0.48,
    "1": 0.67,
    "2": 0.99,
    "3": 0.52,
    "4": 0.69
  },
  "eval_per_label": {
    "0": 0.7142857142857143,
    "1": 0.8709677419354839,
    "2": 0.9402985074626866,
    "3": 0.875,
    "4": 0.7230769230769231
  },
  "few_shot_equal_eval_vs_huge": true,
  "few_shot_n_eval": 10,
  "few_shot_n_huge": 10
}
```

## H2_few_shot_parity
**supported** — Eval and huge use identical eval-aligned few-shot
```json
{
  "few_shot_equal_eval_vs_huge": true,
  "few_shot_n_eval": 10,
  "few_shot_n_huge": 10
}
```

## H5_label_mix_partial
**supported** — Balanced huge label mix explains part of overall gap
```json
{
  "huge_overall": 0.67,
  "huge_reweighted_to_test_label_dist": 0.75175,
  "huge_per_label": {
    "0": 0.48,
    "1": 0.67,
    "2": 0.99,
    "3": 0.52,
    "4": 0.69
  },
  "eval_per_label": {
    "0": 0.7142857142857143,
    "1": 0.8709677419354839,
    "2": 0.9402985074626866,
    "3": 0.875,
    "4": 0.7230769230769231
  }
}
```

## H9_scoring_format
**inconclusive** — Canonical label compare fixes 4 vs 4.0 and id vs name
```json
{
  "raw_string_accuracy": 0.83,
  "canonical_accuracy": 0.83,
  "saved_correct_column_accuracy": 0.83,
  "rows_fixed_by_canonical": 0
}
```

## H8_huge_train_pool
**inconclusive** — Huge prediction on train pool was misconfigured (not just hard data)
```json
{
  "manifest_overall_accuracy": 0.67,
  "manifest_allowed_labels": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ],
  "manifest_few_shot_style": "eval_aligned_card_ids",
  "bug_before_fix": "allowed_labels=prompt_names + name few-shot + numeric preds dropped",
  "fix_applied": "label_ids + load_eval_aligned_few_shot + store card id preds",
  "parquet_rows": 500,
  "parquet_accuracy": 0.67
}
```
