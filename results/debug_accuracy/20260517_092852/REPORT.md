# debug_accuracy (minimal)

**API batch calls:** 10

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

## H8_huge_train_pool
**inconclusive** — Huge prediction train pool low accuracy — config audit
```json
{
  "error": "missing D:\\Coding\\VSCFiles\\IndependentProjects\\AI\\uni-judge-research\\data\\huge_prediction_representatives\\pubmed_20k_rct\\20260516_221123\\manifest.json",
  "fix": "use label_ids + eval-aligned few-shot (patched)"
}
```

## H1_train_vs_test
**inconclusive** — Train pool harder than test on same eval prompt (live probes)
```json
{
  "test_accuracy": 0.6666666666666666,
  "train_accuracy": 0.8,
  "train_harder": false
}
```

## H2_few_shot
**supported** — Few-shot helps on test probes
```json
{
  "with": 0.6666666666666666,
  "without": 0.16666666666666666,
  "helps": true
}
```

## H3_batch_vs_single
**inconclusive** — Batch vs singleton API calls (3 probes)
```json
{
  "n": 3,
  "accuracy_batch": 0.6666666666666666,
  "accuracy_single_calls": 0.6666666666666666,
  "preds_equal": true
}
```

## H8_huge_misconfig
**supported** — Old huge_prediction settings hurt train-pool accuracy vs eval settings
```json
{
  "H8_old_huge_train_acc": 0.0,
  "H8_eval_on_train_acc": 0.8,
  "H8_delta_eval_minus_old": 0.8,
  "runs": {
    "H8_eval_on_train": {
      "config": "eval_baseline",
      "n": 5,
      "accuracy": 0.8,
      "preds": [
        {
          "sample_id": "train_7290",
          "gold": "0",
          "pred": "0"
        },
        {
          "sample_id": "train_137301",
          "gold": "3",
          "pred": "0"
        },
        {
          "sample_id": "train_159814",
          "gold": "4",
          "pred": "4"
        },
        {
          "sample_id": "train_156638",
          "gold": "1",
          "pred": "1"
        },
        {
          "sample_id": "train_95701",
          "gold": "2",
          "pred": "2"
        }
      ]
    },
    "H8_huge_old_on_train": {
      "config": "huge_old",
      "n": 5,
      "accuracy": 0.0,
      "preds": [
        {
          "sample_id": "train_7290",
          "gold": "0",
          "pred": "objective"
        },
        {
          "sample_id": "train_137301",
          "gold": "3",
          "pred": "objective"
        },
        {
          "sample_id": "train_159814",
          "gold": "4",
          "pred": "results"
        },
        {
          "sample_id": "train_156638",
          "gold": "1",
          "pred": "conclusions"
        },
        {
          "sample_id": "train_95701",
          "gold": "2",
          "pred": "methods"
        }
      ]
    }
  }
}
```

## H_repro_live_probes
**supported** — Live eval baseline on hardcoded test probes
```json
{
  "accuracy": 0.6666666666666666,
  "saved_agreement": 1.0,
  "api_batch_calls": 10
}
```
