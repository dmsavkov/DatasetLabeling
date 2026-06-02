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

## H8_huge_train_pool
**inconclusive** — Huge prediction train pool low accuracy — config audit
```json
{
  "error": "missing D:\\Coding\\VSCFiles\\IndependentProjects\\AI\\uni-judge-research\\data\\huge_prediction_representatives\\pubmed_20k_rct\\20260516_221123\\manifest.json",
  "fix": "use label_ids + eval-aligned few-shot (patched)"
}
```
