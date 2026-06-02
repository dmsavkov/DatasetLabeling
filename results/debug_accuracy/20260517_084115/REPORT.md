# debug_accuracy run

## H0_scoring_audit: Eval CSV correct column matches raw id==pred scoring
**verdict:** audit_ok
```json
{
  "run": "bs5_think_high",
  "report_accuracy": 0.765,
  "n": 200.0,
  "pred_null_rate": 0.0,
  "accuracy_raw_string": 0.765,
  "accuracy_canonical_names": 0.765,
  "accuracy_normalize_card_ids": 0.765,
  "accuracy_normalize_prompt_names": 0.0,
  "accuracy_reported_correct_col": 0.765
}
```

## H0_scoring_audit: Eval CSV correct column matches raw id==pred scoring
**verdict:** audit_ok
```json
{
  "run": "bs10_think_high",
  "report_accuracy": 0.83,
  "n": 200.0,
  "pred_null_rate": 0.0,
  "accuracy_raw_string": 0.83,
  "accuracy_canonical_names": 0.83,
  "accuracy_normalize_card_ids": 0.83,
  "accuracy_normalize_prompt_names": 0.0,
  "accuracy_reported_correct_col": 0.83
}
```

## H0_scoring_audit: Eval CSV correct column matches raw id==pred scoring
**verdict:** audit_ok
```json
{
  "run": "bs3_think_high",
  "report_accuracy": 0.775,
  "n": 200.0,
  "pred_null_rate": 0.0,
  "accuracy_raw_string": 0.775,
  "accuracy_canonical_names": 0.775,
  "accuracy_normalize_card_ids": 0.775,
  "accuracy_normalize_prompt_names": 0.0,
  "accuracy_reported_correct_col": 0.775
}
```

## H6_name_vs_id_scoring: On saved eval preds (numeric), canonical-name scoring delta
**verdict:** audit
```json
{
  "n": 200.0,
  "pred_null_rate": 0.0,
  "accuracy_raw_string": 0.83,
  "accuracy_canonical_names": 0.83,
  "accuracy_normalize_card_ids": 0.83,
  "accuracy_normalize_prompt_names": 0.0,
  "accuracy_reported_correct_col": 0.83,
  "delta_canon_minus_raw": 0.0
}
```

## H1_train_vs_test: Train pool differs from test tier (proxies); eval accuracy is test-only
**verdict:** inconclusive
```json
{
  "test200_mean_char_len": 150.555,
  "test200_mean_at_tokens": 1.78,
  "test200_mean_word_count": 26.355,
  "train5000_sample200_mean_char_len": 156.595,
  "train5000_sample200_mean_at_tokens": 1.895,
  "train5000_sample200_mean_word_count": 27.795,
  "eval_test200_accuracy_bs10": 0.83,
  "note": "True H1 needs live accuracy on matched train vs test samples; run --live."
}
```

## H4_convergence_80: Bootstrap mean accuracy on full test200 eval run
**verdict:** supported
```json
{
  "full_split_accuracy": 0.83,
  "bootstrap_mean": 0.8303875000000001,
  "bootstrap_std": 0.02651862258395031,
  "bootstrap_p5": 0.785,
  "bootstrap_p95": 0.875,
  "n_boot": 2000,
  "n_rows": 200
}
```

## H5_first_rows_harder: Accuracy by position bin in shuffled eval predictions.csv
**verdict:** supported
```json
{
  "by_bin": {
    "first_20": 0.6,
    "next_40": 0.9,
    "middle_40": 0.7,
    "next_40b": 0.75,
    "last_60": 0.7833333333333333
  },
  "first_20": 0.6,
  "rest_mean": 0.7833333333333333,
  "delta_first_vs_rest": -0.18333333333333335,
  "run": "bs5"
}
```

## H5_first_rows_harder: Accuracy by position bin in shuffled eval predictions.csv
**verdict:** supported
```json
{
  "by_bin": {
    "first_20": 0.7,
    "next_40": 0.925,
    "middle_40": 0.775,
    "next_40b": 0.85,
    "last_60": 0.8333333333333334
  },
  "first_20": 0.7,
  "rest_mean": 0.8444444444444444,
  "delta_first_vs_rest": -0.1444444444444445,
  "run": "bs10"
}
```
