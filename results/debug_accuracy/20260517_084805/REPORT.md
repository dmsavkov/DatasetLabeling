# debug_accuracy run

## Live reproduction (API)
```json
{
  "configs": [
    {
      "config": "exact_bs10_baseline",
      "n_compared": 200,
      "live_accuracy": 0.83,
      "saved_accuracy_on_same_ids": 0.83,
      "pred_label_agreement_with_saved": 1.0,
      "both_correct_rate": 0.83,
      "live_pred_null_rate": 0.0,
      "first_20_live_accuracy": 0.7,
      "after_first_20_live_accuracy": 0.8444444444444444,
      "mismatched_pred_count": 0
    },
    {
      "config": "ablation_no_thinking",
      "n_compared": 200,
      "live_accuracy": 0.83,
      "saved_accuracy_on_same_ids": 0.83,
      "pred_label_agreement_with_saved": 1.0,
      "both_correct_rate": 0.83,
      "live_pred_null_rate": 0.0,
      "first_20_live_accuracy": 0.7,
      "after_first_20_live_accuracy": 0.8444444444444444,
      "mismatched_pred_count": 0
    },
    {
      "config": "ablation_no_few_shot",
      "n_compared": 200,
      "live_accuracy": 0.365,
      "saved_accuracy_on_same_ids": 0.83,
      "pred_label_agreement_with_saved": 0.365,
      "both_correct_rate": 0.335,
      "live_pred_null_rate": 0.0,
      "first_20_live_accuracy": 0.25,
      "after_first_20_live_accuracy": 0.37777777777777777,
      "mismatched_pred_count": 127
    },
    {
      "config": "ablation_prompt_names",
      "n_compared": 200,
      "live_accuracy": 0.0,
      "saved_accuracy_on_same_ids": 0.83,
      "pred_label_agreement_with_saved": 0.0,
      "both_correct_rate": 0.0,
      "live_pred_null_rate": 0.0,
      "first_20_live_accuracy": 0.0,
      "after_first_20_live_accuracy": 0.0,
      "mismatched_pred_count": 200
    },
    {
      "config": "ablation_bs5",
      "n_compared": 200,
      "live_accuracy": 0.765,
      "saved_accuracy_on_same_ids": 0.83,
      "pred_label_agreement_with_saved": 0.85,
      "both_correct_rate": 0.725,
      "live_pred_null_rate": 0.0,
      "first_20_live_accuracy": 0.6,
      "after_first_20_live_accuracy": 0.7833333333333333,
      "mismatched_pred_count": 30
    },
    {
      "config": "ablation_thinking_off_no_fewshot",
      "n_compared": 200,
      "live_accuracy": 0.365,
      "saved_accuracy_on_same_ids": 0.83,
      "pred_label_agreement_with_saved": 0.365,
      "both_correct_rate": 0.335,
      "live_pred_null_rate": 0.0,
      "first_20_live_accuracy": 0.25,
      "after_first_20_live_accuracy": 0.37777777777777777,
      "mismatched_pred_count": 127
    }
  ],
  "total_api_batch_calls": 161,
  "reproduces_saved_83pct": true,
  "pred_labels_match_saved_run": 1.0,
  "hypotheses_live": {
    "H1_train_vs_test": {
      "test": {
        "split": "test200",
        "n": 15,
        "accuracy_raw_id": 0.7333333333333333
      },
      "train_sample": {
        "split": "train5000",
        "n": 15,
        "accuracy_raw_id": 0.7333333333333333
      },
      "train_harder_if_lower_accuracy": false
    },
    "H2_few_shot": {
      "n": 30,
      "accuracy_with_few_shot": 0.7666666666666667,
      "accuracy_without_few_shot": 0.3333333333333333,
      "few_shot_helps": true,
      "delta": 0.4333333333333334
    },
    "H3_batch_vs_single": {
      "n": 10,
      "accuracy_one_batch": 0.8,
      "accuracy_n_singleton_calls": 0.6,
      "preds_equal": false,
      "batch_hurts_if_lower": false
    }
  }
}
```

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

## H7_adhoc_vs_pipeline: Ad-hoc GenAI scripts are not equivalent to evaluate_google_llm pipeline
**verdict:** audit
```json
{
  "pipeline_eval": {
    "prompt": "baseline_v1 batch [{id, label}, ...]",
    "allowed_labels": "card numeric ids 0-4",
    "few_shot": "tier_10 after shuffle (train rows)",
    "thinking": "high for gemma4_31b_think_high",
    "data": "test/tier_200",
    "scoring": "str(true_label) == str(pred_label)"
  },
  "request_google_py": {
    "prompt": "custom single-sentence JSON [{label}]",
    "allowed_labels": "only labels present in 5 pool rows",
    "few_shot": "none",
    "thinking": "default off",
    "batching": "multi-prompt array; parser unlike parse_batch_predictions"
  },
  "t_py_label_mapping": "BACKGROUND/METHODS uppercase strings != eval numeric ids",
  "scoring_trap": "If you compare pred '2' to true_label_str 'METHODS' you always fail. If you score numeric preds with prompt-name allowed_labels, accuracy_normalize_prompt_names=0 (see H6)."
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
