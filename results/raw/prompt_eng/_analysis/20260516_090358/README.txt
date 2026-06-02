Artifacts:
- rows_long.csv: one row per (run, index) with gold, pred, correct, confidence, source.
- summary_by_run.csv: accuracy and paths.
- wide_aligned_by_i.csv: merged preds where gold sequences match across runs (same indices).
- wide_with_disagreement.csv: adds n_distinct_preds when multiple slugs aligned.
- indices_where_runs_disagree.csv: rows where models picked different labels.
- confusion__*.png: confusion matrices.
- reliability__*.png: primary confidence vs correct (per sample, no binning).
- reliability__*__<metric>.png: alternate x-axes when all-label softmax fields exist.
- conf_box__*.png / conf_box__*__<metric>.png: same metrics by wrong/right.
- pairwise_disagreement.png: fraction of indices where preds differ (aligned runs only).
- accuracy_by_slug.png: quick ranking.

Primary confidence (column `confidence`):
  - DiNCo all-label runs: softmax_max_prob (winner mass after softmax over 5 labels).
  - Other runs: heuristic per schema (model JSON %, VCavg, CoVe coverage, top2 prob, etc.).

DiNCo all-label extra columns in rows_long.csv:
  - raw_scores / raw_confidence_pred: independent 0–1 scores per label (not a distribution).
  - softmax_probs / softmax_prob_pred: softmax(raw_scores); pred = argmax; prob_pred == max_prob.
  - softmax_margin_top1_top2: gap between top-2 softmax masses.
  - label_entropy_bits, label_entropy_normalized: spread of softmax (normalized = 1 − H/log2(5)).
  - reliability_combined: 0.5·softmax_max_prob + 0.5·label_entropy_normalized (legacy blend).

Runs with different EVAL_N or different gold rows are only partially merged; check summary n_rows.
