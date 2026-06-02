Artifacts:
- rows_long.csv: one row per (run, index) with gold, pred, correct, confidence, source.
- summary_by_run.csv: accuracy and paths.
- wide_aligned_by_i.csv: merged preds where gold sequences match across runs (same indices).
- wide_with_disagreement.csv: adds n_distinct_preds when multiple slugs aligned.
- indices_where_runs_disagree.csv: rows where models picked different labels.
- confusion__*.png: confusion matrices.
- reliability__*.png: per-sample raw confidence vs correct (no binning).
- conf_box__*.png: raw confidence by outcome (points + box, no binning).
- pairwise_disagreement.png: fraction of indices where preds differ (aligned runs only).
- accuracy_by_slug.png: quick ranking.

Confidence mapping is heuristic per schema (model JSON, VCavg, CoVe coverage, DiNCo norm score, top2 prob).
Runs with different EVAL_N or different gold rows are only partially merged; check summary n_rows.
