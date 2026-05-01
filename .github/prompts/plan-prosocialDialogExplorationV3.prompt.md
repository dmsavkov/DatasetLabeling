## Plan: Prosocial Dialog Exploration v3

Create one notebook-centered experiment harness around [prosocial-dialog-exploration.ipynb](prosocial-dialog-exploration.ipynb) that uses flash-lite for prompt synthesis and gemma for prediction by default, with no phi experiments anywhere. The notebook should load the full ProsocialDialog feature set early, build an enriched representative sample dataset for later filtering/retrieval, and run a complete comparison matrix on the same frozen shared test25 from the earlier runs.

**Steps**
1. Establish the shared evaluation anchor first: load the bundle, freeze the exact prior test25 indices, and surface the row metadata plus label counts before any new experiment runs. Keep the shared split constant across every variant so all comparisons are apples-to-apples.
2. Extend the early data-loading path to keep all features, not just context and final label. The notebook should preserve safety_annotations and safety_annotation_reasons so they can be used both for filtering and for prompt construction.
3. Add two metric helpers up front: the first is the adjusted-distance metric using the raw absolute label distance on the original 5-level scale; the second is the 3-class collapse mapping that merges possibly_needs_caution, probably_needs_caution, and needs_caution into needs_caution for the separate collapse experiment.
4. Build a full-feature artifact writer for the new notebook so every major dataset or experiment snapshot can be saved to results with a timestamped summary and a machine-readable table of used indices, prompt settings, and metrics.
5. Run the first flash-plus-gemma experiment using the enriched full-feature training pool, with safety_annotation_reasons included explicitly in the prompt-optimization input. This becomes the early baseline for all later prompt variants.
6. Add the representative sample-selection pipeline next. Filter to rows where all three safety annotations agree, stratify by final safety_label, embed each of the 5 strata with all-MiniLM-L6-v2 through fastembed, and cluster each stratum with its own KMeans object. Use adaptive k=min(20, stratum_size) so sparse strata do not fail, and record the actual k used in metadata.
7. Persist the selected representative dataset to results with all original columns plus derived features such as unanimous-agreement flags, stratum, embedding references or vectors, cluster id, cluster centroid distance, and representative-sample markers. Derive a top100 pool from this artifact for later retrieval experiments.
8. Use that representative dataset to drive the flash-plus-gemma prompt optimization variant that relies on a small representative few-shot set rather than a random sample. This replaces the earlier phi-based paths entirely.
9. Restore the missing prompt-variant sweep from the first version, but make flash-plus-gemma the default for every variant unless a step explicitly says otherwise. Include at minimum: more few-shot samples, calibrated rubric, optimized rubric, assertion-hardened prompt, and the batch-size-5 inference variant.
10. Implement the batch-5 experiment as a client-side chunking mode over gemma prediction calls, not as a model change. Keep the prompt and examples identical to the comparable flash-plus-gemma baseline so batching is the only variable.
11. Add the dynamic retrieval experiment: embed each test context, fetch the top 3 nearest examples from the top100 representative pool, and use those examples as the few-shot context for optimized prompt plus gemma prediction. Save the retrieved example ids per row for analysis.
12. Add the MoE experiment back, but on Gemma rather than Phi. Use flash as the router or prompt planner, and route to three Gemma-family experts available in the environment. If fewer than three Gemma-family checkpoints are available, treat that as an environment blocker and record it explicitly rather than silently swapping in another model family.
13. Add the label-collapse experiment as its own evaluation branch. Collapse possibly_needs_caution, probably_needs_caution, and needs_caution into needs_caution, then score the same fixed test25 on the collapsed label space alongside the original 5-level runs.
14. Score every non-collapsed experiment with the raw adjusted-distance metric in addition to accuracy, macro F1, weighted F1, and parse-error count. Keep the collapsed experiment separate so the distance metric remains on the original 5-level ordering.
15. Finish with one comparison cell that ranks all runs, writes prediction CSVs and summary JSON files, and regenerates a comparison table that includes the new flash-plus-gemma variants, the batch-5 variant, the retrieval variant, the rubric/assertion variants, the MoE variant, the collapse run, and the adjusted-distance metric.

**Relevant files**
- [prosocial-dialog-exploration.ipynb](prosocial-dialog-exploration.ipynb) — main orchestration notebook for all experiment variants.
- [src/data.py](src/data.py) — needs full-feature loading support and the new adjusted-distance helper.
- [src/dataset_scripts.py](src/dataset_scripts.py) — reusable split helper for the frozen shared test25 and later representative-pool slicing.
- [scripts/run_google_prompt_transfer.py](scripts/run_google_prompt_transfer.py) — reference for flash-lite prompt optimization and gemma-style JSON output contracts.
- [mvp.ipynb](mvp.ipynb) — structural reference for notebook cell order, result summaries, and earlier baseline reporting.
- [data/prosocial-dialog/README (2).md](data/prosocial-dialog/README%20(2).md) — authoritative dataset schema for the extra annotation fields.
- [pyproject.toml](pyproject.toml) — dependency gap check; fastembed, sklearn clustering, and any notebook-only packages need to be validated against the current environment.
- [scripts/run_dspy.py](scripts/run_dspy.py) — keep only as a legacy reference to remove phi-specific assumptions, not as a target workflow.

**Verification**
1. Run the notebook end to end on the frozen shared test25 and verify every variant sees the same source_index set.
2. Confirm the full-feature dataset artifact contains safety_annotations and safety_annotation_reasons and that the representative artifact contains all derived clustering fields.
3. Validate the clustering step by checking that each populated stratum has a recorded KMeans object, a cluster assignment for each row, and no k greater than the stratum size.
4. Validate the batch-5 variant by confirming the chunking logic processes test rows in groups of five while still emitting one prediction per row.
5. Validate the adjusted-distance metric with a hand-checked mapping table and confirm the metric is reported only for the original 5-level runs.
6. Confirm the 3-class collapse run uses the merged label map and does not leak the collapsed metric into the 5-level runs.
7. Check the final artifact set for the comparison table, timestamped CSVs, and summary JSON files, and confirm there are no remaining phi references in the new notebook flow.

**Decisions**
- Default model stack is flash-lite for prompt work and gemma for prediction; no phi experiments remain in this plan.
- The batch-5 experiment is a scheduling/batching variant, not a new model or prompt family.
- If a unanimous stratum is sparse, KMeans k is clamped to the available row count so the selection pipeline remains runnable.
- The MoE variant should use three Gemma-family experts if the environment has them; otherwise the notebook should log the blocker and skip that branch rather than falling back to phi.
- The representative selection artifact is the source of truth for later retrieval experiments, including the top100 candidate pool.

**Further Considerations**
- The exact three Gemma experts for MoE should be chosen from what is actually installed at runtime; if only one Gemma checkpoint exists, the notebook should clearly report that the MoE branch is unavailable.
- The rubric and assertion variants should stay separate in the experiment table even if they share much of the same prompt text, so later analysis can attribute gains correctly.
