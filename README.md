## v1 foundation: datasets + eval harness

This repo is in the “make experiments reproducible” phase: deterministic benchmark datasets, deterministic tiered test splits, and a model-agnostic evaluation harness.

### Directory conventions (authoritative)

- **`src/`**: reusable Python modules (no notebook imports).
  - **`src/datasets/`**: universal processed schema + deterministic split utilities + (later) dataset builders and IO.
  - **`src/models/`**: minimal predictor interface + adapters.
  - **`src/eval/`**: evaluation harness + metrics + artifact writers.
- **`data/processed/`**: the single source of truth for processed datasets and persisted splits/manifests.
  - Layout target: `data/processed/<dataset_name>/<split_name>/<tier_size>/...`
- **`notebooks/`** and **`raw-experiments/`**: behavioral reference only (not imported by `src/`).

### v1 “contracts” (kept small and stable)

- **Universal processed samples schema**: `src/datasets/schema.py`
- **Model predictor interface + prediction payload**: `src/models/interfaces.py`
- **Eval artifact naming (report/predictions)**: `src/eval/artifacts.py`
