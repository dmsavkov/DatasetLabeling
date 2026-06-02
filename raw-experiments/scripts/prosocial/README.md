# Prosocial Experiment Scripts

## Used Tech

- Python 3.13.5
- uv
- Ollama integrations use pure HTTP requests with ManualOllamaLM (no Ollama SDK)

## What lives here

- `runner.py`: main orchestration for the v3 matrix.
- `constants.py`: shared experiment constants and run config.
- `prompting.py`: prompt/rubric optimization and prompt builders.
- `retrieval.py`: representative-set build, embeddings, and top-3 retrieval.
- `inference.py`: async prediction path, statement extraction, MoE routing.
- `reporting.py`: metrics summaries, markdown table, collapse workflow.

## Workflow (concise)

1. Load train/valid/test with full features.
2. Build representative clustered dataset and top-100 retrieval pool.
3. Use Flash to optimize prompts/rubric.
4. Run experiment matrix on fixed test-25 with Gemma/Flash variants.
5. Save per-run predictions + summaries, comparison table, final summary.

## Entrypoint

Use the compatibility entry script:

- `python scripts/run_prosocial_experiments_v3.py --prepare-only`
- `python scripts/run_prosocial_experiments_v3.py`

Optional fast prep smoke-run:

- `python scripts/run_prosocial_experiments_v3.py --prepare-only --prepare-max-rows 500`
