from __future__ import annotations

import argparse
import os
from time import perf_counter

import dotenv
import pandas as pd
from openai import AsyncOpenAI, OpenAI

from prosocial.constants import (
    BASE_URL,
    CALIBRATED_RUBRIC,
    FIXED_TEST25_INDICES,
    FLASH_MODEL,
    GEMMA_MODEL,
    GEMMA_MOE_EXPERTS,
    MAX_CONCURRENCY,
    MAX_RETRIES,
    SEED,
    STRICT_ASSERTIONS,
    ExperimentRun,
)
from prosocial.prompting import build_optimizer_examples, optimize_prompt_with_flash, optimize_rubric_with_flash
from prosocial.reporting import (
    get_available_model_ids,
    run_collapse_experiment,
    run_model_experiment,
    write_comparison_markdown,
)
from prosocial.retrieval import build_representative_dataset, build_retrieval_map
from src.data import now_stamp, save_json
from src.dataset_scripts import load_prosocial_dialog_bundle, select_rows_by_source_index


dotenv.load_dotenv()


def build_runs(
    *,
    baseline_prompt: str,
    representative_prompt: str,
    optimized_prompt_with_optimized_rubric: str,
    moe_experts: list[str],
) -> list[ExperimentRun]:
    runs = [
        ExperimentRun(
            name="exp_full_features_reasons_flash_gemma",
            optimized_prompt=baseline_prompt,
            prediction_model=GEMMA_MODEL,
            assertion_text=STRICT_ASSERTIONS,
            batch_size=1,
        ),
        ExperimentRun(
            name="exp_representative_prompt_flash_gemma",
            optimized_prompt=representative_prompt,
            prediction_model=GEMMA_MODEL,
            assertion_text=STRICT_ASSERTIONS,
            batch_size=1,
        ),
        # ExperimentRun(
        #     name="exp_more_fewshot_flash_gemma",
        #     optimized_prompt=baseline_prompt,
        #     prediction_model=GEMMA_MODEL,
        #     assertion_text=STRICT_ASSERTIONS,
        #     batch_size=1,
        # ),
        # ExperimentRun(
        #     name="exp_calibrated_rubric_flash_gemma",
        #     optimized_prompt=baseline_prompt,
        #     prediction_model=GEMMA_MODEL,
        #     assertion_text=STRICT_ASSERTIONS,
        #     batch_size=1,
        # ),
        # ExperimentRun(
        #     name="exp_optimized_rubric_flash_gemma",
        #     optimized_prompt=optimized_prompt_with_optimized_rubric,
        #     prediction_model=GEMMA_MODEL,
        #     assertion_text=STRICT_ASSERTIONS,
        #     batch_size=1,
        # ),
        # ExperimentRun(
        #     name="exp_assertion_hardened_flash_gemma",
        #     optimized_prompt=baseline_prompt,
        #     prediction_model=GEMMA_MODEL,
        #     assertion_text=STRICT_ASSERTIONS + "\n- Reject non-JSON output as invalid.",
        #     batch_size=1,
        # ),
        # ExperimentRun(
        #     name="exp_batch5_flash_gemma",
        #     optimized_prompt=baseline_prompt,
        #     prediction_model=GEMMA_MODEL,
        #     assertion_text=STRICT_ASSERTIONS,
        #     batch_size=5,
        # ),
        # ExperimentRun(
        #     name="exp_two_call_extract_statements_flash_gemma",
        #     optimized_prompt=baseline_prompt,
        #     prediction_model=GEMMA_MODEL,
        #     assertion_text=STRICT_ASSERTIONS,
        #     enable_statement_extraction=True,
        #     batch_size=1,
        # ),
        # ExperimentRun(
        #     name="exp_dynamic_top3_retrieval_flash_gemma",
        #     optimized_prompt=baseline_prompt,
        #     prediction_model=GEMMA_MODEL,
        #     assertion_text=STRICT_ASSERTIONS,
        #     include_dynamic_retrieval=True,
        #     batch_size=1,
        # ),
        # ExperimentRun(
        #     name="exp_flash_direct_predictions",
        #     optimized_prompt=baseline_prompt,
        #     prediction_model=FLASH_MODEL,
        #     assertion_text=STRICT_ASSERTIONS,
        #     batch_size=1,
        # ),
    ]

    if len(moe_experts) >= 3:
        runs.append(
            ExperimentRun(
                name="exp_moe_flash_gemma_experts",
                optimized_prompt=baseline_prompt,
                prediction_model=GEMMA_MODEL,
                assertion_text=STRICT_ASSERTIONS,
                batch_size=1,
                moe_experts=moe_experts,
            )
        )

    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prosocial Dialog experiment matrix v3.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--prepare-max-rows",
        type=int,
        default=0,
        help="Optional cap for train+valid rows during preparation (0 = full set).",
    )
    args = parser.parse_args()

    bundle = load_prosocial_dialog_bundle(include_all_features=True)
    train_df = bundle["train_df"]
    valid_df = bundle["valid_df"]
    test_df = bundle["test_df"]
    labels = bundle["label_order"]

    results_dir = bundle["results_dir"] / "prosocial_v3"
    results_dir.mkdir(parents=True, exist_ok=True)

    test25_df = select_rows_by_source_index(test_df, FIXED_TEST25_INDICES)
    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    if args.prepare_max_rows > 0:
        train_valid_df = train_valid_df.head(int(args.prepare_max_rows)).reset_index(drop=True)

    representative_df, top100_df, representative_meta = build_representative_dataset(
        pool_df=train_valid_df,
        labels=labels,
        results_dir=results_dir,
        seed=args.seed,
    )

    print(
        {
            "representative_total": len(representative_df),
            "top100_total": len(top100_df),
            "fixed_test25_total": len(test25_df),
            "results_dir": str(results_dir),
        }
    )

    if args.prepare_only:
        print({"status": "prepared_only", "meta": representative_meta})
        return

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY before running model experiments")

    sync_client = OpenAI(api_key=api_key, base_url=BASE_URL, max_retries=MAX_RETRIES)
    async_client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL, max_retries=MAX_RETRIES)

    available_models = get_available_model_ids(sync_client)
    baseline_examples = build_optimizer_examples(
        train_valid_df,
        count=50,
        labels=labels,
        seed=args.seed,
        include_reasons=True,
    )

    t_prompt = perf_counter()
    baseline_prompt = optimize_prompt_with_flash(
        sync_client,
        examples=baseline_examples,
        labels=labels,
        rubric=CALIBRATED_RUBRIC,
        assertions=STRICT_ASSERTIONS,
        objective="maximize macro-F1 and reduce severe under-classification",
    )
    baseline_prompt_seconds = perf_counter() - t_prompt

    representative_examples = build_optimizer_examples(
        representative_df[representative_df["is_cluster_representative"]],
        count=50,
        labels=labels,
        seed=args.seed,
        include_reasons=True,
    )

    optimized_rubric = optimize_rubric_with_flash(
        sync_client,
        examples=baseline_examples,
        labels=labels,
        current_rubric=CALIBRATED_RUBRIC,
    )

    optimized_prompt_with_optimized_rubric = optimize_prompt_with_flash(
        sync_client,
        examples=baseline_examples,
        labels=labels,
        rubric=optimized_rubric,
        assertions=STRICT_ASSERTIONS,
        objective="improve boundary calibration across adjacent caution classes",
    )

    representative_prompt = optimize_prompt_with_flash(
        sync_client,
        examples=representative_examples,
        labels=labels,
        rubric=CALIBRATED_RUBRIC,
        assertions=STRICT_ASSERTIONS,
        objective="maximize performance using clustered representative exemplars",
    )

    fewshot_static = baseline_examples[:10]
    more_fewshot_static = baseline_examples[:20]
    retrieval_map = build_retrieval_map(top100_df=top100_df, test_df=test25_df)

    moe_experts = [m for m in GEMMA_MOE_EXPERTS if (not available_models or m in available_models)]
    if len(moe_experts) < 3:
        print(
            {
                "moe_status": "skipped",
                "reason": "fewer than 3 Gemma experts available",
                "available_models_detected": available_models[:20],
                "requested_experts": GEMMA_MOE_EXPERTS,
            }
        )

    runs = build_runs(
        baseline_prompt=baseline_prompt,
        representative_prompt=representative_prompt,
        optimized_prompt_with_optimized_rubric=optimized_prompt_with_optimized_rubric,
        moe_experts=moe_experts,
    )

    comparison_rows: list[dict[str, float | int | str]] = []
    base_results_for_collapse: pd.DataFrame | None = None
    base_summary_for_collapse: dict[str, str] | None = None

    for run in runs:
        static_fewshots = more_fewshot_static if run.name == "exp_more_fewshot_flash_gemma" else fewshot_static
        results_df, summary, pred_path, summary_path = run_model_experiment(
            run=run,
            test25_df=test25_df,
            labels=labels,
            async_client=async_client,
            results_dir=results_dir,
            max_concurrency=args.max_concurrency,
            retrieval_map=retrieval_map,
            static_fewshots=static_fewshots,
        )

        print(
            {
                "run": run.name,
                "summary": str(summary_path),
                "predictions": str(pred_path),
                "accuracy": summary["metrics"]["accuracy"],
                "macro_f1": summary["metrics"]["macro_f1"],
                "adjusted_distance": summary["metrics"]["adjusted_distance"],
            }
        )

        comparison_rows.append(
            {
                "workflow": run.name,
                "accuracy": float(summary["metrics"]["accuracy"]),
                "macro_f1": float(summary["metrics"]["macro_f1"]),
                "weighted_f1": float(summary["metrics"]["weighted_f1"]),
                "adjusted_distance": float(summary["metrics"]["adjusted_distance"]),
                "parse_error_count": int(summary["metrics"]["parse_error_count"]),
            }
        )

        if run.name == "exp_full_features_reasons_flash_gemma":
            base_results_for_collapse = results_df
            base_summary_for_collapse = {"workflow": run.name, "summary_path": str(summary_path)}

    if base_results_for_collapse is not None and base_summary_for_collapse is not None:
        collapse_summary = run_collapse_experiment(
            name="exp_label_collapse_3class",
            base_results_df=base_results_for_collapse,
            summary_base=base_summary_for_collapse,
            results_dir=results_dir,
        )
        print(
            {
                "run": "exp_label_collapse_3class",
                "accuracy": collapse_summary["metrics"]["accuracy"],
                "macro_f1": collapse_summary["metrics"]["macro_f1"],
            }
        )

    stamp = now_stamp()
    comparison_csv = results_dir / f"comparison_{stamp}.csv"
    comparison_md = results_dir / f"comparison_{stamp}.md"
    pd.DataFrame(comparison_rows).to_csv(comparison_csv, index=False)
    write_comparison_markdown(comparison_rows, comparison_md)

    final_summary = {
        "workflow": "prosocial_v3_matrix",
        "models": {
            "flash": FLASH_MODEL,
            "gemma_default": GEMMA_MODEL,
            "moe_experts": moe_experts,
        },
        "google_openai_base_url": BASE_URL,
        "timing_seconds": {"baseline_prompt_optimization": float(baseline_prompt_seconds)},
        "shared_test25_indices": FIXED_TEST25_INDICES,
        "comparison_artifacts": {
            "comparison_csv": str(comparison_csv),
            "comparison_md": str(comparison_md),
        },
        "representative_meta": representative_meta,
        "experiments_run": [row["workflow"] for row in comparison_rows],
    }
    final_summary_path = results_dir / f"prosocial_v3_summary_{stamp}.json"
    save_json(final_summary_path, final_summary)

    print(
        {
            "status": "completed",
            "comparison_csv": str(comparison_csv),
            "comparison_md": str(comparison_md),
            "final_summary": str(final_summary_path),
        }
    )


if __name__ == "__main__":
    main()
