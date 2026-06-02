# pyright: basic

from __future__ import annotations

import asyncio
import json
import os
from time import perf_counter
from typing import Any

import pandas as pd
from openai import AsyncOpenAI, OpenAI
import dotenv

from src.data import evaluate_predictions, now_stamp, save_json
from src.dataset_scripts import load_prosocial_dialog_bundle, make_dspy_sample_splits

dotenv.load_dotenv()

SEED = 42
SMART_MODEL = "gemini-3.1-flash-lite-preview"
REGULAR_MODEL = "gemma-3-4b-it"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_CONCURRENCY = 30
MAX_RETRIES = 30


def extract_first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found")

    depth = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError("No complete JSON object found")

    return json.loads(text[start : end + 1])


def build_smart_user_prompt(examples: list[dict[str, Any]], labels: list[str]) -> str:
    return (
        "You are optimizing a classification prompt for a smaller model (gemma-3-4b-it).\n"
        "Task: classify a user context into exactly one safety label.\n"
        f"Allowed labels: {labels}\n\n"
        "Using the examples below, create a high-performance prompt for the smaller model.\n"
        "The prompt must include:\n"
        "1) concise decision rules\n"
        "2) chain-of-thought instruction (think step by step silently)\n"
        "3) few-shot exemplars\n"
        "4) strict output contract: JSON object with key 'label'\n\n"
        "Return strictly one JSON object with this schema:\n"
        "{\n"
        "  \"optimized_prompt\": \"string\",\n"
        "  \"few_shot_count\": number\n"
        "}\n\n"
        "Training examples JSON:\n"
        f"{json.dumps(examples, ensure_ascii=True)}"
    )


def ask_smart_for_prompt(client: OpenAI, examples: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=SMART_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert prompt engineer for safety classification.",
            },
            {
                "role": "user",
                "content": build_smart_user_prompt(examples, labels),
            },
        ],
        temperature=0.2,
    )
    text = response.choices[0].message.content or ""
    payload = extract_first_json_object(text)
    if "optimized_prompt" not in payload:
        raise ValueError("Smart-model response missing optimized_prompt")
    return payload


def build_gemma_user_prompt(optimized_prompt: str, labels: list[str], context: str) -> str:
    return (
        f"{optimized_prompt}\n\n"
        f"Allowed labels: {labels}\n"
        "Return ONLY one JSON object: {\"label\": \"<one_allowed_label>\"}.\n"
        "Do not use markdown or code fences.\n\n"
        f"Context:\n{context}"
    )


async def label_one(
    row: tuple[int, str, str],
    *,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    optimized_prompt: str,
    labels: list[str],
) -> dict[str, Any]:
    source_index, context, true_label = row
    async with semaphore:
        response = await client.chat.completions.create(
            model=REGULAR_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": build_gemma_user_prompt(optimized_prompt, labels, context),
                }
            ],
            temperature=0.0,
        )

    raw_text = response.choices[0].message.content or ""
    pred_label = "parse_error"
    try:
        parsed = extract_first_json_object(raw_text)
        pred_label = str(parsed.get("label", "")).strip().lower()
    except Exception:
        pass

    if pred_label not in labels:
        pred_label = "parse_error"

    return {
        "context": context,
        "true_label": true_label,
        "pred_label": pred_label,
    }


async def run_async_inference(
    valid_subset: pd.DataFrame,
    *,
    optimized_prompt: str,
    labels: list[str],
    client: AsyncOpenAI,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(min(MAX_CONCURRENCY, len(valid_subset)))
    rows = [
        (int(r[0]), str(r[1]), str(r[2]))
        for r in valid_subset[["source_index", "context", "safety_label"]].itertuples(index=False, name=None)
    ]
    tasks = [
        label_one(
            row,
            client=client,
            semaphore=semaphore,
            optimized_prompt=optimized_prompt,
            labels=labels,
        )
        for row in rows
    ]
    return await asyncio.gather(*tasks)


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY before running this script")

    bundle = load_prosocial_dialog_bundle()
    train_df = bundle["train_df"]
    test_df = bundle["test_df"]
    labels = bundle["label_order"]
    results_dir = bundle["results_dir"]

    prompt_sample = train_df.sample(n=50, random_state=SEED).reset_index(drop=True)
    test_subset = make_dspy_sample_splits(test_df, seed=SEED, sample_size=50, train_size=25)["dspy_test_df"].reset_index(drop=True)

    smart_examples = [
        {
            "context": str(r[0]),
            "label": str(r[1]),
        }
        for r in prompt_sample[["context", "safety_label"]].itertuples(index=False, name=None)
    ]

    sync_client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        max_retries=MAX_RETRIES,
    )
    async_client = AsyncOpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        max_retries=MAX_RETRIES,
    )

    t0 = perf_counter()
    smart_payload = ask_smart_for_prompt(sync_client, smart_examples, labels)
    prompt_seconds = perf_counter() - t0

    optimized_prompt = str(smart_payload["optimized_prompt"])

    t1 = perf_counter()
    results = asyncio.run(run_async_inference(test_subset, optimized_prompt=optimized_prompt, labels=labels, client=async_client))
    infer_seconds = perf_counter() - t1

    preds_df = pd.DataFrame(results)
    valid_eval_df = preds_df[preds_df["pred_label"] != "parse_error"].copy()
    if len(valid_eval_df) > 0:
        metrics = evaluate_predictions(
            pd.Series(valid_eval_df["true_label"], dtype="string"),
            pd.Series(valid_eval_df["pred_label"], dtype="string"),
            labels,
        )
    else:
        metrics = {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "report": {}}

    parse_error_count = int((preds_df["pred_label"] == "parse_error").sum())

    stamp = now_stamp()
    pred_path = results_dir / f"mvp_google_gemma_preds_test25_{stamp}.csv"
    summary_path = results_dir / f"mvp_google_gemma_summary_{stamp}.json"

    preds_df[["context", "true_label", "pred_label"]].to_csv(pred_path, index=False)

    save_json(
        summary_path,
        {
            "workflow": "gemini_prompt_transfer_to_gemma",
            "models": {
                "smart": SMART_MODEL,
                "regular": REGULAR_MODEL,
            },
            "google_openai_base_url": BASE_URL,
            "sample_sizes": {
                "prompt_sample": 50,
                "test_eval": 25,
            },
            "used_indexes": {
                "prompt_sample_50": [int(v) for v in prompt_sample["source_index"].tolist()],
                "test_subset_25": [int(v) for v in test_subset["source_index"].tolist()],
            },
            "timing_seconds": {
                "prompt_generation": float(prompt_seconds),
                "inference": float(infer_seconds),
                "total": float(prompt_seconds + infer_seconds),
            },
            "metrics": {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "parse_error_count": parse_error_count,
            },
            "optimized_prompt": optimized_prompt,
            "artifacts": {
                "predictions_csv": str(pred_path),
            },
        },
    )

    print(
        {
            "summary": str(summary_path),
            "predictions": str(pred_path),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "parse_error_count": parse_error_count,
        }
    )


if __name__ == "__main__":
    main()