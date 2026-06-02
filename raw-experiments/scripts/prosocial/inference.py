from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
from openai import AsyncOpenAI

from prosocial.constants import FLASH_MODEL
from prosocial.prompting import build_prediction_prompt, extract_first_json_object, parse_prediction_label


async def extract_statements_with_flash(*, client: AsyncOpenAI, context: str) -> list[str]:
    user_prompt = (
        "Extract concise safety-relevant statements from the context for downstream classification.\n"
        "Return exactly one JSON object: {\"statements\": [\"...\", \"...\"]}.\n"
        "Use short factual statements; do not classify.\n\n"
        f"Context:\n{context}"
    )
    response = await client.chat.completions.create(
        model=FLASH_MODEL,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.0,
    )
    try:
        payload = extract_first_json_object(response.choices[0].message.content or "")
        values = payload.get("statements", [])
        return [str(v).strip() for v in values if str(v).strip()] if isinstance(values, list) else []
    except Exception:
        return []


def route_moe_expert(context: str, experts: list[str]) -> str:
    if not experts:
        raise ValueError("MoE experts list is empty")
    if len(experts) == 1:
        return experts[0]

    context_l = context.lower()
    urgent = ["kill", "murder", "suicide", "bomb", "rape", "pedo", "weapon", "shoot"]
    high_risk = ["hate", "racist", "slur", "abuse", "threat", "assault", "violent"]

    if any(t in context_l for t in urgent):
        return experts[min(2, len(experts) - 1)]
    if len(context) > 260 or any(t in context_l for t in high_risk):
        return experts[min(1, len(experts) - 1)]
    return experts[0]


async def predict_one(
    row: dict[str, Any],
    *,
    client: AsyncOpenAI,
    labels: list[str],
    optimized_prompt: str,
    prediction_model: str,
    assertion_text: str,
    static_fewshots: list[dict[str, Any]],
    retrieval_map: dict[int, list[dict[str, Any]]] | None,
    enable_statement_extraction: bool,
    moe_experts: list[str] | None,
) -> dict[str, Any]:
    source_index = int(row["source_index"])
    context = str(row["context"])
    true_label = str(row["safety_label"])

    selected_model = route_moe_expert(context, moe_experts) if moe_experts is not None else prediction_model
    extracted_statements = await extract_statements_with_flash(client=client, context=context) if enable_statement_extraction else []
    fewshots = retrieval_map.get(source_index, []) if retrieval_map is not None else static_fewshots

    response = await client.chat.completions.create(
        model=selected_model,
        messages=[
            {
                "role": "user",
                "content": build_prediction_prompt(
                    optimized_prompt=optimized_prompt,
                    labels=labels,
                    context=context,
                    assertion_text=assertion_text,
                    fewshot_examples=fewshots,
                    extracted_statements=extracted_statements,
                ),
            }
        ],
        temperature=0.0,
    )

    pred_label = parse_prediction_label(response.choices[0].message.content or "", labels)
    return {
        "source_index": source_index,
        "context": context,
        "true_label": true_label,
        "pred_label": pred_label,
        "model_used": selected_model,
        "statement_count": len(extracted_statements),
    }


async def run_inference(
    test_df: pd.DataFrame,
    *,
    client: AsyncOpenAI,
    labels: list[str],
    optimized_prompt: str,
    prediction_model: str,
    assertion_text: str,
    static_fewshots: list[dict[str, Any]],
    batch_size: int,
    max_concurrency: int,
    retrieval_map: dict[int, list[dict[str, Any]]] | None,
    enable_statement_extraction: bool,
    moe_experts: list[str] | None,
) -> list[dict[str, Any]]:
    rows = test_df[["source_index", "context", "safety_label"]].to_dict(orient="records")
    if not rows:
        return []

    semaphore = asyncio.Semaphore(min(max_concurrency, len(rows)))

    async def guarded(row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await predict_one(
                row,
                client=client,
                labels=labels,
                optimized_prompt=optimized_prompt,
                prediction_model=prediction_model,
                assertion_text=assertion_text,
                static_fewshots=static_fewshots,
                retrieval_map=retrieval_map,
                enable_statement_extraction=enable_statement_extraction,
                moe_experts=moe_experts,
            )

    all_results: list[dict[str, Any]] = []
    for start in range(0, len(rows), max(1, batch_size)):
        chunk = rows[start : start + max(1, batch_size)]
        all_results.extend(await asyncio.gather(*(guarded(row) for row in chunk)))
    return all_results
