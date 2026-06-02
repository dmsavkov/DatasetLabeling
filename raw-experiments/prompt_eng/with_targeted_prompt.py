# %% [markdown]
# ### With targeted prompt — OpenAI batch + rubric (JSON extraction)
# Preliminary pass (first batch, cheap retries) → full evaluation. See experimentation-rules.mdc.

# %%
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report as clf_report_str
from tqdm.auto import tqdm

import prompt_eng_common as pec

EXPERIMENT_SLUG = "with_targeted_prompt"

TARGET_PROMPT_BLOCK = """
### CLASSIFICATION RUBRIC: CONTRASTING BORDERLINE CASES

To distinguish between high-overlap categories, apply the following logic:

1. METHODS vs. RESULTS (The "Action-Data" Split)
   - METHODS: Focus on the protocol and researcher intent. If the sentence describes WHAT was done (e.g., "stratified," "performed," "measured," "tests were used"), it is METHODS.
   - RESULTS: Focus on what the world did back. If the sentence describes WHAT was found (e.g., "revealed," "increased," "was @%," "showed improvement"), it is RESULTS.
   - BORDERLINE RULE: A sentence describing the "number of participants who did X" is a RESULT of the recruitment process, even if it feels like a procedural description.

2. RESULTS vs. CONCLUSIONS (The "Fact-Meaning" Split)
   - RESULTS: Hard empirical data. Observations that are true regardless of theory (e.g., "Group A was 10% higher than Group B").
   - CONCLUSIONS: Interpretations, implications, or future-facing statements. Look for hedged language or significance (e.g., "suggests," "may indicate," "is a promising intervention," "is warranted").
   - BORDERLINE RULE: If a sentence reports a discovery (e.g., "revealed candidate genes"), it is a RESULT. If it explains the potential impact of those genes, it is a CONCLUSION.

3. OBJECTIVE vs. BACKGROUND (The "Goal-Context" Split)
   - BACKGROUND: The status quo or the problem in the field.
   - OBJECTIVE: The specific "mission statement" of this study (e.g., "We aimed to," "This study assesses").
"""

VALID_LABELS = {"background", "objective", "methods", "results", "conclusions"}
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
SKIP_PRELIMINARY = pec.env_bool("SKIP_PRELIMINARY", False)
PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)

TARGET_MODEL = pec.DEFAULT_MODEL


def process_batch_direct(idx: int, batch: list[str], client: OpenAI):
    n = len(batch)
    logger.debug("Batch {}: {} sentences", idx, n)
    formatted_input = "\n".join([f"{i + 1}. {txt}" for i, txt in enumerate(batch)])
    system_msg = (
        f"Classify a list of medical sentences. VALID LABELS: [background, objective, methods, results, conclusions]. "
        f"Return a JSON LIST containing exactly {n} labels as strings.\n{TARGET_PROMPT_BLOCK}"
    )
    try:
        t_req = time.perf_counter()
        response = client.chat.completions.create(
            model=TARGET_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": f"Classify these sentences and return ONLY a JSON list of {n} labels:\n{formatted_input}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        dt = time.perf_counter() - t_req
        raw_content = response.choices[0].message.content or ""
        logger.info("Batch {} completed in {:.2f}s (response chars={})", idx, dt, len(raw_content))
        if idx == 0:
            logger.debug("Batch 0 raw (truncated): {}...", raw_content[:600])

        json_match = re.search(r"\[.*\]", raw_content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
        else:
            data = json.loads(raw_content)

        preds: list[str] = []
        if isinstance(data, list):
            preds = [str(p).lower().strip() for p in data]
        elif isinstance(data, dict):
            potential_list = next((v for v in data.values() if isinstance(v, list)), None)
            if potential_list:
                preds = [str(p).lower().strip() for p in potential_list]
            else:
                for i in range(1, n + 1):
                    val = data.get(str(i), data.get(i, "error"))
                    preds.append(str(val).lower().strip())

        preds = [p if p in VALID_LABELS else "error" for p in preds]
        if len(preds) < n:
            preds += ["error"] * (n - len(preds))
    except Exception as e:
        logger.exception("Batch {} failed: {}", idx, e)
        preds = ["error"] * n

    return idx, preds[:n]


def predictions_records(
    texts: list[str], preds: list[str], truths: list[str], *, max_text_chars: int = 320
) -> list[dict]:
    out = []
    for i, (tx, pr, tr) in enumerate(zip(texts, preds, truths)):
        out.append(
            {
                "i": i,
                "pred": pr,
                "true": tr,
                "text": (tx[:max_text_chars] + ("…" if len(tx) > max_text_chars else "")),
            }
        )
    return out


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    logger.info(
        "Model={} batch_size={} prelim_retries={} main_retries={}",
        TARGET_MODEL,
        BATCH_SIZE,
        pec.PRELIM_MAX_RETRIES,
        pec.MAIN_MAX_RETRIES,
    )

    logger.info("Loading PubMed splits...")
    eval_df, _, _, _ = pec.load_hf_pubmed_splits()
    eval_df = eval_df.sample(30, random_state=pec.DEFAULT_SEED)

    eval_texts = eval_df["text"].tolist()
    true_labels = eval_df["label_name"].tolist()
    text_batches = [eval_texts[i : i + BATCH_SIZE] for i in range(0, len(eval_texts), BATCH_SIZE)]
    logger.info("Built {} batches ({} eval rows).", len(text_batches), len(eval_texts))

    # ----- Preliminary: first batch only -----
    if not SKIP_PRELIMINARY and text_batches:
        logger.info("=== PRELIMINARY: first batch only (max_retries={}) ===", pec.PRELIM_MAX_RETRIES)
        prelim_client = pec.get_openai_client(max_retries=pec.PRELIM_MAX_RETRIES)
        t0 = time.perf_counter()
        fb_idx, fb = 0, text_batches[0]
        _, prelim_preds = process_batch_direct(fb_idx, fb, prelim_client)
        prelim_dt = time.perf_counter() - t0
        prelim_true = true_labels[: len(fb)]
        prelim_acc = accuracy_score(prelim_true, prelim_preds[: len(prelim_true)])
        pre_d, pre_t = pec.sklearn_classification_reports(prelim_true, prelim_preds[: len(prelim_true)])

        for i, sent in enumerate(fb):
            logger.info("Prelim [{}] text snippet: {}...", i + 1, sent[:120].replace("\n", " "))
        for i, (p, t) in enumerate(zip(prelim_preds, prelim_true)):
            logger.info("Prelim [{}] pred={} true={}", i + 1, p, t)
        logger.info("Preliminary batch accuracy (single batch): {:.2%}", prelim_acc)

        pec.save_phase(
            run_dir,
            "preliminary",
            metrics={
                "accuracy": float(prelim_acc),
                "n_samples": len(fb),
                "n_batches": 1,
                "batch_indices": [0],
                "classification_report": pre_d,
                "classification_report_text": pre_t,
            },
            settings={
                "TARGET_MODEL": TARGET_MODEL,
                "BATCH_SIZE": BATCH_SIZE,
                "max_retries": pec.PRELIM_MAX_RETRIES,
            },
            predictions=predictions_records(fb, prelim_preds, prelim_true),
            duration_seconds=prelim_dt,
            notes="First batch only; cheap retries",
        )

        if PRELIM_ONLY:
            logger.warning("PRELIM_ONLY=1 — skipping full run.")
            return

    elif SKIP_PRELIMINARY:
        logger.info("SKIP_PRELIMINARY=1 — skipping preliminary phase.")

    # ----- Full run -----
    logger.info("=== FULL RUN (max_retries={}) ===", pec.MAIN_MAX_RETRIES)
    full_client = pec.get_openai_client(max_retries=pec.MAIN_MAX_RETRIES)
    t_full = time.perf_counter()
    all_indexed_results: list = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(process_batch_direct, i, b, full_client): i for i, b in enumerate(text_batches)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="batches"):
            all_indexed_results.append(future.result())

    all_indexed_results.sort(key=lambda x: x[0])
    all_preds = [p for _, batch_preds in all_indexed_results for p in batch_preds]
    full_dt = time.perf_counter() - t_full

    y_true = true_labels[: len(all_preds)]
    acc = accuracy_score(y_true, all_preds)
    report_d, report_txt = pec.sklearn_classification_reports(y_true, all_preds)
    logger.info("Full run finished in {:.2f}s — accuracy {:.2%}", full_dt, acc)

    pec.save_phase(
        run_dir,
        "full",
        metrics={
            "accuracy": float(acc),
            "n_samples": len(all_preds),
            "n_batches": len(text_batches),
            "classification_report": report_d,
            "classification_report_text": report_txt,
        },
        settings={"TARGET_MODEL": TARGET_MODEL, "BATCH_SIZE": BATCH_SIZE, "max_retries": pec.MAIN_MAX_RETRIES},
        predictions=predictions_records(eval_texts[: len(all_preds)], all_preds, y_true),
        duration_seconds=full_dt,
        notes="All batches; threaded executor",
    )

    print("\n" + clf_report_str(y_true, all_preds))
    logger.info("Done. Artifacts under {}", run_dir)


if __name__ == "__main__":
    main()
