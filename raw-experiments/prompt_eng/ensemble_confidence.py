# %% [markdown]
# ### Ensemble of confidence — multiple models, batched requests, async preserved
# Preliminary: first batch only (prints only). Full: FULL_N rows (saved under results/raw/prompt_eng/...).
#
# Key: we do **batch prompting** (one request per (model, batch_of_texts)) and keep async via ThreadPool.

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report as clf_report_str
from tqdm.auto import tqdm

import prompt_eng_common as pec

EXPERIMENT_SLUG = "ensemble_confidence"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
FULL_N = int(os.getenv("FULL_N", "40"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))

SKIP_PRELIMINARY = pec.env_bool("SKIP_PRELIMINARY", False)
PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)

LABEL_CHOICES = list(pec.VALID_LABELS)

MODELS = [
    "gemma-4-31b-it",
    "gemini-3.1-flash-lite-preview",
    "gemma-4-26b-a4b-it",
]


def build_batch_prompt(texts: list[str]) -> tuple[str, str]:
    labels_str = ", ".join(LABEL_CHOICES)
    system_msg = (
        "You are a careful text classification judge.\n"
        f"Available labels (choose only from these): [{labels_str}].\n"
        "You will receive a numbered list of sentences.\n"
        "Return ONLY a JSON list with exactly N objects (same order), where each object is:\n"
        '{"label": "<one of labels>", "confidence": 0.0..1.0}\n'
        "No extra keys. No extra text."
    )
    numbered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
    user_msg = f"N={len(texts)}. Classify each sentence.\n\n{numbered}"
    return system_msg, user_msg


def _extract_json_list(raw: str) -> list | None:
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def parse_batch_verdicts(raw: str, n: int) -> list[dict]:
    """
    Parse model output into exactly n verdict dicts: {"label": str, "confidence": float}.
    Returns 'error' rows if parsing fails.
    """
    raw = raw or ""
    data = _extract_json_list(raw)
    if data is None:
        # Sometimes models return a dict containing a list
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, list):
                        data = v
                        break
        except Exception:
            data = None

    verdicts: list[dict] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                label = str(item.get("label", "error")).strip().lower()
                conf_val = item.get("confidence", item.get("conf", 0.0))
                try:
                    conf = float(conf_val)
                except Exception:
                    conf = 0.0
            else:
                label, conf = "error", 0.0

            if label not in LABEL_CHOICES:
                label = "error"
            if not (0.0 <= conf <= 1.0):
                conf = 0.0
            verdicts.append({"label": label, "confidence": conf})

    if len(verdicts) < n:
        verdicts.extend([{"label": "error", "confidence": 0.0}] * (n - len(verdicts)))
    return verdicts[:n]


def call_model_on_batch(texts: list[str], model_name: str, *, prelim: bool) -> list[dict]:
    client = pec.get_openai_client(max_retries=pec.PRELIM_MAX_RETRIES if prelim else pec.MAIN_MAX_RETRIES)
    system_msg, user_msg = build_batch_prompt(texts)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0.0,
        # Per your rules: default max_tokens=None unless explicitly needed.
        max_tokens=None,
    )
    dt = time.perf_counter() - t0
    raw = resp.choices[0].message.content or ""
    logger.info("Model {} batch n={} finished in {:.2f}s (chars={})", model_name, len(texts), dt, len(raw))
    logger.debug("Model {} raw (trunc): {}...", model_name, raw[:600])
    return parse_batch_verdicts(raw, len(texts))


def ensemble_from_model_outputs(
    *,
    rows: list[dict],
    per_model_verdicts: dict[str, list[dict]],
) -> list[dict]:
    finals: list[dict] = []
    for i, row in enumerate(rows):
        votes = []
        for m, verdicts in per_model_verdicts.items():
            v = verdicts[i] if i < len(verdicts) else {"label": "error", "confidence": 0.0}
            votes.append({"model": m, "label": v["label"], "conf": float(v["confidence"])})

        labels = [v["label"] for v in votes if v["label"] != "error"]
        if not labels:
            final_pred = "error"
            avg_conf = 0.0
            disagreement = True
        else:
            final_pred = max(set(labels), key=labels.count)
            avg_conf = float(np.mean([v["conf"] for v in votes]))
            disagreement = len(set(labels)) > 1

        finals.append(
            {
                "text": str(row["text"])[:400],
                "true": row["label_name"],
                "pred": final_pred,
                "avg_conf": avg_conf,
                "disagreement": disagreement,
                "raw_votes": votes,
            }
        )
    return finals

def save_confidence_figure(df: pd.DataFrame, run_dir: str, phase: str):
    fig, ax = plt.subplots(figsize=(7,4))
    df['avg_conf'].hist(ax=ax, bins=20, alpha=0.7)
    ax.set_title(f"Distribution of average ensemble confidence ({phase})")
    ax.set_xlabel("Average Confidence (majority vote)")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(run_dir, f"ensemble_confidence_hist_{phase}.png")
    plt.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved confidence histogram to {out_path}")

def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    eval_df, _, _, _ = pec.load_hf_pubmed_splits()

    if not SKIP_PRELIMINARY:
        logger.info("=== PRELIMINARY: first batch only (prints only) ===")
        pre_df = eval_df.head(BATCH_SIZE)
        pre_rows = [row for _, row in pre_df.iterrows()]
        pre_rows_d = [{"text": r["text"], "label_name": r["label_name"]} for r in pre_rows]
        texts = [r["text"] for r in pre_rows_d]
        logger.info("Prelim batch size {} | labels choices {}", len(texts), LABEL_CHOICES)

        per_model: dict[str, list[dict]] = {}
        for m in MODELS:
            per_model[m] = call_model_on_batch(texts, m, prelim=True)
        pre_final = ensemble_from_model_outputs(rows=pre_rows_d, per_model_verdicts=per_model)

        for i, item in enumerate(pre_final):
            logger.info(
                "Prelim[{}] pred={} true={} avg_conf={:.2f} disagree={}",
                i,
                item["pred"],
                item["true"],
                item["avg_conf"],
                item["disagreement"],
            )
        # Cheap sanity assertion: we should get one vote per model per item
        assert all(len(x["raw_votes"]) == len(MODELS) for x in pre_final), "Missing votes in preliminary parse"
        if PRELIM_ONLY:
            logger.warning("PRELIM_ONLY=1 — skip full.")
            return

    logger.info("=== FULL: {} rows ===", FULL_N)
    eval_subset = eval_df.sample(FULL_N, random_state=pec.DEFAULT_SEED)
    rows_list = [{"text": r["text"], "label_name": r["label_name"]} for _, r in eval_subset.iterrows()]
    batches: list[list[dict]] = [rows_list[i : i + BATCH_SIZE] for i in range(0, len(rows_list), BATCH_SIZE)]

    logger.info("Full batches: {} (batch_size={})", len(batches), BATCH_SIZE)
    t1 = time.perf_counter()
    ensemble_results: list[dict] = []

    def process_one_batch(batch_idx: int, rows: list[dict]) -> tuple[int, list[dict]]:
        texts = [r["text"] for r in rows]
        # Async within batch: submit (model, batch) calls in parallel.
        per_model: dict[str, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(MODELS))) as ex:
            futs = {ex.submit(call_model_on_batch, texts, m, prelim=False): m for m in MODELS}
            for fut in as_completed(futs):
                m = futs[fut]
                per_model[m] = fut.result()
        finals = ensemble_from_model_outputs(rows=rows, per_model_verdicts=per_model)
        return batch_idx, finals

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futs = {executor.submit(process_one_batch, i, b): i for i, b in enumerate(batches)}
        indexed: list[tuple[int, list[dict]]] = []
        for fut in tqdm(as_completed(futs), total=len(futs), desc="batches"):
            indexed.append(fut.result())

    indexed.sort(key=lambda x: x[0])
    for _, finals in indexed:
        ensemble_results.extend(finals)

    dt_full = time.perf_counter() - t1

    ens_df = pd.DataFrame(ensemble_results)
    acc = float(accuracy_score(ens_df["true"], ens_df["pred"]))
    logger.info("Ensemble accuracy {:.2%} | disagreements {}", acc, int(ens_df["disagreement"].sum()))

    high_conf_mask = ens_df["avg_conf"] > 0.8
    extra_metrics: dict = {"n_rows": len(ens_df), "accuracy": acc}
    if high_conf_mask.any():
        extra_metrics["accuracy_high_conf"] = float(
            accuracy_score(ens_df[high_conf_mask]["true"], ens_df[high_conf_mask]["pred"])
        )

    cr_d, cr_t = pec.sklearn_classification_reports(ens_df["true"], ens_df["pred"])
    pec.save_phase(
        run_dir,
        "full",
        metrics={
            **extra_metrics,
            "classification_report": cr_d,
            "classification_report_text": cr_t,
        },
        settings={"MODELS": MODELS, "FULL_N": FULL_N, "num_retries": pec.MAIN_MAX_RETRIES, "LABEL_CHOICES": LABEL_CHOICES},
        predictions=ensemble_results,
        duration_seconds=dt_full,
    )

    # Save figure visualizing ensemble confidence for all predictions
    save_confidence_figure(ens_df, run_dir, "full")

    print(ens_df[["true", "pred", "avg_conf", "disagreement"]].head(10).to_string())
    print(clf_report_str(ens_df["true"], ens_df["pred"]))

    try:
        from IPython.display import display
        display(ens_df[["true", "pred", "avg_conf", "disagreement"]].head(10))
    except Exception:
        pass

if __name__ == "__main__":
    main()
