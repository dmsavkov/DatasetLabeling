# %% [markdown]
# ### Centroid few-shot — embedding centroid anchors + DSPy batch classify
# Preliminary: first batch after centroids built. Full: all batches. See experimentation-rules.mdc.

# %%
from __future__ import annotations

import ast
import json
import os
import re
import time

import dspy
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report as clf_report_str, pairwise_distances_argmin_min

import prompt_eng_common as pec

EXPERIMENT_SLUG = "centroid_fewshot"
NUM_TEST_SAMPLES = int(os.getenv("NUM_TEST_SAMPLES", "40"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
TRAIN_POOL_N = int(os.getenv("TRAIN_POOL_N", "100"))
SKIP_PRELIMINARY = pec.env_bool("SKIP_PRELIMINARY", False)
PRELIM_ONLY = pec.env_bool("PRELIM_ONLY", False)
MODEL = os.getenv("MODEL", "gemma-4-31b-it")

CANON_LABELS = list(pec.VALID_LABELS)


def _normalize_label(s: str) -> str:
    v = (s or "").strip().lower()
    if v in ("conclusion", "concl", "concl.", "conclusive"):
        return "conclusions"
    if v == "method":
        return "methods"
    if v == "result":
        return "results"
    if v == "objective(s)":
        return "objective"
    return v


class PubMedCentroidSignature(dspy.Signature):
    """Classify medical sentences. Anchoring examples are centroid representatives per category."""

    input_texts = dspy.InputField(desc="Numbered list of sentences.")
    predicted_labels = dspy.OutputField(desc="JSON list of labels.")


class PubMedCentroidClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(PubMedCentroidSignature)

    def forward(self, input_texts):
        return self.predictor(input_texts=input_texts)


def run_batches(classifier, lm: dspy.LM, text_batches: list[list[str]], *, phase: str) -> list[str]:
    all_preds: list[str] = []
    for i, batch in enumerate(text_batches):
        logger.info("[{}] Batch {}/{} (len={})", phase, i + 1, len(text_batches), len(batch))
        formatted_input = "\n".join([f"{idx + 1}. {txt}" for idx, txt in enumerate(batch)])
        with dspy.context(lm=lm):
            try:
                t0 = time.perf_counter()
                res = classifier(input_texts=formatted_input)
                raw_attr = res["predicted_labels"]
                raw_output = str(raw_attr)
                logger.debug("[{}] batch {} raw (trunc): {}...", phase, i, raw_output[:400])
                preds: list[str] = []
                try:
                    if isinstance(raw_attr, list):
                        preds = [_normalize_label(str(p)) for p in raw_attr]
                    else:
                        m = re.search(r"\[[\s\S]*\]", raw_output)
                        if m:
                            blob = m.group(0)
                            try:
                                data = json.loads(blob)
                            except Exception:
                                data = ast.literal_eval(blob)
                            if isinstance(data, list):
                                preds = [_normalize_label(str(p)) for p in data]
                    if not preds:
                        preds = re.findall(
                            r"(background|objective|methods|results|conclusions)",
                            raw_output.lower(),
                        )
                        preds = [_normalize_label(p) for p in preds]
                except Exception:
                    preds = []
                preds = [p if p in CANON_LABELS else "error" for p in preds]
                dt = time.perf_counter() - t0
                logger.info("[{}] batch {} parsed in {:.2f}s -> {} labels", phase, i, dt, len(preds))
            except Exception as e:
                logger.exception("[{}] batch {} error: {}", phase, i, e)
                preds = ["error"] * len(batch)

        preds = (preds + ["error"] * len(batch))[: len(batch)]
        all_preds.extend(preds)
    return all_preds


def sample_balanced_eval(eval_df: pd.DataFrame, num_samples: int, seed: int) -> pd.DataFrame:
    """
    Stratified sample by canonical `label_name` (must match pec.PUBMED_ID2LABEL / VALID_LABELS).

    Do not map names to integers with a hand-written table: HF class ids follow
    pec.PUBMED_LABEL2ID (e.g. objective=3, conclusions=1), not alphabetical order.
    """
    pec.require_keys(eval_df, "label_name", "label", "text")
    valid_labels = list(pec.VALID_LABELS)
    present_labels = [lab for lab in valid_labels if (eval_df["label_name"].astype(str).str.lower() == lab).any()]
    n_labels = len(present_labels)
    if n_labels == 0:
        raise ValueError("No valid evaluation labels found in the eval set.")

    n_per_label = num_samples // n_labels
    leftover = num_samples % n_labels

    sampled_dfs = []
    for idx, label in enumerate(present_labels):
        label_rows = eval_df[eval_df["label_name"].astype(str).str.lower() == label]
        take_n = n_per_label + (1 if idx < leftover else 0)
        if len(label_rows) == 0:
            logger.warning("No eval rows for label {} in eval set", label)
            continue
        # Sample with or without replacement as needed
        if take_n > len(label_rows):
            sample = label_rows.sample(n=take_n, replace=True, random_state=seed)
        else:
            sample = label_rows.sample(n=take_n, replace=False, random_state=seed)
        sampled_dfs.append(sample)

    if not sampled_dfs:
        raise ValueError("No samples drawn; check label mapping and label values.")
    balanced_eval = pd.concat(sampled_dfs, ignore_index=True)
    balanced_eval = balanced_eval.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(
        "Final balanced eval sample (label_name / raw label):\n{}",
        balanced_eval[["label_name", "label", "text"]].to_string(index=False),
    )
    return balanced_eval


def main() -> None:
    if not pec.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY required.")

    run_dir = pec.begin_run(EXPERIMENT_SLUG)
    eval_df, _, train_df, _ = pec.load_hf_pubmed_splits()

    logger.info("Step 1: train pool n={} embeddings...", TRAIN_POOL_N)
    t_emb = time.perf_counter()
    train_pool_df = train_df.sample(n=min(TRAIN_POOL_N, len(train_df)), random_state=pec.DEFAULT_SEED)
    train_pool_df = train_pool_df.copy()
    train_pool_df["label_name"] = train_pool_df["label"].apply(pec.label_name_from_value)
    train_embeddings = pec.embed_texts(train_pool_df["text"].tolist())
    logger.info("Embeddings shape {} in {:.2f}s", train_embeddings.shape, time.perf_counter() - t_emb)

    logger.info("Step 2: per-class centroid → nearest sentence")
    centroid_examples: list[str] = []
    centroid_label_names = pec.VALID_LABELS if hasattr(pec, "VALID_LABELS") else pec.PUBMED_LABEL_NAMES
    for label in centroid_label_names:
        label_indices = train_pool_df[train_pool_df["label_name"] == label].index
        if len(label_indices) == 0:
            logger.warning("No train rows for label {}", label)
            continue
        class_embeddings = train_embeddings[train_pool_df.index.get_indexer(label_indices)]
        centroid = class_embeddings.mean(axis=0).reshape(1, -1)
        closest_idx, _ = pairwise_distances_argmin_min(centroid, class_embeddings)
        representative_text = train_pool_df.loc[label_indices[closest_idx[0]], "text"]
        centroid_examples.append(f"Label: {label.upper()}\nRepresentative Sentence: {representative_text}")
        logger.info("Label {}: anchor len {} chars", label, len(str(representative_text)))

    centroid_fewshot_str = "\n\n".join(centroid_examples)
    PubMedCentroidSignature.__doc__ += f"\n\n### CENTROID ANCHORS:\n{centroid_fewshot_str}"

    classifier = PubMedCentroidClassifier()
    lm_prelim = dspy.LM(
        model=f"openai/{MODEL}",
        api_base=pec.GOOGLE_OPENAI_BASE_URL,
        api_key=pec.GOOGLE_API_KEY,
        num_retries=pec.PRELIM_MAX_RETRIES,
    )
    lm_full = dspy.LM(
        model=f"openai/{MODEL}",
        api_base=pec.GOOGLE_OPENAI_BASE_URL,
        api_key=pec.GOOGLE_API_KEY,
        num_retries=pec.MAIN_MAX_RETRIES,
    )

    eval_subset = sample_balanced_eval(eval_df, NUM_TEST_SAMPLES, pec.DEFAULT_SEED)
    text_batches = [
        eval_subset["text"].tolist()[i : i + BATCH_SIZE] for i in range(0, len(eval_subset), BATCH_SIZE)
    ]
    all_true = eval_subset["label_name"].tolist()

    if not SKIP_PRELIMINARY and text_batches:
        logger.info("=== PRELIMINARY: batch 0 only ===")
        t0 = time.perf_counter()
        pre_preds = run_batches(classifier, lm_prelim, [text_batches[0]], phase="prelim")
        dt = time.perf_counter() - t0
        pre_true = all_true[: len(text_batches[0])]
        pre_acc = accuracy_score(pre_true, pre_preds[: len(pre_true)])
        pre_cr_d, pre_cr_t = pec.sklearn_classification_reports(pre_true, pre_preds[: len(pre_true)])
        logger.info("Prelim batch accuracy: {:.2%}", pre_acc)

        pec.save_phase(
            run_dir,
            "preliminary",
            metrics={
                "accuracy": float(pre_acc),
                "n_samples": len(text_batches[0]),
                "n_batches": 1,
                "classification_report": pre_cr_d,
                "classification_report_text": pre_cr_t,
            },
            settings={"model": f"openai/{MODEL}", "BATCH_SIZE": BATCH_SIZE, "num_retries": pec.PRELIM_MAX_RETRIES},
            predictions=[
                {"pred": p, "true": t, "text": tx[:320]}
                for p, t, tx in zip(pre_preds, pre_true, text_batches[0])
            ],
            duration_seconds=dt,
        )
        if PRELIM_ONLY:
            logger.warning("PRELIM_ONLY=1 — skip full.")
            return

    logger.info("=== FULL: {} batches ===", len(text_batches))
    t1 = time.perf_counter()
    all_preds = run_batches(classifier, lm_full, text_batches, phase="full")
    dt_full = time.perf_counter() - t1

    acc = accuracy_score(all_true, all_preds)
    report_d, report_txt = pec.sklearn_classification_reports(all_true, all_preds)
    logger.info("Full accuracy {:.2%} wall {:.2f}s", acc, dt_full)

    pec.save_phase(
        run_dir,
        "full",
        metrics={
            "accuracy": float(acc),
            "n_samples": len(all_true),
            "n_batches": len(text_batches),
            "classification_report": report_d,
            "classification_report_text": report_txt,
        },
        settings={
            "model": f"openai/{MODEL}",
            "NUM_TEST_SAMPLES": NUM_TEST_SAMPLES,
            "BATCH_SIZE": BATCH_SIZE,
            "TRAIN_POOL_N": TRAIN_POOL_N,
            "num_retries": pec.MAIN_MAX_RETRIES,
        },
        predictions=[
            {"pred": p, "true": t, "text": tx[:320]} for p, t, tx in zip(all_preds, all_true, eval_subset["text"])
        ],
        duration_seconds=dt_full,
    )

    print(clf_report_str(all_true, all_preds))


if __name__ == "__main__":
    main()
