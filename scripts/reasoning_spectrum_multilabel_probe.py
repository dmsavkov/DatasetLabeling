# pyright: basic
"""
Reasoning spectrum probe: multi-label batch classification × 6 logical tiers.

Suites:
  - pubmed (default): 5 PubMed-RCT sentences, 1 API call per tier.
  - banking-all (--tier banking-all): 20 banking77 texts where a prior run had
    is_confusing=true, 4 batch requests × 5 items per tier (6 tiers = 24 calls).

Examples:
  uv run python scripts/reasoning_spectrum_multilabel_probe.py --tier 1
  uv run python scripts/reasoning_spectrum_multilabel_probe.py --tier banking-all
  uv run python scripts/reasoning_spectrum_multilabel_probe.py --tier 3 --suite banking --predictions-path <path>

Environment:
  GOOGLE_API_KEY / GEMINI_API_KEY
  TIER34_MODEL (default gemini-3.5-flash), TIER12_MODEL (gemini-3.1-flash-lite-preview)
  TIER5_MODEL / TIER6_MODEL (gemma-4-26b-a4b-it / gemma-4-31b-it)
  BANKING_CONFUSING_PRED_PATH — source full_predictions.json for banking suite
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig, ThinkingLevel
from loguru import logger

from src.models.llm.google_genai_batch import (
    _make_google_genai_client,
    _parts_from_response,
    _usage_from_response,
)
from src.prompts.baseline import BatchItem
from src.prompts.parsing import (
    confusion_from_label_lists,
    multilabel_confusion_kind,
    parse_multilabel_batch,
)
from src.utils.env import load_dotenv_if_present

load_dotenv_if_present()

PUBMED_LABELS = ["background", "objective", "methods", "results", "conclusions"]
BATCH_SIZE = 5

DEFAULT_BANKING_PRED_PATH = (
    Path("results")
    / "prompt_eng"
    / "20260519_092956"
    / "multilabel_confusion_probe"
    / "multilabel_gemma4_31b_think_high_banking10_test200"
    / "full_predictions.json"
)

PromptStrategy = Literal["stripped", "external_cot", "booster"]


@dataclass(frozen=True, slots=True)
class DatasetContext:
    name: str
    allowed_labels: list[str]
    task_title: str
    domain_line: str


@dataclass(frozen=True, slots=True)
class TierSpec:
    tier: int
    name: str
    model_id: str
    thinking_budget: int | None
    thinking_level: str | None
    include_thoughts: bool
    prompt_strategy: PromptStrategy
    notes: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _api_key() -> str:
    for env in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        v = os.getenv(env)
        if v:
            return str(v)
    raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY.")


def _banking_labels() -> list[str]:
    card = _repo_root() / "data" / "processed" / "banking-10" / "dataset_card.json"
    if card.is_file():
        payload = json.loads(card.read_text(encoding="utf-8"))
        return [str(x) for x in payload["labels"]]
    return [
        "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit",
        "card_payment_fee_charged",
        "cash_withdrawal_charge",
        "declined_cash_withdrawal",
        "direct_debit_payment_not_recognised",
        "transaction_charged_twice",
        "transfer_fee_charged",
        "transfer_not_received_by_recipient",
        "wrong_amount_of_cash_received",
    ]


def _pubmed_context() -> DatasetContext:
    return DatasetContext(
        name="pubmed",
        allowed_labels=list(PUBMED_LABELS),
        task_title="PubMed-RCT rhetorical roles (multi-label)",
        domain_line="biomedical abstract sentences",
    )


def _banking_context() -> DatasetContext:
    return DatasetContext(
        name="banking",
        allowed_labels=_banking_labels(),
        task_title="Banking customer intent (multi-label)",
        domain_line="banking customer support utterances",
    )


def _tier_matrix() -> dict[int, TierSpec]:
    tier12 = os.getenv("TIER12_MODEL", "gemini-3.1-flash-lite-preview")
    tier34 = os.getenv("TIER34_MODEL", "gemini-3.5-flash")
    tier5 = os.getenv("TIER5_MODEL", "gemma-4-26b-a4b-it")
    tier6 = os.getenv("TIER6_MODEL", "gemma-4-31b-it")
    return {
        1: TierSpec(
            tier=1,
            name="pure_intuition",
            model_id=tier12,
            thinking_budget=0,
            thinking_level=None,
            include_thoughts=False,
            prompt_strategy="stripped",
            notes="No internal thinking; no visible CoT.",
        ),
        2: TierSpec(
            tier=2,
            name="surface_external_cot",
            model_id=tier12,
            thinking_budget=0,
            thinking_level=None,
            include_thoughts=False,
            prompt_strategy="external_cot",
            notes="No internal thinking; forced visible reasoning before JSON.",
        ),
        3: TierSpec(
            tier=3,
            name="micro_internal",
            model_id=tier34,
            thinking_budget=1024,
            thinking_level=None,
            include_thoughts=False,
            prompt_strategy="stripped",
            notes="Low thinking_budget.",
        ),
        4: TierSpec(
            tier=4,
            name="deep_internal",
            model_id=tier34,
            thinking_budget=8192,
            thinking_level=None,
            include_thoughts=False,
            prompt_strategy="stripped",
            notes="High thinking_budget.",
        ),
        5: TierSpec(
            tier=5,
            name="gemma_native_heavy",
            model_id=tier5,
            thinking_budget=None,
            thinking_level="high",
            include_thoughts=False,
            prompt_strategy="stripped",
            notes="Gemma: thinking_level=high only (rejects thinking_budget).",
        ),
        6: TierSpec(
            tier=6,
            name="gemma_overclocked_booster",
            model_id=tier6,
            thinking_budget=None,
            thinking_level="high",
            include_thoughts=False,
            prompt_strategy="booster",
            notes="Gemma: thinking_level=high + booster prompt.",
        ),
    }


def wrap_stripped_prompt(base_problem: str) -> str:
    return (
        f"TASK:\n{base_problem}\n\n"
        "CRITICAL FORMATTING CONSTRAINT:\n"
        "Provide only the final JSON result. Do NOT include introductory phrases, "
        "step-by-step breakdowns, or commentary outside the JSON array."
    )


def wrap_external_cot_prompt(base_problem: str) -> str:
    return (
        f"TASK:\n{base_problem}\n\n"
        "CRITICAL FORMATTING CONSTRAINT:\n"
        "You may think out loud first:\n"
        "1. Break down constraints of each sentence.\n"
        "2. Under a heading \"Logical Process\", outline steps per item id.\n"
        "3. The last line of your reply MUST start with exactly: Final Answer:\n"
        "4. After that prefix, output ONLY the JSON array from the task — valid JSON, "
        "no markdown fences, no trailing commentary.\n"
    )


def wrap_booster_prompt(base_problem: str, *, ctx: DatasetContext) -> str:
    boundary = (
        "Edge cases and boundary confusions between rhetorical sections."
        if ctx.name == "pubmed"
        else "Edge cases between similar banking intents (fees vs declined vs transfer delays)."
    )
    return (
        f"TASK:\n{base_problem}\n\n"
        "CRITICAL COGNITIVE INSTRUCTIONS:\n"
        "Before the JSON output, use internal processing to cross-examine your first instinct:\n"
        "- Two distinct hypotheses per difficult item.\n"
        f"- {boundary}\n"
        "- Reverse-check before committing labels.\n\n"
        "Visible output: ONLY the JSON array from the task (no prose outside JSON)."
    )


def _apply_strategy(strategy: PromptStrategy, base_problem: str, *, ctx: DatasetContext) -> str:
    if strategy == "stripped":
        return wrap_stripped_prompt(base_problem)
    if strategy == "external_cot":
        return wrap_external_cot_prompt(base_problem)
    return wrap_booster_prompt(base_problem, ctx=ctx)


def _build_base_problem(*, ctx: DatasetContext, items: list[BatchItem]) -> str:
    labels_str = ", ".join(json.dumps(l) for l in ctx.allowed_labels)
    payload = {
        "allowed_labels": ctx.allowed_labels,
        "items": [{"id": it.id, "text": it.text} for it in items],
        "output_schema": [{"id": "string", "labels": ["string", "..."]}],
    }
    return (
        f"Classify {ctx.task_title}.\n"
        f"Domain: {ctx.domain_line}.\n"
        f"Allowed labels: [{labels_str}]\n"
        "For each item id, return ALL labels that genuinely apply (0 to N labels).\n"
        "If uncertain between labels, you may return multiple. If none apply, return [].\n"
        "Output ONLY a JSON array: "
        '[{"id": "...", "labels": ["...", ...]}, ...] with one object per input id.\n\n'
        "Items JSON:\n"
        + json.dumps(payload, ensure_ascii=True)
    )


def _system_instruction(strategy: PromptStrategy, *, ctx: DatasetContext) -> str:
    base = (
        f"You are a strict multi-label classifier for {ctx.domain_line}.\n"
        "Never invent labels outside the allowed set.\n"
        "Confusion policy: return [] if no label fits; return multiple labels only when genuinely ambiguous."
    )
    if strategy == "external_cot":
        return (
            base
            + "\nYou may write reasoning before the answer, but the substring after the exact "
            'prefix "Final Answer:" must be only a JSON array matching the schema — no prose, '
            "no markdown code fences."
        )
    return base + "\nVisible reply must be only the JSON array (no markdown fences, no commentary)."


def _thinking_config(spec: TierSpec) -> ThinkingConfig | None:
    if spec.thinking_budget is not None:
        return ThinkingConfig(
            thinking_budget=int(spec.thinking_budget),
            include_thoughts=bool(spec.include_thoughts),
        )
    if spec.thinking_level:
        level_map = {"low": ThinkingLevel.LOW, "high": ThinkingLevel.HIGH}
        lvl = level_map.get(spec.thinking_level.lower())
        if lvl is not None:
            return ThinkingConfig(
                thinking_level=lvl,
                include_thoughts=bool(spec.include_thoughts),
            )
    return None


def _generate_config(spec: TierSpec) -> GenerateContentConfig:
    kwargs: dict[str, Any] = {"temperature": 0.0}
    tc = _thinking_config(spec)
    if tc is not None:
        kwargs["thinking_config"] = tc
    return GenerateContentConfig(**kwargs)


def _telemetry(resp: Any, *, elapsed_s: float) -> dict[str, Any]:
    um = getattr(resp, "usage_metadata", None)
    usage, raw_um = _usage_from_response(resp)
    prompt_t = getattr(um, "prompt_token_count", None) if um is not None else None
    cand_t = getattr(um, "candidates_token_count", None) if um is not None else None
    thoughts_t = getattr(um, "thoughts_token_count", None) if um is not None else None
    if thoughts_t is None and um is not None:
        thoughts_t = getattr(um, "thinking_token_count", None)
    visible = None
    if cand_t is not None:
        visible = int(cand_t) - int(thoughts_t or 0)
    return {
        "elapsed_s": round(elapsed_s, 4),
        "prompt_token_count": prompt_t,
        "candidates_token_count": cand_t,
        "thoughts_token_count": thoughts_t,
        "visible_answer_tokens_est": visible,
        "usage_in": usage.in_tokens,
        "usage_out": usage.out_tokens,
        "usage_metadata": raw_um,
    }


def _merge_telemetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    prompt_t = sum(int(r["prompt_token_count"] or 0) for r in rows)
    cand_t = sum(int(r["candidates_token_count"] or 0) for r in rows)
    thoughts_t = sum(int(r["thoughts_token_count"] or 0) for r in rows)
    elapsed = sum(float(r["elapsed_s"] or 0) for r in rows)
    return {
        "elapsed_s": round(elapsed, 4),
        "n_batch_calls": len(rows),
        "prompt_token_count": prompt_t,
        "candidates_token_count": cand_t,
        "thoughts_token_count": thoughts_t or None,
        "visible_answer_tokens_est": cand_t - thoughts_t if cand_t else None,
        "per_batch": rows,
    }


def _norm_pubmed_label(value: object) -> str:
    if value is None:
        return "background"
    if isinstance(value, (int, float)) and int(value) == value:
        names = PUBMED_LABELS
        idx = int(value)
        if 0 <= idx < len(names):
            return names[idx]
    text = str(value).strip().lower()
    if text in ("conclusion", "concl"):
        return "conclusions"
    if text == "method":
        return "methods"
    if text == "result":
        return "results"
    return text if text in PUBMED_LABELS else "background"


def load_five_pubmed_samples(*, seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    raw = load_dataset("armanc/pubmed-rct20k")
    test = raw["test"]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx in range(len(test)):
        lab = _norm_pubmed_label(test[idx]["label"])
        if lab in seen:
            continue
        seen.add(lab)
        rows.append(
            {
                "id": str(len(rows)),
                "sample_id": None,
                "text": str(test[idx]["text"]),
                "gold_label": lab,
                "source_is_confusing": None,
                "prior_pred_labels": None,
            }
        )
        if len(rows) >= 5:
            break
    if len(rows) < 5:
        import pandas as pd

        df = pd.DataFrame(test).sample(n=5, random_state=seed).reset_index(drop=True)
        rows = [
            {
                "id": str(i),
                "sample_id": None,
                "text": str(df.loc[i, "text"]),
                "gold_label": _norm_pubmed_label(df.loc[i, "label"]),
                "source_is_confusing": None,
                "prior_pred_labels": None,
            }
            for i in range(5)
        ]
    return rows


def load_banking_confusing_samples(
    path: Path,
    *,
    n: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Rows from a multilabel_confusion_probe run where is_confusing is true."""
    if not path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON array in {path}")
    confusing = [r for r in raw if isinstance(r, dict) and r.get("is_confusing") is True]
    confusing.sort(key=lambda r: str(r.get("sample_id", "")))
    if len(confusing) < n:
        raise ValueError(f"Only {len(confusing)} is_confusing rows in {path}; need {n}")
    if len(confusing) > n:
        import random

        rng = random.Random(seed)
        confusing = rng.sample(confusing, n)
        confusing.sort(key=lambda r: str(r.get("sample_id", "")))
    rows: list[dict[str, Any]] = []
    for i, rec in enumerate(confusing):
        sid = str(rec.get("sample_id", f"row_{i}"))
        prior = rec.get("pred_labels")
        if prior is None and isinstance(rec.get("raw"), dict):
            prior = rec["raw"].get("pred_labels")
        rows.append(
            {
                "id": str(i),
                "sample_id": sid,
                "text": str(rec.get("text", "")),
                "gold_label": str(rec.get("true_label", "")),
                "source_is_confusing": True,
                "prior_pred_labels": prior if isinstance(prior, list) else None,
                "prior_n_pred_labels": rec.get("n_pred_labels"),
            }
        )
    return rows


def _score_items(
    *,
    items: list[BatchItem],
    gold_by_id: dict[str, str],
    parsed: dict[str, list[str]],
    sample_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    per_item: list[dict[str, Any]] = []
    for it in items:
        labels = parsed.get(it.id, [])
        confusing, primary = confusion_from_label_lists(labels)
        kind = multilabel_confusion_kind(labels)
        gold = gold_by_id.get(it.id)
        meta = sample_meta.get(it.id, {})
        per_item.append(
            {
                "id": it.id,
                "sample_id": meta.get("sample_id"),
                "text": it.text,
                "gold_label": gold,
                "pred_labels": labels,
                "n_pred_labels": len(labels),
                "confusion_kind": kind,
                "pred_primary": primary,
                "is_confusing": confusing,
                "gold_in_pred_set": gold in labels if gold else None,
                "top1_correct": primary == gold if primary and gold else False,
                "source_is_confusing": meta.get("source_is_confusing"),
                "prior_pred_labels": meta.get("prior_pred_labels"),
            }
        )
    return per_item


def _aggregate_metrics(per_item: list[dict[str, Any]]) -> dict[str, Any]:
    n_items = len(per_item)
    n_top1 = sum(1 for r in per_item if r["top1_correct"])
    n_gold_in = sum(1 for r in per_item if r["gold_in_pred_set"])
    n_none = sum(1 for r in per_item if r["confusion_kind"] == "none")
    n_multi = sum(1 for r in per_item if r["confusion_kind"] == "multi")
    n_single = sum(1 for r in per_item if r["confusion_kind"] == "single")
    n_conf = sum(1 for r in per_item if r["is_confusing"])
    return {
        "n_items": n_items,
        "top1_accuracy": n_top1 / n_items if n_items else 0.0,
        "gold_in_set_rate": n_gold_in / n_items if n_items else 0.0,
        "n_confusing": n_conf,
        "confusion_rate": n_conf / n_items if n_items else 0.0,
        "n_pred_labels_zero": n_none,
        "n_pred_labels_multi": n_multi,
        "n_pred_labels_single": n_single,
        "rate_pred_zero": n_none / n_items if n_items else 0.0,
        "rate_pred_multi": n_multi / n_items if n_items else 0.0,
    }


def _call_batch(
    client: genai.Client,
    spec: TierSpec,
    *,
    ctx: DatasetContext,
    batch_items: list[BatchItem],
    batch_index: int,
) -> tuple[dict[str, list[str]], dict[str, Any], str, str | None, list[dict[str, Any]]]:
    base = _build_base_problem(ctx=ctx, items=batch_items)
    user = _apply_strategy(spec.prompt_strategy, base, ctx=ctx)
    system = _system_instruction(spec.prompt_strategy, ctx=ctx)
    cfg = _generate_config(spec)
    cfg_dump = cfg.model_dump(exclude_none=True)
    cfg_dump["system_instruction"] = system
    cfg2 = GenerateContentConfig.model_validate(cfg_dump)
    t0 = time.perf_counter()
    resp = client.models.generate_content(model=spec.model_id, contents=user, config=cfg2)
    elapsed = time.perf_counter() - t0
    answer_text, thought_text, parts = _parts_from_response(resp)
    parsed = parse_multilabel_batch(answer_text, allowed_labels=ctx.allowed_labels)
    tel = _telemetry(resp, elapsed_s=elapsed)
    tel["batch_index"] = batch_index
    tel["batch_size"] = len(batch_items)
    return parsed, tel, answer_text, thought_text, parts


def run_tier(
    client: genai.Client,
    spec: TierSpec,
    *,
    ctx: DatasetContext,
    samples: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    items = [BatchItem(id=s["id"], text=s["text"]) for s in samples]
    gold_by_id = {s["id"]: str(s["gold_label"]) for s in samples}
    meta_by_id = {s["id"]: s for s in samples}

    batches: list[list[BatchItem]] = []
    for start in range(0, len(items), batch_size):
        batches.append(items[start : start + batch_size])

    all_parsed: dict[str, list[str]] = {}
    tel_rows: list[dict[str, Any]] = []
    answer_previews: list[str] = []
    thought_previews: list[str | None] = []

    for bi, batch in enumerate(batches):
        logger.info("  batch {}/{} (n={})", bi + 1, len(batches), len(batch))
        parsed, tel, answer_text, thought_text, parts = _call_batch(
            client, spec, ctx=ctx, batch_items=batch, batch_index=bi
        )
        all_parsed.update(parsed)
        tel_rows.append(tel)
        answer_previews.append(answer_text[:2000])
        thought_previews.append(thought_text[:1500] if thought_text else None)
        _ = parts

    per_item = _score_items(
        items=items,
        gold_by_id=gold_by_id,
        parsed=all_parsed,
        sample_meta=meta_by_id,
    )
    metrics = _aggregate_metrics(per_item)
    metrics["n_batch_requests"] = len(batches)
    metrics["batch_size"] = batch_size

    return {
        "tier": spec.tier,
        "tier_name": spec.name,
        "suite": ctx.name,
        "model_id": spec.model_id,
        "prompt_strategy": spec.prompt_strategy,
        "thinking_budget": spec.thinking_budget,
        "thinking_level": spec.thinking_level,
        "notes": spec.notes,
        "telemetry": _merge_telemetry(tel_rows),
        "answer_text_preview": "\n---\n".join(answer_previews)[:8000],
        "thought_text_preview": "\n---\n".join(t for t in thought_previews if t)[:4000] or None,
        "metrics": metrics,
        "items": per_item,
    }


def _resolve_tier_plan(tier_arg: str) -> tuple[list[int], str]:
    """Return (tier ids 1-6, suite name: pubmed | banking)."""
    key = tier_arg.strip().lower()
    if key == "banking-all":
        return list(range(1, 7)), "banking"
    if key == "all":
        return list(range(1, 7)), "pubmed"
    try:
        n = int(key)
    except ValueError as e:
        raise SystemExit(
            f"Unknown --tier {tier_arg!r}. Use 1-6, all, or banking-all."
        ) from e
    if n not in range(1, 7):
        raise SystemExit(f"Tier must be 1-6, got {n}")
    return [n], "pubmed"


def main() -> None:
    ap = argparse.ArgumentParser(description="Reasoning spectrum × multi-label batches.")
    _ = ap.add_argument(
        "--tier",
        type=str,
        default="1",
        help="Tier 1-6, all (pubmed 5×1 batch), or banking-all (20 confusing × 4 batches × 6 tiers).",
    )
    _ = ap.add_argument(
        "--suite",
        type=str,
        default=None,
        choices=("pubmed", "banking"),
        help="Override dataset suite (default: pubmed, or banking when --tier banking-all).",
    )
    _ = ap.add_argument(
        "--predictions-path",
        type=Path,
        default=None,
        help="Banking: full_predictions.json with is_confusing rows (default: BANKING_CONFUSING_PRED_PATH).",
    )
    _ = ap.add_argument("--seed", type=int, default=42)
    _ = ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    _ = ap.add_argument("--out-dir", type=Path, default=None)
    _ = ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tier_ids, suite_from_tier = _resolve_tier_plan(str(args.tier))
    suite = str(args.suite or suite_from_tier).lower()
    matrix = _tier_matrix()
    batch_size = max(1, int(args.batch_size))

    if suite == "banking":
        ctx = _banking_context()
        pred_path = Path(
            os.getenv("BANKING_CONFUSING_PRED_PATH", str(DEFAULT_BANKING_PRED_PATH))
        )
        if args.predictions_path is not None:
            pred_path = args.predictions_path
        pred_path = pred_path.resolve()
        samples = load_banking_confusing_samples(pred_path, n=20, seed=int(args.seed))
        suite_label = "banking_confusing20"
    else:
        ctx = _pubmed_context()
        samples = load_five_pubmed_samples(seed=int(args.seed))
        suite_label = "pubmed5"
        if len(samples) != batch_size and suite == "pubmed":
            batch_size = len(samples)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else _repo_root() / "results" / "reasoning_spectrum" / f"{stamp}_{suite_label}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "suite": suite_label,
        "dataset": ctx.name,
        "n_items": len(samples),
        "batch_size": batch_size,
        "n_batch_requests_per_tier": (len(samples) + batch_size - 1) // batch_size,
        "n_tiers": len(tier_ids),
        "total_api_calls_est": len(tier_ids) * ((len(samples) + batch_size - 1) // batch_size),
        "samples": samples,
        "tiers_planned": [matrix[t].name for t in tier_ids],
        "allowed_labels": ctx.allowed_labels,
    }
    if suite == "banking":
        manifest["source_predictions_path"] = str(pred_path)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "eval_samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")

    if args.dry_run:
        for t in tier_ids:
            logger.info("Tier {}: {}", t, json.dumps(asdict(matrix[t]), indent=2))
        logger.info(
            "Dry run: suite={} items={} batches/tier={} tiers={} est_calls={}",
            suite_label,
            len(samples),
            manifest["n_batch_requests_per_tier"],
            len(tier_ids),
            manifest["total_api_calls_est"],
        )
        return

    client = _make_google_genai_client(api_key=_api_key(), http_retry_attempts=5)
    results: list[dict[str, Any]] = []

    for t in tier_ids:
        spec = matrix[t]
        logger.info(
            "=== Tier {} ({}) model={} suite={} ===",
            t,
            spec.name,
            spec.model_id,
            suite_label,
        )
        try:
            row = run_tier(
                client,
                spec,
                ctx=ctx,
                samples=samples,
                batch_size=batch_size,
            )
        except Exception as e:
            logger.exception("Tier {} failed: {}", t, e)
            row = {
                "tier": t,
                "tier_name": spec.name,
                "suite": suite_label,
                "error": str(e),
            }
        results.append(row)
        path = out_dir / f"tier_{t:02d}_{spec.name}.json"
        path.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
        if "telemetry" in row:
            tel = row["telemetry"]
            m = row.get("metrics") or {}
            logger.info(
                "Tier {} done: top1_acc={:.2f} gold_in_set={:.2f} "
                "confusion_rate={:.2f} (zero={} multi={}) batches={} thoughts_t={} elapsed_s={}",
                t,
                m.get("top1_accuracy", 0),
                m.get("gold_in_set_rate", 0),
                m.get("confusion_rate", 0),
                m.get("n_pred_labels_zero"),
                m.get("n_pred_labels_multi"),
                tel.get("n_batch_calls"),
                tel.get("thoughts_token_count"),
                tel.get("elapsed_s"),
            )

    summary = {
        "suite": suite_label,
        "n_items": len(samples),
        "batch_size": batch_size,
        "tiers": [
            {
                "tier": r.get("tier"),
                "name": r.get("tier_name"),
                "model_id": r.get("model_id"),
                "top1_accuracy": (r.get("metrics") or {}).get("top1_accuracy"),
                "gold_in_set_rate": (r.get("metrics") or {}).get("gold_in_set_rate"),
                "confusion_rate": (r.get("metrics") or {}).get("confusion_rate"),
                "n_pred_labels_zero": (r.get("metrics") or {}).get("n_pred_labels_zero"),
                "n_pred_labels_multi": (r.get("metrics") or {}).get("n_pred_labels_multi"),
                "n_pred_labels_single": (r.get("metrics") or {}).get("n_pred_labels_single"),
                "n_batch_requests": (r.get("metrics") or {}).get("n_batch_requests"),
                "thoughts_token_count": (r.get("telemetry") or {}).get("thoughts_token_count"),
                "elapsed_s": (r.get("telemetry") or {}).get("elapsed_s"),
                "error": r.get("error"),
            }
            for r in results
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote results to {}", out_dir)


if __name__ == "__main__":
    main()
