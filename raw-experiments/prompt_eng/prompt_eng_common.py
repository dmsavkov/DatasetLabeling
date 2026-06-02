"""
Lightweight shared helpers for prompt_eng exploratory scripts.

Keeps imports side-effect free except dotenv. Dataset loading hits Hugging Face
only when you call load_hf_pubmed_splits().

Results layout (default): <save_root>/results/raw/prompt_eng/<experiment_slug>/<run_id>/
  Same tree regardless of experiment; each run gets its own timestamped folder only.
  Optional: *_classification_report.json when metrics include sklearn dict report.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from datasets import load_dataset
from loguru import logger
from openai import OpenAI
from sklearn.model_selection import train_test_split

try:
    import dotenv

    dotenv.load_dotenv()
except Exception:
    pass

# One-line stderr logging; override with LOG_LEVEL=DEBUG
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level=os.getenv("LOG_LEVEL", "INFO"),
)

# --- Secrets / paths ---


def secret(name: str) -> str | None:
    try:
        from google.colab import userdata

        v = userdata.get(name)
        if v:
            return str(v)
    except Exception:
        pass
    v = os.getenv(name)
    return str(v) if v else None


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve())

# Retries: experimentation-rules — preliminary cheap, main evaluation higher
PRELIM_MAX_RETRIES = int(os.getenv("PRELIM_MAX_RETRIES", "3"))
MAIN_MAX_RETRIES = int(os.getenv("MAIN_MAX_RETRIES", "25"))

PREDICTIONS_EXPORT_MAX = int(os.getenv("PREDICTIONS_EXPORT_MAX", "200"))


def resolve_save_root() -> Path:
    try:
        from google.colab import drive  # noqa: F401
    except Exception:
        pass
    if Path("/content/drive/MyDrive").exists():
        return Path("/content/drive/MyDrive/data_selection_showdown")
    return PROJECT_ROOT


SAVE_ROOT = resolve_save_root()
DEFAULT_SEED = int(os.getenv("SEED", "42"))

# Canonical PubMed-RCT label mapping (HF armanc/pubmed-rct20k)
PUBMED_LABEL_NAMES = ["background", "conclusions", "methods", "objective", "results"]
PUBMED_ID2LABEL = {idx: name for idx, name in enumerate(PUBMED_LABEL_NAMES)}
PUBMED_LABEL2ID = {name: idx for idx, name in PUBMED_ID2LABEL.items()}

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-3.1-flash-lite-preview")
GEMMA26 = "gemma-4-26b-a4b-it"
GOOGLE_OPENAI_BASE_URL = os.getenv(
    "GOOGLE_OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)

GOOGLE_API_KEY = secret("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

HF_TOKEN = secret("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Prompt_eng artifacts: always under save_root/results/raw/prompt_eng/ (never replace whole results/raw).
# Override only with explicit PROMPT_ENG_RESULTS_ROOT pointing at that leaf folder, not results/ alone.
PROMPT_ENG_RESULTS_ROOT = Path(
    os.getenv("PROMPT_ENG_RESULTS_ROOT", str(SAVE_ROOT / "results" / "raw" / "prompt_eng"))
)
PROMPT_ENG_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
VALID_LABELS = ["background", "conclusions", "methods", "objective", "results"]


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "yes")


def common_run_settings() -> dict[str, Any]:
    """Snapshot of env-driven knobs for metadata JSON."""
    keys = [
        "SEED",
        "DEFAULT_MODEL",
        "BATCH_SIZE",
        "PRELIM_MAX_RETRIES",
        "MAIN_MAX_RETRIES",
        "GOOGLE_OPENAI_BASE_URL",
        "EVAL_SIZE",
        "FEW_SHOT_SIZE",
        "PRELIM_N",
        "SKIP_PRELIMINARY",
        "PRELIM_ONLY",
        "LOG_LEVEL",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        if k in os.environ:
            out[k] = os.environ[k]
    out["PRELIM_MAX_RETRIES"] = str(PRELIM_MAX_RETRIES)
    out["MAIN_MAX_RETRIES"] = str(MAIN_MAX_RETRIES)
    out["project_root"] = str(PROJECT_ROOT)
    out["save_root"] = str(SAVE_ROOT)
    out["prompt_eng_results_root_resolved"] = str(PROMPT_ENG_RESULTS_ROOT)
    return out


def begin_run(experiment_slug: str) -> Path:
    """Create results/raw/prompt_eng/<slug>/<run_id>/ and return that path."""
    run_id = now_stamp()
    run_dir = PROMPT_ENG_RESULTS_ROOT / experiment_slug / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: {}", run_dir)
    manifest = {
        "experiment_slug": experiment_slug,
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "settings": common_run_settings(),
        "save_root": str(SAVE_ROOT),
        "prompt_eng_results_root": str(PROMPT_ENG_RESULTS_ROOT),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return run_dir


def save_phase(
    run_dir: Path,
    phase: Literal["preliminary", "full"],
    *,
    metrics: dict[str, Any],
    settings: dict[str, Any] | None = None,
    predictions: Any | None = None,
    duration_seconds: float | None = None,
    notes: str | None = None,
) -> None:
    """Write phase metadata and optional predictions (capped at PREDICTIONS_EXPORT_MAX)."""
    cr = metrics.get("classification_report")
    crt = metrics.get("classification_report_text")
    # Avoid bloating metadata.json — reports live in dedicated files.
    metrics_compact = {
        k: v
        for k, v in metrics.items()
        if k not in ("classification_report", "classification_report_text")
    }
    payload: dict[str, Any] = {
        "phase": phase,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration_seconds,
        "metrics": metrics_compact,
        "settings": {**(settings or {}), **common_run_settings()},
        "notes": notes,
    }
    meta_path = run_dir / f"{phase}_metadata.json"
    meta_path.write_text(json.dumps(payload, indent=2, default=str))
    try:
        rel = meta_path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = meta_path
    logger.info("Wrote {}", rel)

    if isinstance(cr, dict):
        cr_json = run_dir / f"{phase}_classification_report.json"
        cr_json.write_text(json.dumps(cr, indent=2, default=str))
        logger.info("Wrote {}", cr_json.name)
    if isinstance(crt, str) and crt.strip():
        (run_dir / f"{phase}_classification_report.txt").write_text(crt)
        logger.info("Wrote {}_classification_report.txt", phase)

    if predictions is None:
        return
    try:
        n = len(predictions)  # type: ignore[arg-type]
    except TypeError:
        n = 1
    if n <= PREDICTIONS_EXPORT_MAX:
        pred_path = run_dir / f"{phase}_predictions.json"
        pred_path.write_text(json.dumps(predictions, indent=2, default=str))
        logger.info("Wrote predictions (n={}) -> {}", n, pred_path.name)
    else:
        note = run_dir / f"{phase}_predictions_skipped.txt"
        note.write_text(f"predictions not exported: n={n} > {PREDICTIONS_EXPORT_MAX}\n")
        logger.warning("Skipped predictions export (n={} > {})", n, PREDICTIONS_EXPORT_MAX)


def label_id_from_value(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if value.is_integer():
            return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    lowered = text.lower()
    if lowered in PUBMED_LABEL2ID:
        return PUBMED_LABEL2ID[lowered]
    return None


def label_name_from_value(value: Any) -> str | None:
    label_id = label_id_from_value(value)
    if label_id is not None and label_id in PUBMED_ID2LABEL:
        return PUBMED_ID2LABEL[label_id]
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def get_openai_client(*, max_retries: int | None = None) -> OpenAI:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY missing (env or Colab secrets).")
    mr = MAIN_MAX_RETRIES if max_retries is None else max_retries
    return OpenAI(api_key=GOOGLE_API_KEY, base_url=GOOGLE_OPENAI_BASE_URL, max_retries=mr)


_embedder = None


def get_embedder():
    """Lazy fastembed TextEmbedding (sentence-transformers/all-MiniLM-L6-v2)."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        _embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


def embed_texts(texts: list[str]) -> np.ndarray:
    emb = get_embedder()
    vectors = [np.asarray(vector, dtype=np.float32) for vector in emb.embed(texts)]
    return np.vstack(vectors) if vectors else np.zeros((0, 384), dtype=np.float32)


def load_hf_pubmed_splits(
    seed: int | None = None,
    *,
    eval_size: int | None = None,
    few_shot_size: int | None = None,
    shuffle_eval: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load armanc/pubmed-rct20k: stratified eval slice from test, few-shot from train,
    then shuffle full train/test like the original notebook.
    """
    rng = int(seed) if seed is not None else DEFAULT_SEED
    es = eval_size if eval_size is not None else int(os.getenv("EVAL_SIZE", "100"))
    fs = few_shot_size if few_shot_size is not None else int(os.getenv("FEW_SHOT_SIZE", "30"))
    np.random.seed(rng)

    dataset_name = "armanc/pubmed-rct20k"
    logger.info("Loading HF dataset: {}", dataset_name)
    t0 = time.perf_counter()
    raw = load_dataset(dataset_name)
    logger.info("HF load finished in {:.2f}s", time.perf_counter() - t0)
    test_df = pd.DataFrame(raw["test"])
    train_df = pd.DataFrame(raw["train"])

    _, eval_df = train_test_split(
        test_df, test_size=es, stratify=test_df["label"], random_state=rng
    )
    eval_df = eval_df.copy()
    eval_df["label_name"] = eval_df["label"].apply(label_name_from_value)

    _, few_shot_train_df = train_test_split(
        train_df, test_size=fs, stratify=train_df["label"], random_state=rng
    )
    few_shot_train_df = few_shot_train_df.copy()
    few_shot_train_df["label_name"] = few_shot_train_df["label"].apply(label_name_from_value)

    train_df = train_df.sample(frac=1, random_state=rng).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=rng).reset_index(drop=True)

    if shuffle_eval:
        eval_df = eval_df.sample(frac=1, random_state=rng).reset_index(drop=True)

    logger.info(
        "Splits: eval={} few_shot={} train_rows={} test_rows={}",
        len(eval_df),
        len(few_shot_train_df),
        len(train_df),
        len(test_df),
    )
    return eval_df, few_shot_train_df, train_df, test_df


def require_keys(df: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame missing columns {missing}; have {list(df.columns)}")


def sample_stratified_eval_subset(
    eval_df: pd.DataFrame,
    *,
    n_total: int,
    min_per_class: int,
    label_col: str = "label_name",
    labels: list[str] | None = None,
    seed: int,
) -> pd.DataFrame:
    """
    Stratified sample: at least `min_per_class` rows per label, total `n_total`.
    Remaining slots after the floor are assigned in fixed label order (deterministic).
    """
    labs = list(labels) if labels is not None else list(VALID_LABELS)
    sub = eval_df[eval_df[label_col].astype(str).str.lower().isin(labs)].copy()
    pieces: list[pd.DataFrame] = []
    for lab in labs:
        rows = sub[sub[label_col].astype(str).str.lower() == lab]
        if len(rows) < min_per_class:
            raise ValueError(f"Not enough eval rows for label {lab!r}: have {len(rows)}, need {min_per_class}")
        pieces.append(rows.sample(n=min_per_class, random_state=seed))
    base = pd.concat(pieces, ignore_index=False)
    remaining = n_total - len(base)
    if remaining < 0:
        raise ValueError(f"n_total={n_total} < {min_per_class} * n_labels")
    if remaining > 0:
        rest = sub.drop(index=base.index, errors="ignore")
        if len(rest) < remaining:
            raise ValueError(f"Not enough extra eval rows: need {remaining}, have {len(rest)}")
        extra = rest.sample(n=remaining, random_state=seed + 999)
        base = pd.concat([base, extra], ignore_index=False)
    return base.sample(frac=1, random_state=seed).reset_index(drop=True)


def sample_balanced_train_fewshot(
    train_df: pd.DataFrame,
    n_examples: int,
    *,
    label_col: str = "label_name",
    labels: list[str] | None = None,
    seed: int,
) -> pd.DataFrame:
    """Up to n_examples rows, spread evenly across labels then random fill."""
    labs = list(labels) if labels is not None else list(VALID_LABELS)
    df = train_df.copy()
    if label_col not in df.columns:
        raise KeyError(f"train_df missing {label_col}")
    per = n_examples // len(labs)
    leftover = n_examples % len(labs)
    parts: list[pd.DataFrame] = []
    for i, lab in enumerate(labs):
        take = per + (1 if i < leftover else 0)
        if take <= 0:
            continue
        rows = df[df[label_col].astype(str).str.lower() == lab]
        if len(rows) == 0:
            continue
        parts.append(rows.sample(n=min(take, len(rows)), random_state=seed + i))
    if not parts:
        raise ValueError("No train rows for few-shot sampling")
    out = pd.concat(parts, ignore_index=False)
    if len(out) < n_examples:
        rest = df.drop(index=out.index, errors="ignore")
        need = n_examples - len(out)
        if len(rest) >= need:
            extra = rest.sample(n=need, random_state=seed)
            out = pd.concat([out, extra], ignore_index=False)
    return out.head(n_examples).reset_index(drop=True)


def format_fewshot_block(df: pd.DataFrame, *, text_col: str = "text", label_col: str = "label_name") -> str:
    lines: list[str] = []
    for j, (_, row) in enumerate(df.iterrows()):
        tx = str(row[text_col])[:500]
        lb = str(row[label_col]).lower()
        lines.append(f"Example {j+1} [label={lb}]: {tx}")
    return "\n".join(lines)


def sklearn_classification_reports(
    y_true,
    y_pred,
    *,
    zero_division: int | str = 0,
) -> tuple[dict[str, Any], str]:
    """Sklearn classification_report as dict + printable string."""
    from sklearn.metrics import classification_report

    d = classification_report(y_true, y_pred, output_dict=True, zero_division=zero_division)
    t = classification_report(y_true, y_pred, zero_division=zero_division)
    return d, t
