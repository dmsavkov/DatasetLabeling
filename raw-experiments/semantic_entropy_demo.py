"""
Concise, self-contained demo: semantic entropy of LLM responses.

Architecture:
sample -> 5 answers -> NLI-equivalence clustering (A⇒B and B⇒A) -> entropy

Notes / assumptions:
- NLI equivalence is defined as: answers i and j are equivalent iff
  entail(i→j) >= threshold AND entail(j→i) >= threshold.
- Probabilities are uniform per answer: 5 answers => each has p=0.2. Cluster probabilities are sums.

Run:
  - set env var GEMINI_API_KEY (or GOOGLE_API_KEY / OPENAI_API_KEY pointing to same key)
  - python raw-experiments/semantic_entropy_demo.py
  
Exploration conclusions: Overall useless.
  - NLI may be wrong. Minor differences in expalanations (even when meanign is preserved) lead to different clusters.
  - LLM answers are almost identical. 
  - More emotions didnt' make the model less consistent. Maybe, emotions are too distinguishable? Still, model makes confident mistakes in some cases.
  
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Literal, cast
from collections.abc import Callable
from typing import Final

from openai import AsyncOpenAI
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from sentence_transformers import CrossEncoder

"""
Unified LLM client resolution: exact `model_id` → async HTTP client.

Routing is table-driven (`LLMClientRegistry`). Callers use only `get_async_llm_client`;
there is no separate “OpenAI vs Hugging Face” API surface—unknown ids fail fast.

New providers or families are added by registering additional `(model_id, factory)` pairs.
"""
_GOOGLE_GENAI_OPENAI_BASE_URL: Final[str] = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)

_DEFAULT_MAX_RETRIES: Final[int] = 20


def _google_genai_api_key() -> str:
    for env in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(env)
        if v:
            return v
    raise ValueError(
        "No API key found for Google Generative AI. Set GEMINI_API_KEY or GOOGLE_API_KEY. "
        + "(OPENAI_API_KEY is accepted if it holds the same key)."
    )


def _make_google_openai_compat_client() -> AsyncOpenAI:
    """Factory for models using Google’s OpenAI-compatible Generative Language endpoint."""

    return AsyncOpenAI(
        api_key=_google_genai_api_key(),
        base_url=_GOOGLE_GENAI_OPENAI_BASE_URL,
        max_retries=_DEFAULT_MAX_RETRIES,
    )


@dataclass
class LLMClientRegistry:
    """
    Maps exact model id strings to zero-arg factories that build transport clients.

    Factories may share implementation when multiple ids use the same endpoint/auth;
    registration still lists each supported id explicitly.
    """

    _factories: dict[str, Callable[[], AsyncOpenAI]] = field(default_factory=dict)

    def register(self, model_id: str, factory: Callable[[], AsyncOpenAI]) -> None:
        mid = model_id.strip()
        if mid in self._factories:
            raise ValueError(f"Duplicate model_id registration: {mid!r}")
        self._factories[mid] = factory

    def get_client(self, model_id: str) -> AsyncOpenAI:
        mid = model_id.strip()
        if mid not in self._factories:
            raise ValueError(
                f"Unknown model_id {mid!r}. Registered ids: {sorted(self._factories)}"
            )
        return self._factories[mid]()

    def registered_ids(self) -> frozenset[str]:
        return frozenset(self._factories.keys())


default_registry = LLMClientRegistry()


def _bootstrap_default_registry() -> None:
    _google = _make_google_openai_compat_client
    for mid in (
        "gemini-3.1-flash-lite-preview",
        "gemma-3-4b-it",
    ):
        default_registry.register(mid, _google)


_bootstrap_default_registry()


def get_async_llm_client(model_id: str) -> AsyncOpenAI:
    """
    Return an async LLM client for the given exact registered `model_id`.

    This is the single entrypoint used by experiment code; routing is handled by
    `default_registry`. Lifecycle: typical HTTP usage does not require explicit close.
    """
    return default_registry.get_client(model_id)



Label = Literal["anger", "fear", "joy", "sadness", "surprise"]
ALLOWED_LABELS: tuple[Label, ...] = ("anger", "fear", "joy", "sadness", "surprise")


@dataclass(frozen=True, slots=True)
class Sample:
    id: str
    difficulty: Literal["easy", "medium", "hard"]
    text: str
    target_label: Label
    allowed_labels: tuple[Label, ...] = ALLOWED_LABELS


@dataclass(frozen=True, slots=True)
class LLMAnswer:
    raw_text: str
    label: Label | None
    explanation: str | None

    def canonical_text(self) -> str:
        # Used for semantic clustering. Keep stable formatting to reduce spurious splits.
        lbl = self.label or "unknown"
        expl = (self.explanation or "").strip()
        return f"label={lbl}\nexplanation={expl}"


def _make_samples() -> list[Sample]:
    # Harder, subjective emotion classification (5-way).
    return [
        Sample(
            id="s1",
            difficulty="medium",
            target_label="sadness",
            text=(
                "I keep rereading the last message they sent and it just feels quiet now. "
                "Everyone says time helps, but today it doesn't."
            ),
        ),
        Sample(
            id="s2",
            difficulty="hard",
            target_label="anger",
            text=(
                "They apologized, again, and somehow I'm the one expected to be 'understanding' "
                "while nothing changes. I'm so tired of pretending this is fine."
            ),
        ),
        Sample(
            id="s3",
            difficulty="hard",
            target_label="fear",
            text=(
                "My boss said 'we'll talk tomorrow' with that smile. Now I'm replaying every "
                "mistake I've made this month and I can't sleep."
            ),
        ),
        Sample(
            id="s4",
            difficulty="hard",
            target_label="surprise",
            text=(
                "I opened the door and everyone yelled my name. I genuinely thought I was just "
                "picking up a package. I didn't see any of this coming."
            ),
        ),
        Sample(
            id="s5",
            difficulty="hard",
            target_label="joy",
            text=(
                "I got the email at 2am and I laughed out loud. After months of doubt, it finally "
                "worked. I can't stop smiling."
            ),
        ),
    ]


def _build_prompt(sample: Sample) -> list[ChatCompletionMessageParam]:
    allowed = list(sample.allowed_labels)
    # Force strict JSON to simplify parsing.
    user: ChatCompletionMessageParam = {
        "role": "user",
        "content": (
            "You are labeling short posts. Task: classify the primary emotion expressed.\n"
            f"Allowed labels: {allowed}\n"
            "Return JSON only with keys: label, explanation.\n"
            "Rules:\n"
            "- label must be exactly one of the allowed labels\n"
            "- explanation must be <= 25 words\n\n"
            f"Text: {sample.text}"
        ),
    }
    return [user]


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _extract_json_any(text: str) -> object:
    """
    Extract a JSON object/array from an LLM response.

    Strategy:
    - If there's a fenced block, parse that content first.
    - Otherwise parse the first {...} or [...] span that decodes.
    """
    t = (text or "").strip()
    if not t:
        raise ValueError("Empty response")

    m = _JSON_FENCE_RE.search(t)
    if m:
        candidate = m.group(1).strip()
        return cast(object, json.loads(candidate))

    # Try full text as JSON.
    try:
        return cast(object, json.loads(t))
    except Exception:
        pass

    # Heuristic scan for a JSON object or array.
    first_obj = t.find("{")
    first_arr = t.find("[")
    starts = [i for i in (first_obj, first_arr) if i != -1]
    if not starts:
        raise ValueError("No JSON start token found")

    start = min(starts)
    for end in range(len(t), start + 1, -1):
        if t[end - 1] not in ("}", "]"):
            continue
        snippet = t[start:end]
        try:
            return cast(object, json.loads(snippet))
        except Exception:
            continue
    raise ValueError("Failed to parse JSON from response")


def _parse_answer(raw_text: str) -> LLMAnswer:
    try:
        obj = _extract_json_any(raw_text)
    except Exception as exc:
        return LLMAnswer(raw_text=raw_text, label=None, explanation=f"json_parse_error: {exc}")

    if not isinstance(obj, dict):
        return LLMAnswer(raw_text=raw_text, label=None, explanation="json_not_object")

    obj_d = cast(dict[str, object], obj)
    label = obj_d.get("label", None)
    explanation = obj_d.get("explanation", None)

    if label not in ALLOWED_LABELS:
        label_out: Label | None = None
    else:
        label_out = label

    expl_out = explanation if isinstance(explanation, str) else None
    return LLMAnswer(raw_text=raw_text, label=label_out, explanation=expl_out)


def _entailment_index(nli: CrossEncoder) -> int:
    """
    Determine which logit corresponds to entailment.

    For `cross-encoder/nli-distilroberta-base` the common ordering is:
      0=contradiction, 1=neutral, 2=entailment
    We still try to infer this from id2label when available.
    """
    try:
        model = getattr(nli, "model", None)
        cfg = getattr(model, "config", None) if model is not None else None
        id2label = getattr(cfg, "id2label", None) if cfg is not None else None
        if isinstance(id2label, dict):
            for k, v in id2label.items():
                if isinstance(v, str) and "entail" in v.lower():
                    return int(k)
    except Exception:
        pass
    return 2


def _clusters_by_bi_entailment(
    texts: list[str],
    *,
    nli: CrossEncoder,
    threshold: float,
    batch_size: int = 32,
) -> tuple[list[list[int]], list[list[float]]]:
    """
    Correct "equivalence" clustering:
    - Compute entailment probs for all ordered pairs (i→j), i!=j.
    - Put an undirected edge i~j iff entail(i→j) and entail(j→i) are both >= threshold.
    - Connected components in that undirected graph are clusters.

    Returns (clusters, entail_prob_matrix).
    """
    n = len(texts)
    if n == 0:
        return ([], [])
    if n == 1:
        return ([[0]], [[1.0]])

    entail_idx = _entailment_index(nli)
    # Build ordered pairs (i, j) for i!=j.
    pairs: list[tuple[str, str]] = []
    ij: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pairs.append((texts[i], texts[j]))
            ij.append((i, j))

    # CrossEncoder.predict returns shape (num_pairs, num_labels) or (num_pairs,)
    scores = nli.predict(
        pairs,
        batch_size=int(batch_size),
        convert_to_numpy=True,
        show_progress_bar=False,
        apply_softmax=True,
    )

    # Normalize to entailment probabilities.
    ent: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for idx, (i, j) in enumerate(ij):
        row = scores[idx]
        if isinstance(row, (float, int)):
            # Not expected for 3-way NLI, but handle gracefully.
            ent[i][j] = float(row)
            continue
        ent[i][j] = float(row[int(entail_idx)])
    for i in range(n):
        ent[i][i] = 1.0

    # Build undirected adjacency by bi-entailment.
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if ent[i][j] >= float(threshold) and ent[j][i] >= float(threshold):
                adj[i].append(j)
                adj[j].append(i)

    # Connected components.
    seen: set[int] = set()
    clusters: list[list[int]] = []
    for i in range(n):
        if i in seen:
            continue
        q = [i]
        seen.add(i)
        comp: list[int] = []
        while q:
            cur = q.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comp.sort()
        clusters.append(comp)

    clusters.sort(key=lambda c: (-len(c), c))
    return (clusters, ent)


def _entropy_from_uniform_clusters(cluster_sizes: list[int], *, per_answer_prob: float) -> tuple[float, float]:
    """
    Return entropy in nats and bits.
    """
    ps = [s * per_answer_prob for s in cluster_sizes if s > 0]
    h_nats = -sum(p * math.log(p) for p in ps if p > 0.0)
    if abs(h_nats) < 1e-12:
        h_nats = 0.0
    h_bits = h_nats / math.log(2.0) if h_nats > 0.0 else 0.0
    return h_nats, h_bits


async def _aget_one_answer(
    *,
    model_id: str,
    messages: list[ChatCompletionMessageParam],
    temperature: float,
    max_tokens: int,
) -> str:
    client = get_async_llm_client(model_id)
    resp = await client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    if not resp.choices or not resp.choices[0].message or resp.choices[0].message.content is None:
        return ""
    return str(resp.choices[0].message.content)


async def _aget_k_answers(
    *,
    model_id: str,
    sample: Sample,
    k: int,
    temperature: float,
    max_tokens: int,
    max_concurrency: int,
) -> list[LLMAnswer]:
    messages = _build_prompt(sample)
    sem = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def run_one(i: int) -> LLMAnswer:
        async with sem:
            raw = await _aget_one_answer(
                model_id=model_id,
                messages=messages,
                temperature=float(temperature) + 0.05 * float(i),
                max_tokens=max_tokens,
            )
        return _parse_answer(raw)

    return await asyncio.gather(*(run_one(i) for i in range(int(k))))


async def main() -> None:
    model_id = "gemma-3-4b-it"
    _ = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")

    samples = _make_samples()
    k = 5
    per_answer_prob = 1.0 / float(k)

    # NLI-based equivalence graph.
    nli_model_id = "cross-encoder/nli-distilroberta-base"
    nli = CrossEncoder(nli_model_id)
    entail_threshold = 0.80

    logger.info("Model: {}", model_id)
    logger.info("NLI model: {} (bi-entail threshold={})", nli_model_id, entail_threshold)
    logger.info("Answers per sample: {} (uniform p={})", k, per_answer_prob)

    per_sample_bits: list[float] = []
    for s in samples:
        logger.info("----")
        logger.info("Sample {} ({})", s.id, s.difficulty)
        logger.info("Text: {}", s.text)

        answers = await _aget_k_answers(
            model_id=model_id,
            sample=s,
            k=k,
            temperature=0.7,
            max_tokens=200,
            max_concurrency=5,
        )

        canon = [a.canonical_text() for a in answers]
        clusters, _ent = _clusters_by_bi_entailment(canon, nli=nli, threshold=entail_threshold, batch_size=32)
        cluster_sizes = [len(c) for c in clusters]

        h_nats, h_bits = _entropy_from_uniform_clusters(cluster_sizes, per_answer_prob=per_answer_prob)
        per_sample_bits.append(float(h_bits))

        logger.info("Clusters: {} (sizes={})", len(clusters), cluster_sizes)
        logger.info("Semantic entropy: {:.4f} nats / {:.4f} bits", h_nats, h_bits)

        for ci, idxs in enumerate(clusters, start=1):
            labels = [answers[i].label for i in idxs]
            logger.info("  Cluster {}: idxs={}, labels={}", ci, idxs, labels)
            # Show one representative canonical text.
            rep = canon[idxs[0]]
            logger.info("    rep:\n{}", rep)

    if per_sample_bits:
        avg_bits = float(sum(per_sample_bits) / len(per_sample_bits))
        logger.info("====")
        logger.info("Final semantic entropy (avg over {} samples): {:.4f} bits", len(per_sample_bits), avg_bits)


if __name__ == "__main__":
    asyncio.run(main())

