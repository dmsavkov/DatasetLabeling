from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .baseline import BatchItem, build_llm_batch_messages


PromptBuilder = Callable[
    ...,
    list[dict[str, str]],
]


@dataclass(frozen=True, slots=True)
class PromptSpec:
    prompt_id: str
    build_messages: Callable[
        ...,
        list[dict[str, str]],
    ]


def _baseline_v1(
    *,
    allowed_labels: list[str],
    items: list[BatchItem],
    few_shot: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    return build_llm_batch_messages(allowed_labels=allowed_labels, items=items, few_shot=few_shot)


def _engineered_brief_rationale_v1(
    *,
    allowed_labels: list[str],
    items: list[BatchItem],
    few_shot: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """
    “CoT-like” variant without breaking strict JSON:
    - nudges the model to reason silently
    - still requires ONLY a JSON array output
    """
    msgs = build_llm_batch_messages(allowed_labels=allowed_labels, items=items, few_shot=few_shot)
    # msgs[0] is system, msgs[1] is user per build_llm_batch_messages contract
    sys = msgs[0]["content"]
    msgs[0]["content"] = (
        sys
        + "\n\nThink step-by-step silently. If needed, form a brief internal rationale per item, "
        + "but output must remain ONLY the JSON array schema described above."
    )
    return msgs


PROMPTS: dict[str, PromptSpec] = {
    "baseline_v1": PromptSpec(prompt_id="baseline_v1", build_messages=_baseline_v1),
    "engineered_brief_rationale_v1": PromptSpec(
        prompt_id="engineered_brief_rationale_v1",
        build_messages=_engineered_brief_rationale_v1,
    ),
}


def get_prompt(prompt_id: str) -> PromptSpec:
    pid = prompt_id.strip()
    if pid not in PROMPTS:
        raise ValueError(f"Unknown prompt_id {pid!r}. Known: {sorted(PROMPTS)}")
    return PROMPTS[pid]

