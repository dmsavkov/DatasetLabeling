from __future__ import annotations

import pytest

from src.prompts.baseline import BatchItem
from src.prompts.registry import get_prompt


def test_prompt_registry_unknown_id_raises() -> None:
    with pytest.raises(ValueError, match="Unknown prompt_id"):
        _ = get_prompt("nope")


def test_prompt_registry_baseline_builds_messages() -> None:
    p = get_prompt("baseline_v1")
    msgs = p.build_messages(
        allowed_labels=["a", "b"],
        items=[BatchItem(id="0", text="hello")],
        few_shot=[("x", "a")],
    )
    assert isinstance(msgs, list)
    assert msgs and msgs[0]["role"] == "system"


def test_prompt_registry_engineered_variant_mutates_system_message() -> None:
    base = get_prompt("baseline_v1").build_messages(
        allowed_labels=["a", "b"],
        items=[BatchItem(id="0", text="hello")],
        few_shot=None,
    )
    eng = get_prompt("engineered_brief_rationale_v1").build_messages(
        allowed_labels=["a", "b"],
        items=[BatchItem(id="0", text="hello")],
        few_shot=None,
    )
    assert "Think step-by-step silently" not in base[0]["content"]
    assert "Think step-by-step silently" in eng[0]["content"]

