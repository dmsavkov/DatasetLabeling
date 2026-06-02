# pyright: basic
from __future__ import annotations

import inspect

from src.data_selection import huge_prediction_representatives as hp


def test_huge_prediction_batch_uses_card_label_ids() -> None:
    src = inspect.getsource(hp._run_shuffled_batch_predictions)
    assert "allowed_labels=ctx.label_ids" in src
    assert "allowed_labels=ctx.prompt_labels" not in src


def test_huge_prediction_imports_eval_aligned_few_shot() -> None:
    src = inspect.getsource(hp.run_huge_prediction_representatives)
    assert "load_eval_aligned_few_shot" in src
