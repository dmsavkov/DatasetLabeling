# pyright: basic
from __future__ import annotations

import pandas as pd
import pytest

from src.eval.metrics import compute_confusion_stats, compute_performance_excluding_confused


def test_metrics_exclude_confused_rows() -> None:
    df = pd.DataFrame(
        {
            "true_label": ["a", "a", "b", "b"],
            "pred_label": ["a", None, "b", "a"],
            "is_confusing": [False, True, False, False],
        }
    )
    bundle = compute_performance_excluding_confused(df)
    assert bundle.confusion_stats is not None
    assert bundle.confusion_stats.n_total == 4
    assert bundle.confusion_stats.n_confusing == 1
    assert bundle.confusion_stats.n_scored == 3
    # scored rows: correct a/a, correct b/b, wrong b/a -> acc 2/3
    assert bundle.metrics.accuracy == pytest.approx(2 / 3)


def test_confusion_stats_zero_multi() -> None:
    df = pd.DataFrame(
        {
            "is_confusing": [True, False, True],
            "n_pred_labels": [0, 1, 2],
        }
    )
    stats = compute_confusion_stats(df)
    assert stats is not None
    assert stats.n_zero_labels == 1
    assert stats.n_multi_labels == 1
