import pandas as pd

from src.datasets.splitter import stratified_take


def test_stratified_take_is_actually_stratified_within_tolerance() -> None:
    # 3-class distribution: 70/20/10
    df = pd.DataFrame(
        {
            "text": [f"t{i}" for i in range(1000)],
            "label": (["a"] * 700) + (["b"] * 200) + (["c"] * 100),
        }
    )
    out = stratified_take(df, 100, "label", seed=42)
    counts = out["label"].value_counts().to_dict()

    # Expected: 70/20/10 with small integer rounding wiggle.
    assert abs(int(counts.get("a", 0)) - 70) <= 2
    assert abs(int(counts.get("b", 0)) - 20) <= 2
    assert abs(int(counts.get("c", 0)) - 10) <= 2

