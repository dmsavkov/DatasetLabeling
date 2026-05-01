import pandas as pd

from src.datasets.splitter import stratified_take


def test_stratified_take_is_deterministic() -> None:
    df = pd.DataFrame(
        {
            "text": [f"t{i}" for i in range(100)],
            "label": ["a"] * 60 + ["b"] * 30 + ["c"] * 10,
        }
    )
    out1 = stratified_take(df, 20, "label", seed=42)
    out2 = stratified_take(df, 20, "label", seed=42)
    assert out1.equals(out2)


def test_stratified_take_respects_n_and_has_all_columns() -> None:
    df = pd.DataFrame({"text": ["x"] * 50, "label": ["a"] * 25 + ["b"] * 25})
    out = stratified_take(df, 7, "label", seed=1)
    assert len(out) == 7
    assert set(out.columns) == set(df.columns)


def test_stratified_take_handles_single_class() -> None:
    df = pd.DataFrame({"text": [f"x{i}" for i in range(10)], "label": ["a"] * 10})
    out = stratified_take(df, 5, "label", seed=123)
    assert len(out) == 5
    assert set(out["label"].unique().tolist()) == {"a"}

