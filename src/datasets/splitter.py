from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def stratified_take(df: pd.DataFrame, n: int, label_col: str, seed: int) -> pd.DataFrame:
    """
    Deterministically select `n` rows from `df` while approximately preserving
    the label distribution of `label_col`.

    Determinism rules:
    - class tie-breaks are resolved by sorting labels lexicographically
    - within each class, rows are shuffled using a seeded RNG and then taken

    Edge-case behavior:
    - if `n` >= len(df): returns a copy of df
    - if stratification is impossible (e.g., too few rows for some classes),
      it falls back to a deterministic seeded sample across the whole df.
    """

    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return df.iloc[0:0].copy()
    if n >= len(df):
        return df.copy()
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not in dataframe columns")

    # Work on stable index positions (avoid pandas index surprises).
    df = df.reset_index(drop=True)

    rng = np.random.default_rng(int(seed))

    # If there is only one class, sampling is straightforward.
    labels = df[label_col].astype(str)
    unique_labels = sorted(labels.unique().tolist())
    if len(unique_labels) <= 1:
        idx = rng.permutation(len(df))[:n]
        return df.iloc[np.sort(idx)].reset_index(drop=True)

    # Compute deterministic per-class allocations.
    counts = labels.value_counts().to_dict()
    total = len(df)
    proportions = {lab: counts[lab] / total for lab in unique_labels}

    # Initial allocation via floor of expected counts.
    alloc = {lab: int(np.floor(n * proportions[lab])) for lab in unique_labels}
    allocated = sum(alloc.values())

    # Distribute remainder by largest fractional part; tie-break by label name.
    if allocated < n:
        fracs = [(lab, (n * proportions[lab]) - alloc[lab]) for lab in unique_labels]
        fracs.sort(key=lambda x: (-x[1], x[0]))
        for lab, _ in fracs:
            if allocated >= n:
                break
            alloc[lab] += 1
            allocated += 1

    # If we have enough budget to cover each class at least once, try to do so.
    if n >= len(unique_labels):
        for lab in unique_labels:
            if alloc[lab] == 0:
                # Take one from this class, and remove one from the largest-allocated class.
                donor = max(unique_labels, key=lambda l: (alloc[l], l))
                if alloc[donor] <= 1:
                    # Can't donate without breaking totals; keep current allocation.
                    break
                alloc[lab] = 1
                alloc[donor] -= 1

    # Clamp allocations to available rows per class.
    for lab in unique_labels:
        alloc[lab] = min(alloc[lab], counts[lab])

    # If clamping reduced the total, redistribute remaining to classes with slack.
    allocated = sum(alloc.values())
    if allocated < n:
        slack = [(lab, counts[lab] - alloc[lab]) for lab in unique_labels]
        slack.sort(key=lambda x: (-x[1], x[0]))
        for lab, s in slack:
            if allocated >= n:
                break
            if s <= 0:
                continue
            take = min(s, n - allocated)
            alloc[lab] += take
            allocated += take

    if allocated != n:
        # Final fallback: deterministic global sample.
        idx = rng.permutation(len(df))[:n]
        return df.iloc[np.sort(idx)].reset_index(drop=True)

    chosen_idx: list[int] = []
    for lab in unique_labels:
        k = alloc[lab]
        if k <= 0:
            continue
        lab_idx = df.index[labels == lab].to_numpy()
        # Deterministic shuffle per label.
        lab_perm = rng.permutation(lab_idx)
        chosen_idx.extend(lab_perm[:k].tolist())

    # Deterministic final ordering: preserve original row order for readability/stability.
    chosen_idx = sorted(chosen_idx)
    return df.iloc[chosen_idx].reset_index(drop=True)


def label_distribution(df: pd.DataFrame, label_col: str) -> dict[str, int]:
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not in dataframe columns")
    s = df[label_col].astype(str).value_counts()
    return {k: int(v) for k, v in s.to_dict().items()}
