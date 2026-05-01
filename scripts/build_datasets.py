from __future__ import annotations

from pprint import pprint

from src.datasets.registry import build_all_splits


def main() -> None:
    results = build_all_splits(seed=42, test_tiers=(20, 200, 5000), train_seed_tiers=(10, 100, 5000))
    pprint(results)


if __name__ == "__main__":
    main()

