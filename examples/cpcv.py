"""Combinatorial purged CV over a synthetic bar series, plus an HTML report.

Runs every C(n_groups, test_groups) purged combination, prints the test blocks
per split, reconstructs the backtest paths, and writes a self-contained HTML
report next to this file.
"""

from math import comb

import pandas as pd

from walkforward import CombinatorialPurgedSplit, fold_stability, write_report


def make_bars() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=240, freq="D")
    return pd.DataFrame({"close": range(len(index))}, index=index)


if __name__ == "__main__":
    bars = make_bars()
    splitter = CombinatorialPurgedSplit(n_groups=6, test_groups=2, purge=5, embargo=2)

    splits = list(splitter.split(bars))
    print(f"groups=6 test_groups=2 -> {len(splits)} splits (C(6,2)={comb(6, 2)})")
    print(f"reconstructed backtest paths: {splitter.get_n_paths()}")

    for fold, (train_idx, test_idx) in enumerate(splits[:3], start=1):
        train = bars.iloc[train_idx]
        test = bars.iloc[test_idx]
        print(
            f"split {fold}: train {len(train)} bars, "
            f"test {test.index.min().date()}..{test.index.max().date()}"
        )

    # One per-path "score" stand-in: here just the size of each path's test set.
    path_sizes = [sum(len(test) for _train, test in path) for path in splitter.paths(bars)]
    distribution = {
        "n_paths": splitter.get_n_paths(),
        "n_splits": splitter.get_n_splits(bars),
        "path_test_bars": path_sizes[0],
    }

    metrics = fold_stability([0.62, 0.58, 0.55, 0.60, 0.57])

    # Build the HTML report from the splits as (train, test) DataFrame pairs.
    pairs = [(bars.iloc[tr], bars.iloc[te]) for tr, te in splits]
    write_report(
        "examples/cpcv_report.html",
        pairs,
        title="CPCV demo report",
        metrics=metrics,
        path_distribution=distribution,
    )
    print("wrote examples/cpcv_report.html")
