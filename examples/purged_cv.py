"""Run purged walk-forward CV over a synthetic bar series."""

import pandas as pd

from walkforward import PurgedWalkForwardSplit


def make_bars() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=220, freq="D")
    return pd.DataFrame({"close": range(len(index))}, index=index)


if __name__ == "__main__":
    bars = make_bars()
    splitter = PurgedWalkForwardSplit(train_size=90, test_size=20, purge=5, embargo=2)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(bars), start=1):
        train = bars.iloc[train_idx]
        test = bars.iloc[test_idx]
        print(
            f"fold {fold}: train {train.index.min().date()}..{train.index.max().date()} "
            f"-> test {test.index.min().date()}..{test.index.max().date()}"
        )
