"""Print rolling and expanding walk-forward folds for a synthetic daily series."""

import pandas as pd

from walkforward import walk_forward_split


def make_series() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=420, freq="D")
    return pd.DataFrame({"signal": range(len(index))}, index=index)


def show(expanding: bool) -> None:
    mode = "expanding" if expanding else "rolling"
    print(f"\n{mode}")
    for fold in walk_forward_split(
        make_series(),
        train_days=120,
        val_days=30,
        step_days=30,
        expanding=expanding,
    ):
        print(
            f"train {fold.train_start.date()}..{fold.train_end.date()} "
            f"-> val {fold.val_start.date()}..{fold.val_end.date()}"
        )


if __name__ == "__main__":
    show(expanding=False)
    show(expanding=True)
