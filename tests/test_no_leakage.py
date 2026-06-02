import pandas as pd
import pytest

from walkforward import walk_forward_split
from walkforward.splitter import _assert_no_leakage


def test_every_yielded_fold_has_no_train_val_leakage():
    index = pd.date_range("2024-01-01", periods=500, freq="D")
    data = pd.DataFrame({"value": range(500)}, index=index)

    for fold in walk_forward_split(data, train_days=120, val_days=30, step_days=15):
        assert fold.train.index.max() < fold.val.index.min()


def test_manual_overlapping_split_trips_leakage_assertion():
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    data = pd.DataFrame({"value": range(10)}, index=index)
    train = data.iloc[:6]
    val = data.iloc[5:]

    with pytest.raises(AssertionError, match="leakage"):
        _assert_no_leakage(train, val)
