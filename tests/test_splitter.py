import pandas as pd
import pytest

from walkforward import WalkForward, classify_robustness, walk_forward_split


def daily_frame(days: int = 730) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=days, freq="D")
    return pd.DataFrame({"value": range(days)}, index=index)


def test_rolling_fold_count_matches_expected_envelope():
    train_days = 365
    val_days = 90
    step_days = 30
    folds = list(
        walk_forward_split(
            daily_frame(),
            train_days=train_days,
            val_days=val_days,
            step_days=step_days,
        )
    )
    span_days = (daily_frame().index.max() - daily_frame().index.min()).days
    full_window_count = ((span_days - train_days - val_days) // step_days) + 1
    partial_window_count = ((span_days - train_days) // step_days) + 1
    assert len(folds) >= 1
    assert full_window_count <= len(folds) <= partial_window_count


def test_expanding_folds_have_non_decreasing_train_length():
    folds = list(
        walk_forward_split(daily_frame(), train_days=180, val_days=60, step_days=45, expanding=True)
    )
    lengths = [len(fold.train) for fold in folds]
    assert lengths == sorted(lengths)
    assert len(set(lengths)) > 1


def test_walk_forward_class_yields_train_val_pairs():
    pairs = list(WalkForward(180, 60, 45).split(daily_frame()))
    assert pairs
    train, val = pairs[0]
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)


@pytest.mark.parametrize(
    ("train_rank", "val_rank", "n", "label"),
    [
        (1, 1, 10, "STRONG ROBUST"),
        (2, 3, 10, "robust"),
        (2, 8, 10, "OVERFIT (good train, bad val)"),
        (8, 2, 10, "underperformer"),
        (8, 9, 10, "weak both"),
    ],
)
def test_classify_robustness_labels_rank_quadrants(train_rank, val_rank, n, label):
    assert classify_robustness(train_rank, val_rank, n) == label


def test_requires_datetime_index():
    with pytest.raises(TypeError):
        list(walk_forward_split(pd.DataFrame({"x": [1, 2]}), train_days=1, val_days=1, step_days=1))
