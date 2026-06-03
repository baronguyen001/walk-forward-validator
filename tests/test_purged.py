import pytest

from walkforward import PurgedWalkForwardSplit


def test_purge_removes_expected_gap_between_train_and_test():
    train_idx, test_idx = next(
        PurgedWalkForwardSplit(train_size=5, test_size=3, purge=2).split(range(20))
    )

    assert train_idx == [0, 1, 2, 3, 4]
    assert test_idx == [7, 8, 9]
    assert set(train_idx).isdisjoint(test_idx)
    assert 5 not in train_idx + test_idx
    assert 6 not in train_idx + test_idx
    assert test_idx[0] - train_idx[-1] - 1 == 2


def test_embargo_moves_next_test_window_past_after_test_bars():
    splits = list(
        PurgedWalkForwardSplit(
            train_size=5,
            test_size=3,
            purge=2,
            embargo=1,
        ).split(range(20))
    )

    first_test = splits[0][1]
    second_train, second_test = splits[1]
    embargo_idx = first_test[-1] + 1

    assert second_test[0] == embargo_idx + 1
    assert embargo_idx not in second_train
    assert embargo_idx not in second_test


def test_every_purged_fold_has_no_train_test_leakage():
    splitter = PurgedWalkForwardSplit(train_size=20, test_size=5, step_size=2, purge=4)

    for train_idx, test_idx in splitter.split(range(80)):
        assert max(train_idx) + 4 < min(test_idx)
        assert set(train_idx).isdisjoint(test_idx)


def test_expanding_purged_split_keeps_train_start_pinned():
    splits = list(
        PurgedWalkForwardSplit(
            train_size=4,
            test_size=2,
            purge=1,
            expanding=True,
        ).split(range(15))
    )

    assert all(train_idx[0] == 0 for train_idx, _ in splits)
    assert [len(train_idx) for train_idx, _ in splits] == [4, 6, 8, 10, 12]


def test_purged_split_validates_window_sizes():
    with pytest.raises(ValueError):
        PurgedWalkForwardSplit(train_size=0, test_size=3)

    with pytest.raises(ValueError):
        PurgedWalkForwardSplit(train_size=3, test_size=3, purge=-1)
