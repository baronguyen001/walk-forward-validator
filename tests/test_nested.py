import pytest

from walkforward import NestedSplit, NestedWalkForwardSplit


def test_golden_sample():
    splitter = NestedWalkForwardSplit(train_size=10, test_size=5, step_size=5, inner_splits=2)

    splits = list(splitter.split(range(20)))

    assert len(splits) == 2
    assert splits == [
        NestedSplit(
            train_idx=list(range(0, 10)),
            test_idx=list(range(10, 15)),
            inner=[
                ([0, 1, 2], [3, 4, 5]),
                ([0, 1, 2, 3, 4, 5], [6, 7, 8]),
            ],
        ),
        NestedSplit(
            train_idx=list(range(5, 15)),
            test_idx=list(range(15, 20)),
            inner=[
                ([5, 6, 7], [8, 9, 10]),
                ([5, 6, 7, 8, 9, 10], [11, 12, 13]),
            ],
        ),
    ]


def test_inner_indices_are_in_outer_train_and_before_outer_test():
    splitter = NestedWalkForwardSplit(train_size=10, test_size=5, step_size=5, inner_splits=2)

    for split in splitter.split(range(20)):
        train_idx = set(split.train_idx)
        for inner_train, inner_val in split.inner:
            assert set(inner_train).issubset(train_idx)
            assert set(inner_val).issubset(train_idx)
            assert max(inner_val) < min(split.test_idx)


def test_purge_delays_and_shortens_inner_validation_windows():
    splitter = NestedWalkForwardSplit(
        train_size=10,
        test_size=5,
        step_size=5,
        inner_splits=2,
        purge=2,
    )

    splits = list(splitter.split(range(20)))

    assert splits[0].inner == [
        ([0, 1, 2], [5]),
        ([0, 1, 2, 3, 4, 5], [8]),
    ]
    assert splits[1].inner == [
        ([5, 6, 7], [10]),
        ([5, 6, 7, 8, 9, 10], [13]),
    ]


def test_too_many_inner_splits_yield_no_inner_folds():
    splitter = NestedWalkForwardSplit(train_size=4, test_size=2, step_size=2, inner_splits=9)

    splits = list(splitter.split(range(8)))

    assert len(splits) == 2
    assert splits[0].inner == []
    assert splits[1].inner == []


def test_invalid_inner_splits():
    with pytest.raises(ValueError, match="inner_splits must be >= 1"):
        NestedWalkForwardSplit(train_size=4, test_size=2, inner_splits=0)

    with pytest.raises(TypeError, match="inner_splits must be an integer"):
        NestedWalkForwardSplit(train_size=4, test_size=2, inner_splits=True)


def test_get_n_splits_matches_split():
    splitter = NestedWalkForwardSplit(train_size=6, test_size=3, step_size=3, inner_splits=2)

    assert splitter.get_n_splits(range(15)) == len(list(splitter.split(range(15))))


def test_expanding_outer_windows_keep_inner_folds_expanding():
    splitter = NestedWalkForwardSplit(
        train_size=4,
        test_size=2,
        step_size=2,
        inner_splits=1,
        expanding=True,
    )

    splits = list(splitter.split(range(10)))

    assert splits[0].train_idx == [0, 1, 2, 3]
    assert splits[0].inner == [([0, 1], [2, 3])]
    assert splits[1].train_idx == [0, 1, 2, 3, 4, 5]
    assert splits[1].inner == [([0, 1, 2], [3, 4, 5])]


def test_yielded_lists_are_independent_copies():
    splitter = NestedWalkForwardSplit(train_size=10, test_size=5, step_size=5, inner_splits=2)

    first, second = list(splitter.split(range(20)))
    first.train_idx.append(999)

    assert 999 not in second.train_idx
    assert second.train_idx == list(range(5, 15))
