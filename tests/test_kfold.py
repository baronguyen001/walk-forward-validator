from __future__ import annotations

import pytest

from walkforward import PurgedKFold


def test_test_folds_partition_all_samples() -> None:
    n = 53
    kf = PurgedKFold(n_splits=5)
    test_sets = [test for _, test in kf.split(range(n))]
    assert len(test_sets) == 5
    flat = [i for fold in test_sets for i in fold]
    assert sorted(flat) == list(range(n))  # exact partition, no overlap
    assert kf.get_n_splits() == 5


def test_train_and_test_are_disjoint() -> None:
    kf = PurgedKFold(n_splits=4)
    for train, test in kf.split(range(40)):
        assert set(train).isdisjoint(test)


def test_purge_and_embargo_remove_neighbours() -> None:
    n, purge, embargo = 40, 2, 3
    kf = PurgedKFold(n_splits=4, purge=purge, embargo=embargo)
    for train, test in kf.split(range(n)):
        test_start, test_end = test[0], test[-1] + 1  # test_end exclusive
        # the exclusion band is [test_start - purge, test_end + purge + embargo)
        for i in train:
            assert i < test_start - purge or i >= test_end + purge + embargo


def test_uneven_fold_sizes() -> None:
    # 11 / 3 -> folds of size 4, 4, 3
    kf = PurgedKFold(n_splits=3)
    sizes = [len(test) for _, test in kf.split(range(11))]
    assert sizes == [4, 4, 3]


def test_invalid_params() -> None:
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1)
    with pytest.raises(TypeError):
        PurgedKFold(n_splits=3, purge=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        list(PurgedKFold(n_splits=5).split(range(3)))  # n < n_splits
