from math import comb

import pytest

from walkforward import CombinatorialPurgedSplit


def test_split_count_equals_n_choose_k():
    splitter = CombinatorialPurgedSplit(n_groups=6, test_groups=2)
    data = range(60)

    splits = list(splitter.split(data))

    assert len(splits) == comb(6, 2) == 15
    assert splitter.get_n_splits(data) == 15


@pytest.mark.parametrize(
    ("n_groups", "test_groups"),
    [(4, 1), (5, 2), (6, 2), (6, 3), (8, 4), (10, 3)],
)
def test_split_count_matches_combinations_across_shapes(n_groups, test_groups):
    splitter = CombinatorialPurgedSplit(n_groups=n_groups, test_groups=test_groups)
    data = range(n_groups * 9)

    assert splitter.get_n_splits(data) == comb(n_groups, test_groups)
    assert len(list(splitter.split(data))) == comb(n_groups, test_groups)


def test_n_paths_formula():
    splitter = CombinatorialPurgedSplit(n_groups=6, test_groups=2)

    assert splitter.get_n_paths() == comb(6, 2) * 2 // 6 == 5


def test_test_size_equals_chosen_groups():
    splitter = CombinatorialPurgedSplit(n_groups=5, test_groups=2)
    data = range(50)

    for _train_idx, test_idx in splitter.split(data):
        # Two evenly sized groups of 10 bars each.
        assert len(test_idx) == 20


def test_train_and_test_never_overlap():
    splitter = CombinatorialPurgedSplit(n_groups=6, test_groups=2, purge=3, embargo=2)
    data = range(60)

    for train_idx, test_idx in splitter.split(data):
        assert set(train_idx).isdisjoint(test_idx)


def test_purge_and_embargo_remove_neighbouring_train_bars():
    purge, embargo = 3, 2
    splitter = CombinatorialPurgedSplit(
        n_groups=6, test_groups=2, purge=purge, embargo=embargo
    )
    data = range(60)

    for train_idx, test_idx in splitter.split(data):
        train_set = set(train_idx)
        test_set = set(test_idx)
        # No train bar within `purge` bars before a test bar.
        for pos in test_set:
            for offset in range(1, purge + 1):
                assert (pos - offset) not in train_set or (pos - offset) in test_set
        # No train bar within `purge + embargo` bars after a test run end.
        sorted_test = sorted(test_set)
        run_ends = {
            value
            for value in sorted_test
            if (value + 1) not in test_set
        }
        for end in run_ends:
            for offset in range(1, purge + embargo + 1):
                assert (end + offset) not in train_set


def test_no_leakage_within_purge_window_strict():
    splitter = CombinatorialPurgedSplit(n_groups=8, test_groups=3, purge=4, embargo=1)
    data = range(80)

    for train_idx, test_idx in splitter.split(data):
        train_set = set(train_idx)
        # Every test bar is at least `purge + 1` bars from the nearest train bar
        # on the purged side.
        for pos in test_idx:
            for offset in range(1, 5):
                assert (pos - offset) not in train_set or (pos - offset) in set(test_idx)


def test_deterministic_ordering():
    splitter = CombinatorialPurgedSplit(n_groups=6, test_groups=2, purge=1)
    data = range(60)

    first = [(tuple(tr), tuple(te)) for tr, te in splitter.split(data)]
    second = [(tuple(tr), tuple(te)) for tr, te in splitter.split(data)]

    assert first == second
    # Lexicographic combination order: first split tests groups {0, 1}.
    assert first[0][1] == tuple(range(0, 20))


@pytest.mark.parametrize(
    ("n_groups", "test_groups"),
    [(4, 1), (5, 2), (6, 2), (6, 3), (8, 4), (10, 3)],
)
def test_paths_tile_the_series_exactly_once(n_groups, test_groups):
    splitter = CombinatorialPurgedSplit(
        n_groups=n_groups, test_groups=test_groups, purge=2, embargo=1
    )
    n_samples = n_groups * 11
    data = range(n_samples)

    paths = splitter.paths(data)

    assert len(paths) == splitter.get_n_paths()
    for path in paths:
        covered = sorted(idx for _train, test in path for idx in test)
        assert covered == list(range(n_samples))


def test_paths_blocks_have_disjoint_train_test():
    splitter = CombinatorialPurgedSplit(n_groups=6, test_groups=2, purge=2, embargo=1)
    data = range(66)

    for path in splitter.paths(data):
        for train_idx, test_idx in path:
            assert set(train_idx).isdisjoint(test_idx)


def test_uneven_sample_count_still_tiles():
    # 61 is not divisible by 6; early groups absorb the remainder.
    splitter = CombinatorialPurgedSplit(n_groups=6, test_groups=2)
    data = range(61)

    splits = list(splitter.split(data))
    assert len(splits) == comb(6, 2)
    for path in splitter.paths(data):
        covered = sorted(idx for _train, test in path for idx in test)
        assert covered == list(range(61))


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        CombinatorialPurgedSplit(n_groups=1, test_groups=1)

    with pytest.raises(ValueError):
        # test_groups must be strictly less than n_groups.
        CombinatorialPurgedSplit(n_groups=4, test_groups=4)

    with pytest.raises(ValueError):
        CombinatorialPurgedSplit(n_groups=5, test_groups=2, purge=-1)


def test_rejects_too_few_samples():
    splitter = CombinatorialPurgedSplit(n_groups=6, test_groups=2)
    with pytest.raises(ValueError):
        list(splitter.split(range(4)))
