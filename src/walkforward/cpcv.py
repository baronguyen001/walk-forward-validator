"""Combinatorial purged cross-validation (CPCV) index splits.

This implements the combinatorial purged cross-validation scheme described by
Marcos López de Prado (*Advances in Financial Machine Learning*, ch. 12). The
series is cut into ``n_groups`` contiguous blocks; every combination of
``test_groups`` blocks is used as a test set, leaving the remaining blocks for
training. That produces ``C(n_groups, test_groups)`` splits instead of a single
walk-forward path.

Why bother? A plain walk-forward backtest gives **one** out-of-sample equity
path, so its Sharpe/return estimate has high variance and is easy to overfit.
CPCV reconstructs ``C(n_groups, test_groups) * test_groups / n_groups`` distinct
out-of-sample *paths* from the same data, so you can look at the distribution of
a metric across paths rather than trusting a single number. The variance of the
mean across those paths is far lower than the variance of any one path, which is
the whole point of the method.

Leakage is killed exactly as in :mod:`walkforward.purged`: ``purge`` bars are
dropped from the training set on both sides of every contiguous test block, and
``embargo`` bars immediately *after* each test block are also dropped. Because
the test blocks need not be adjacent, purge/embargo are applied around each
contiguous run of test indices independently.
"""

from __future__ import annotations

from collections.abc import Iterator, Sized
from dataclasses import dataclass
from itertools import combinations
from math import comb


def _validate_int(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _group_bounds(n_samples: int, n_groups: int) -> list[tuple[int, int]]:
    """Return ``[start, stop)`` integer-position bounds for each contiguous group.

    Groups are as even as possible; when ``n_samples`` is not divisible by
    ``n_groups`` the earliest groups absorb the remainder (one extra bar each).
    """

    base, remainder = divmod(n_samples, n_groups)
    bounds: list[tuple[int, int]] = []
    start = 0
    for group in range(n_groups):
        size = base + (1 if group < remainder else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def _contiguous_runs(positions: list[int]) -> list[tuple[int, int]]:
    """Collapse a sorted index list into ``[start, stop)`` contiguous runs."""

    runs: list[tuple[int, int]] = []
    run_start = positions[0]
    prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        runs.append((run_start, prev + 1))
        run_start = pos
        prev = pos
    runs.append((run_start, prev + 1))
    return runs


@dataclass(frozen=True)
class CombinatorialPurgedSplit:
    """Yield purged train/test index pairs for every group combination.

    The data is split into ``n_groups`` contiguous blocks. Each combination of
    ``test_groups`` blocks becomes a test set; the remaining bars (minus the
    purge/embargo gaps around every contiguous test run) become the training
    set. Combinations are enumerated in lexicographic group order, so iteration
    is deterministic.

    Attributes:
        n_groups: Number of contiguous blocks the series is cut into (``N``).
        test_groups: Blocks held out as test per combination (``k``, ``1 <= k < N``).
        purge: Bars removed from training on each side of every test run.
        embargo: Bars removed from training immediately after every test run.
    """

    n_groups: int
    test_groups: int = 1
    purge: int = 0
    embargo: int = 0

    def __post_init__(self) -> None:
        _validate_int("n_groups", self.n_groups, minimum=2)
        _validate_int("test_groups", self.test_groups, minimum=1)
        _validate_int("purge", self.purge, minimum=0)
        _validate_int("embargo", self.embargo, minimum=0)
        if self.test_groups >= self.n_groups:
            raise ValueError("test_groups must be < n_groups (need at least one train group)")

    def get_n_splits(self, data: Sized | None = None) -> int:
        """Return the number of combinations, ``C(n_groups, test_groups)``.

        Independent of ``data``; the argument is accepted for API symmetry with
        :class:`~walkforward.purged.PurgedWalkForwardSplit`.
        """

        return comb(self.n_groups, self.test_groups)

    def get_n_paths(self) -> int:
        """Return the number of reconstructed backtest paths.

        Each path threads one test prediction through every group. With ``k``
        test groups per combination there are ``C(N, k) * k / N`` such paths.
        """

        return self.get_n_splits() * self.test_groups // self.n_groups

    def split(self, data: Sized) -> Iterator[tuple[list[int], list[int]]]:
        """Yield ``(train_idx, test_idx)`` integer-position splits.

        One split per combination of ``test_groups`` blocks. Purge and embargo
        bars are excluded from ``train_idx`` around each contiguous test run, so
        ``set(train_idx)`` and ``set(test_idx)`` never touch within the gap.
        """

        n_samples = len(data)
        if n_samples < self.n_groups:
            raise ValueError(
                f"need at least n_groups={self.n_groups} samples, got {n_samples}"
            )

        bounds = _group_bounds(n_samples, self.n_groups)

        for test_combo in combinations(range(self.n_groups), self.test_groups):
            test_positions = sorted(
                pos
                for group in test_combo
                for pos in range(bounds[group][0], bounds[group][1])
            )
            test_set = set(test_positions)

            blocked = set(test_set)
            for run_start, run_stop in _contiguous_runs(test_positions):
                # Purge symmetrically; embargo only forward (after the test run).
                for pos in range(run_start - self.purge, run_start):
                    blocked.add(pos)
                for pos in range(run_stop, run_stop + self.purge + self.embargo):
                    blocked.add(pos)

            train_idx = [pos for pos in range(n_samples) if pos not in blocked]
            test_idx = test_positions

            # Defensive leakage guard mirroring PurgedWalkForwardSplit.
            if self.purge and train_idx:
                train_lookup = set(train_idx)
                for run_start, run_stop in _contiguous_runs(test_positions):
                    for pos in range(run_start - self.purge, run_start):
                        if pos in train_lookup:
                            raise AssertionError("CPCV purge leakage detected")
                    for pos in range(run_stop, run_stop + self.purge):
                        if pos in train_lookup:
                            raise AssertionError("CPCV purge leakage detected")

            yield train_idx, test_idx

    def paths(self, data: Sized) -> list[list[tuple[list[int], list[int]]]]:
        """Group splits into the reconstructed backtest paths.

        Returns ``get_n_paths()`` lists. Each inner list is an ordered sequence
        of ``(train_idx, test_idx)`` splits whose test blocks tile the whole
        series exactly once (one block per group, in group order), i.e. one
        continuous out-of-sample equity path. Useful for collecting a metric per
        path and studying its distribution.

        Each group is tested by ``C(n_groups - 1, test_groups - 1)`` combinations,
        which equals ``get_n_paths()``. Path ``p`` takes, for every group ``g``,
        the ``p``-th combination (in lexicographic order) that tests ``g``. Since
        each group is tested exactly ``get_n_paths()`` times, every path receives
        one test-block per group and therefore tiles the whole series exactly
        once. This is de Prado's path-reconstruction matrix; the assignment is
        deterministic.
        """

        splits = list(self.split(data))
        bounds = _group_bounds(len(data), self.n_groups)

        # split_index -> sorted group ids it tests, in split (lex) order.
        # Derive groups from the group bounds (not contiguous runs) so that
        # adjacent test groups are not merged into a single run.
        combo_groups: list[list[int]] = []
        for _train_idx, test_idx in splits:
            test_set = set(test_idx)
            groups = [
                group
                for group, (start, stop) in enumerate(bounds)
                if start in test_set and stop - 1 in test_set
            ]
            combo_groups.append(groups)

        n_paths = self.get_n_paths()
        # matrix[group][path] = split index supplying that group's test block.
        matrix: list[list[int | None]] = [[None] * n_paths for _ in range(self.n_groups)]
        next_path_for_group = [0] * self.n_groups
        for split_index, groups in enumerate(combo_groups):
            for group in groups:
                column = next_path_for_group[group]
                matrix[group][column] = split_index
                next_path_for_group[group] += 1

        paths: list[list[tuple[list[int], list[int]]]] = []
        for path_index in range(n_paths):
            ordered_splits: list[tuple[list[int], list[int]]] = []
            for group in range(self.n_groups):
                split_index = matrix[group][path_index]
                if split_index is None:
                    continue
                train_idx, _full_test = splits[split_index]
                start, stop = bounds[group]
                ordered_splits.append((train_idx, list(range(start, stop))))
            paths.append(ordered_splits)
        return paths
