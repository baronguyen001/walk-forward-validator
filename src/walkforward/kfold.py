"""Purged, embargoed K-fold cross-validation.

Unlike the walk-forward and combinatorial splitters, ``PurgedKFold`` is a
scikit-learn-style cross-validator: it cuts the sample into ``n_splits``
contiguous test folds and, for each, trains on everything else *minus* a purge
band around the fold and an embargo band after it. This removes the
overlapping-label leakage that plagues naive K-fold on time series, while
keeping the familiar ``split`` / ``get_n_splits`` API.
"""

from __future__ import annotations

from collections.abc import Iterator, Sized
from dataclasses import dataclass


def _validate_int(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _fold_bounds(n_samples: int, n_splits: int) -> list[tuple[int, int]]:
    """Contiguous, near-equal fold boundaries (mirrors numpy.array_split sizing)."""
    base, extra = divmod(n_samples, n_splits)
    bounds: list[tuple[int, int]] = []
    start = 0
    for fold in range(n_splits):
        size = base + (1 if fold < extra else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


@dataclass(frozen=True)
class PurgedKFold:
    """K-fold splits with a purge band around, and embargo band after, each test fold.

    ``purge`` drops training samples within that many positions on either side of
    the test fold (their labels can overlap the test window). ``embargo`` drops a
    further band of training samples immediately after the test fold.
    """

    n_splits: int = 5
    purge: int = 0
    embargo: int = 0

    def __post_init__(self) -> None:
        _validate_int("n_splits", self.n_splits, minimum=2)
        _validate_int("purge", self.purge, minimum=0)
        _validate_int("embargo", self.embargo, minimum=0)

    def split(self, data: Sized) -> Iterator[tuple[list[int], list[int]]]:
        """Yield ``(train_idx, test_idx)`` integer-position splits."""
        n_samples = len(data)
        if n_samples < self.n_splits:
            raise ValueError("n_samples must be >= n_splits")

        for test_start, test_end in _fold_bounds(n_samples, self.n_splits):
            if test_start == test_end:
                continue
            test_idx = list(range(test_start, test_end))
            block_start = max(0, test_start - self.purge)
            block_end = min(n_samples, test_end + self.purge + self.embargo)
            train_idx = [
                i for i in range(n_samples) if i < block_start or i >= block_end
            ]
            if train_idx:
                yield train_idx, test_idx

    def get_n_splits(self, data: Sized | None = None) -> int:
        """Return the number of folds (``n_splits``)."""
        return self.n_splits
