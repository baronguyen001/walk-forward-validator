"""Purged walk-forward index splits."""

from __future__ import annotations

from collections.abc import Iterator, Sized
from dataclasses import dataclass


def _validate_int(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


@dataclass(frozen=True)
class PurgedWalkForwardSplit:
    """Yield walk-forward train/test index pairs with purge and embargo gaps.

    `purge` removes bars between the training window and the test window. `embargo`
    skips bars after each test window before the next test window is allowed to start.
    """

    train_size: int
    test_size: int
    step_size: int | None = None
    purge: int = 0
    embargo: int = 0
    expanding: bool = False

    def __post_init__(self) -> None:
        _validate_int("train_size", self.train_size, minimum=1)
        _validate_int("test_size", self.test_size, minimum=1)
        _validate_int("purge", self.purge, minimum=0)
        _validate_int("embargo", self.embargo, minimum=0)
        if self.step_size is None:
            object.__setattr__(self, "step_size", self.test_size)
        else:
            _validate_int("step_size", self.step_size, minimum=1)

    def split(self, data: Sized) -> Iterator[tuple[list[int], list[int]]]:
        """Yield `(train_idx, test_idx)` integer-position splits."""

        n_samples = len(data)
        anchor = 0

        while anchor < n_samples:
            train_start = 0 if self.expanding else anchor
            train_end = anchor + self.train_size
            test_start = train_end + self.purge
            test_end = test_start + self.test_size

            if test_start >= n_samples:
                break

            train_idx = list(range(train_start, min(train_end, n_samples)))
            test_idx = list(range(test_start, min(test_end, n_samples)))

            if train_idx and test_idx:
                gap = test_idx[0] - train_idx[-1] - 1
                if gap < self.purge:
                    raise AssertionError("purged walk-forward leakage detected")
                yield train_idx, test_idx

            next_anchor = anchor + self.step_size
            first_anchor_after_embargo = test_end + self.embargo - self.train_size - self.purge
            anchor = max(next_anchor, first_anchor_after_embargo)

    def get_n_splits(self, data: Sized) -> int:
        """Return the number of splits produced for `data`."""

        return sum(1 for _ in self.split(data))
