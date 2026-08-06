"""Nested walk-forward splits for leak-free hyper-parameter selection."""

from __future__ import annotations

from collections.abc import Iterator, Sized
from dataclasses import dataclass

from .purged import PurgedWalkForwardSplit


@dataclass(frozen=True)
class NestedSplit:
    """An outer split plus the inner folds used to choose hyper-parameters.

    Tune on `inner` only, then score `test_idx` exactly once with the selected
    parameters. Every index in `inner` is a member of `train_idx`, so the outer
    test block never takes part in selection.
    """

    train_idx: list[int]
    test_idx: list[int]
    inner: list[tuple[list[int], list[int]]]


@dataclass(frozen=True)
class NestedWalkForwardSplit:
    """Configure nested walk-forward validation with expanding inner tuning folds.

    Inner folds always expand from the beginning of each outer training window,
    regardless of `expanding`, which only controls the outer windows. The `purge`
    gap is applied between each inner training window and its validation window,
    mirroring the outer purge. Any tail of the outer training block beyond the
    blocks consumed by the inner folds is intentionally left unused.
    """

    train_size: int
    test_size: int
    inner_splits: int = 3
    step_size: int | None = None
    purge: int = 0
    embargo: int = 0
    expanding: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.inner_splits, bool) or not isinstance(self.inner_splits, int):
            raise TypeError("inner_splits must be an integer")
        if self.inner_splits < 1:
            raise ValueError("inner_splits must be >= 1")

    def split(self, data: Sized) -> Iterator[NestedSplit]:
        """Yield nested splits with original-series positions for every inner fold."""

        outer = PurgedWalkForwardSplit(
            train_size=self.train_size,
            test_size=self.test_size,
            step_size=self.step_size,
            purge=self.purge,
            embargo=self.embargo,
            expanding=self.expanding,
        )
        for train_idx, test_idx in outer.split(data):
            m = len(train_idx)
            k = self.inner_splits
            block = m // (k + 1)
            inner: list[tuple[list[int], list[int]]] = []
            if block >= 1:
                for i in range(k):
                    inner_train = train_idx[: block * (i + 1)]
                    val_start = block * (i + 1) + self.purge
                    val_stop = block * (i + 2)
                    inner_val = train_idx[val_start:val_stop]
                    if inner_train and inner_val:
                        inner.append((inner_train, inner_val))
            yield NestedSplit(train_idx=train_idx[:], test_idx=test_idx[:], inner=inner)

    def get_n_splits(self, data: Sized) -> int:
        """Return the number of outer splits produced for `data`."""

        return sum(1 for _ in self.split(data))
