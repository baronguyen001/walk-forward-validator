"""Fold visualization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any


def _split_train_test(split: Any) -> tuple[Any, Any]:
    train = getattr(split, "train", None)
    test = getattr(split, "val", None)
    if train is not None and test is not None:
        return train, test

    if isinstance(split, Sequence) and not isinstance(split, str) and len(split) == 2:
        return split[0], split[1]

    raise TypeError("each split must be a Fold or a (train, test) pair")


def _values(obj: Any) -> list[Any]:
    if hasattr(obj, "index") and not isinstance(obj, range):
        return list(obj.index)
    return list(obj)


def _is_datetime_like(value: Any) -> bool:
    return isinstance(value, date | datetime) or hasattr(value, "to_pydatetime")


def _as_datetime(value: Any) -> date | datetime:
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return value


def _step_width(values: list[float]) -> float:
    diffs = [
        right - left
        for left, right in zip(values, values[1:], strict=False)
        if right - left > 0
    ]
    return min(diffs) if diffs else 1.0


def _span(values: list[Any]) -> tuple[float, float, bool] | None:
    if not values:
        return None

    if _is_datetime_like(values[0]):
        from matplotlib.dates import date2num

        axis_values = sorted(date2num([_as_datetime(value) for value in values]))
        is_datetime = True
    else:
        axis_values = sorted(float(value) for value in values)
        is_datetime = False

    start = axis_values[0]
    width = axis_values[-1] - start + _step_width(axis_values)
    return start, max(width, 1.0), is_datetime


def fold_plot(splits: Iterable[Any]):
    """Return a matplotlib Figure showing train and test bands for each fold."""

    import matplotlib.pyplot as plt

    parsed = [_split_train_test(split) for split in splits]
    if not parsed:
        raise ValueError("splits must contain at least one fold")

    fig_height = max(2.4, 1.2 + (0.42 * len(parsed)))
    fig, ax = plt.subplots(figsize=(8.5, fig_height), constrained_layout=True)
    uses_datetime = False

    for row, (train, test) in enumerate(parsed):
        band_y = row - 0.32
        train_span = _span(_values(train))
        test_span = _span(_values(test))

        if train_span is not None:
            start, width, is_datetime = train_span
            uses_datetime = uses_datetime or is_datetime
            ax.broken_barh(
                [(start, width)],
                (band_y, 0.28),
                facecolors="#2a9d8f",
                edgecolors="none",
                label="Train" if row == 0 else None,
            )

        if test_span is not None:
            start, width, is_datetime = test_span
            uses_datetime = uses_datetime or is_datetime
            ax.broken_barh(
                [(start, width)],
                (band_y + 0.32, 0.28),
                facecolors="#e9c46a",
                edgecolors="none",
                label="Test" if row == 0 else None,
            )

    ax.set_title("Walk-forward folds")
    ax.set_xlabel("Index")
    ax.set_ylabel("Fold")
    ax.set_yticks(range(len(parsed)))
    ax.set_yticklabels([str(index + 1) for index in range(len(parsed))])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#d9dee3", linewidth=0.8)
    ax.legend(loc="upper right", frameon=False)

    if uses_datetime:
        ax.set_xlabel("Date")
        fig.autofmt_xdate()

    return fig
