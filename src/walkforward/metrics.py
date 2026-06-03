"""Fold-level stability summaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import inf
from statistics import fmean, pstdev
from typing import Any


def _coerce_scores(values: Iterable[float], *, name: str) -> list[float]:
    scores = [float(value) for value in values]
    if not scores:
        raise ValueError(f"{name} must contain at least one score")
    return scores


def _extract_score_columns(
    scores: Iterable[Any] | Mapping[str, Iterable[float]],
    train_scores: Iterable[float] | None,
) -> tuple[list[float] | None, list[float]]:
    if train_scores is not None:
        return _coerce_scores(train_scores, name="train_scores"), _coerce_scores(
            scores, name="scores"
        )

    if isinstance(scores, Mapping):
        test_values = scores.get("test", scores.get("val", scores.get("validation")))
        if test_values is None:
            test_values = scores.get("test_scores")
        if test_values is None:
            raise ValueError("scores mapping must include test, val, validation, or test_scores")

        train_values = scores.get("train", scores.get("train_scores"))
        train = None if train_values is None else _coerce_scores(train_values, name="train_scores")
        return train, _coerce_scores(test_values, name="test_scores")

    rows = list(scores)
    if not rows:
        raise ValueError("scores must contain at least one score")

    first = rows[0]
    if isinstance(first, Mapping):
        train = _coerce_scores((row["train"] for row in rows), name="train_scores")
        test = _coerce_scores(
            (row.get("test", row.get("val", row.get("validation"))) for row in rows),
            name="test_scores",
        )
        return train, test

    if isinstance(first, Sequence) and not isinstance(first, str) and len(first) == 2:
        train = _coerce_scores((row[0] for row in rows), name="train_scores")
        test = _coerce_scores((row[1] for row in rows), name="test_scores")
        return train, test

    return None, _coerce_scores(rows, name="scores")


def _degradation_pct(
    train_mean: float | None,
    test_mean: float,
    *,
    higher_is_better: bool,
) -> float | None:
    if train_mean is None:
        return None
    if train_mean == 0:
        return 0.0 if test_mean == 0 else inf

    if higher_is_better:
        return ((train_mean - test_mean) / abs(train_mean)) * 100
    return ((test_mean - train_mean) / abs(train_mean)) * 100


def fold_stability(
    scores: Iterable[Any] | Mapping[str, Iterable[float]],
    train_scores: Iterable[float] | None = None,
    *,
    higher_is_better: bool = True,
    overfit_threshold_pct: float = 10.0,
) -> dict[str, float | int | bool | None]:
    """Summarize fold scores and flag train/test degradation.

    `scores` may be test-only scores, `(train, test)` pairs, fold dictionaries, or a
    mapping with `train` and `test` score lists. Standard deviations are population
    standard deviations so single-fold runs report `0.0`.
    """

    train, test = _extract_score_columns(scores, train_scores)
    if train is not None and len(train) != len(test):
        raise ValueError("train and test scores must have the same length")

    test_mean = fmean(test)
    train_mean = None if train is None else fmean(train)
    degradation = _degradation_pct(train_mean, test_mean, higher_is_better=higher_is_better)

    return {
        "n": len(test),
        "mean": test_mean,
        "std": pstdev(test),
        "train_mean": train_mean,
        "train_std": None if train is None else pstdev(train),
        "test_mean": test_mean,
        "test_std": pstdev(test),
        "degradation_pct": degradation,
        "overfit": degradation is not None and degradation > overfit_threshold_pct,
    }
