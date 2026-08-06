"""Train/test distribution drift diagnostics for walk-forward folds.

A fold can degrade for two very different reasons: the strategy was overfit, or
the test window simply came from a different world. These helpers answer the
second question with two classic descriptive measures — the two-sample
Kolmogorov-Smirnov statistic and the Population Stability Index.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import fmean, pstdev


@dataclass(frozen=True)
class DriftReport:
    """Distribution comparison between one training window and its test window.

    `mean_shift` is the test mean minus the train mean. `std_ratio` is the test
    population standard deviation over the train one, and is `math.inf` when the
    train window is constant while the test window is not. `n_bins` is the
    number of bins actually used, which can be smaller than the number requested
    when the train window has many tied values.
    """

    n_train: int
    n_test: int
    ks_statistic: float
    ks_p_value: float
    psi: float
    n_bins: int
    mean_shift: float
    std_ratio: float
    verdict: str


def _ks_statistic(train: list[float], test: list[float]) -> float:
    """Return the two-sample supremum distance between the empirical CDFs."""

    train_sorted = sorted(train)
    test_sorted = sorted(test)
    values = sorted(set(train_sorted + test_sorted))
    train_index = 0
    test_index = 0
    statistic = 0.0

    for value in values:
        while train_index < len(train_sorted) and train_sorted[train_index] <= value:
            train_index += 1
        while test_index < len(test_sorted) and test_sorted[test_index] <= value:
            test_index += 1
        train_cdf = train_index / len(train_sorted)
        test_cdf = test_index / len(test_sorted)
        statistic = max(statistic, abs(train_cdf - test_cdf))

    return float(min(1.0, max(0.0, statistic)))


def _ks_p_value(ks_statistic: float, n_train: int, n_test: int) -> float:
    """Return the asymptotic Kolmogorov p-value for a two-sample statistic."""

    en = math.sqrt(n_train * n_test / (n_train + n_test))
    lam = (en + 0.12 + 0.11 / en) * ks_statistic
    if lam == 0.0:
        return 1.0

    value = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return float(min(1.0, max(0.0, value)))


def _bin_edges(train: list[float], n_bins: int) -> list[float]:
    """Return ascending, de-duplicated equal-frequency edges taken from `train`."""

    sorted_train = sorted(train)
    last = len(sorted_train) - 1
    edges: list[float] = []

    for index in range(1, n_bins):
        position = min(last, round(index * len(sorted_train) / n_bins))
        edge = sorted_train[position]
        if not edges or edge != edges[-1]:
            edges.append(edge)
    return edges


def _bin_index(value: float, edges: list[float]) -> int:
    """Return the half-open bin `[edge_lo, edge_hi)` that `value` falls into."""

    for index, edge in enumerate(edges):
        if value < edge:
            return index
    return len(edges)


def _psi(train: list[float], test: list[float], n_bins: int) -> tuple[float, int]:
    """Return the population stability index and the number of bins used."""

    edges = _bin_edges(train, n_bins)
    bin_count = len(edges) + 1
    train_counts = [0] * bin_count
    test_counts = [0] * bin_count

    for value in train:
        train_counts[_bin_index(value, edges)] += 1
    for value in test:
        test_counts[_bin_index(value, edges)] += 1

    total = 0.0
    for train_count, test_count in zip(train_counts, test_counts, strict=True):
        train_fraction = max(float(train_count / len(train)), 1e-6)
        test_fraction = max(float(test_count / len(test)), 1e-6)
        total += (test_fraction - train_fraction) * math.log(test_fraction / train_fraction)

    return float(total), bin_count


def fold_drift(
    train: Sequence[float],
    test: Sequence[float],
    *,
    n_bins: int = 10,
    psi_moderate: float = 0.1,
    psi_severe: float = 0.25,
) -> DriftReport:
    """Return descriptive distribution diagnostics for one train/test fold.

    The PSI thresholds of 0.1 and 0.25 are the conventional industry rule of
    thumb for "worth a look" and "the population moved". These are descriptive
    diagnostics, not a hypothesis test of profitability: a stable verdict does
    not make a strategy good, and a severe one does not make it worthless.
    """

    train_values = [float(value) for value in train]
    test_values = [float(value) for value in test]

    if not train_values:
        raise ValueError("train must contain at least one observation")
    if not test_values:
        raise ValueError("test must contain at least one observation")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    ks_statistic = _ks_statistic(train_values, test_values)
    ks_p_value = _ks_p_value(ks_statistic, len(train_values), len(test_values))
    psi, actual_bins = _psi(train_values, test_values, n_bins)

    train_std = pstdev(train_values)
    test_std = pstdev(test_values)
    if train_std == 0.0:
        std_ratio = 1.0 if test_std == 0.0 else math.inf
    else:
        std_ratio = test_std / train_std

    if psi >= psi_severe:
        verdict = "severe drift"
    elif psi >= psi_moderate:
        verdict = "moderate drift"
    else:
        verdict = "stable"

    return DriftReport(
        n_train=len(train_values),
        n_test=len(test_values),
        ks_statistic=float(ks_statistic),
        ks_p_value=float(ks_p_value),
        psi=float(psi),
        n_bins=actual_bins,
        mean_shift=float(fmean(test_values) - fmean(train_values)),
        std_ratio=float(std_ratio),
        verdict=verdict,
    )


def drift_table(
    folds: Iterable[tuple[Sequence[float], Sequence[float]]],
    *,
    n_bins: int = 10,
    psi_moderate: float = 0.1,
    psi_severe: float = 0.25,
) -> list[DriftReport]:
    """Return one report per `(train, test)` fold, in order.

    An empty iterable returns an empty list rather than raising, so this can be
    dropped straight onto a splitter that produced no folds.
    """

    return [
        fold_drift(
            train,
            test,
            n_bins=n_bins,
            psi_moderate=psi_moderate,
            psi_severe=psi_severe,
        )
        for train, test in folds
    ]
