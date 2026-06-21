"""Circular block bootstrap for time-series statistics.

A single backtest gives one number; the block bootstrap gives a confidence
interval around it that respects autocorrelation. It resamples *contiguous
blocks* of the return series (so short-range dependence survives), recomputes the
statistic on each resample, and reports the empirical interval. Pure stdlib and
fully deterministic for a given seed.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    std_error: float
    n_resamples: int
    block_size: int
    confidence: float


def _circular_blocks(values: Sequence[float], block_size: int, rng: Random) -> list[float]:
    """Build one resample of len(values) from random circular blocks."""
    n = len(values)
    out: list[float] = []
    while len(out) < n:
        start = rng.randrange(n)
        for offset in range(block_size):
            out.append(values[(start + offset) % n])
    return out[:n]


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (``q`` in [0, 1])."""
    if not sorted_values:
        raise ValueError("no values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[low]
    frac = pos - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def block_bootstrap(
    returns: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = statistics.mean,
    *,
    n_resamples: int = 1000,
    block_size: int | None = None,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap a confidence interval for ``statistic`` over ``returns``.

    ``block_size`` defaults to ``ceil(n ** (1/3))`` (a common rule of thumb).
    ``confidence`` is the two-sided interval width, e.g. ``0.95``.
    """
    values = [float(x) for x in returns]
    n = len(values)
    if n < 2:
        raise ValueError("need at least 2 observations")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if block_size is None:
        block_size = max(1, math.ceil(n ** (1 / 3)))
    elif block_size < 1:
        raise ValueError("block_size must be >= 1")

    rng = Random(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        sample = _circular_blocks(values, block_size, rng)
        estimates.append(float(statistic(sample)))
    estimates.sort()

    tail = (1 - confidence) / 2
    std_error = statistics.pstdev(estimates) if len(estimates) > 1 else 0.0
    return BootstrapResult(
        estimate=float(statistic(values)),
        lower=_percentile(estimates, tail),
        upper=_percentile(estimates, 1 - tail),
        std_error=std_error,
        n_resamples=n_resamples,
        block_size=block_size,
        confidence=confidence,
    )
