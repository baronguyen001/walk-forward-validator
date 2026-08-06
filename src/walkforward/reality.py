"""White's Reality Check for data-snooping bias across many candidate strategies.

A backtest that beat a hundred siblings is not the same evidence as a backtest
that beat none. The Reality Check resamples every candidate with the *same*
circular block draw, so the joint distribution of "best result found" is
preserved, and reports how often pure luck reproduces the winner's edge.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random
from statistics import fmean


@dataclass(frozen=True)
class RealityCheckResult:
    """Outcome of a Reality Check over a set of candidate strategies.

    `p_value` is the data-snooping-adjusted probability that the best observed
    performance could have been produced by luck alone across all candidates.
    Each entry of `per_strategy_p_values` is the naive single-strategy
    probability that ignores the search, so it is always the more flattering
    number of the two.
    """

    best_strategy: str
    best_statistic: float
    p_value: float
    per_strategy_p_values: dict[str, float]
    n_strategies: int
    n_observations: int
    n_resamples: int
    block_size: int


def _draw_indices(rng: Random, n: int, block_size: int) -> list[int]:
    """Draw one circular-block index sequence of length `n`."""

    indices: list[int] = []
    while len(indices) < n:
        start = rng.randrange(n)
        indices.extend((start + offset) % n for offset in range(block_size))
    return indices[:n]


def whites_reality_check(
    performance: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
    *,
    benchmark: Sequence[float] | float = 0.0,
    n_resamples: int = 1000,
    block_size: int | None = None,
    seed: int = 0,
) -> RealityCheckResult:
    """Compute White's Reality Check using one shared circular block bootstrap.

    `performance` is either a mapping of name to per-period performance, or a
    sequence of such series which are named `strategy_0`, `strategy_1`, and so
    on. `benchmark` is subtracted from every series and may be a constant or a
    series of the same length. The same resampled positions are applied to every
    candidate, which is what keeps the test a joint one.
    """

    if isinstance(performance, Mapping):
        strategies = {
            str(name): [float(value) for value in series] for name, series in performance.items()
        }
    else:
        strategies = {
            f"strategy_{index}": [float(value) for value in series]
            for index, series in enumerate(performance)
        }

    if not strategies:
        raise ValueError("performance must contain at least one strategy")

    lengths = {len(series) for series in strategies.values()}
    if len(lengths) != 1:
        raise ValueError("all strategies must have the same length")
    n = next(iter(lengths))
    if n < 2:
        raise ValueError("need at least 2 observations")

    if isinstance(benchmark, Sequence) and not isinstance(benchmark, (str, bytes)):
        benchmark_values = [float(value) for value in benchmark]
        if len(benchmark_values) != n:
            raise ValueError("benchmark must match the series length")
    else:
        benchmark_values = [float(benchmark)] * n

    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if block_size is not None and block_size < 1:
        raise ValueError("block_size must be >= 1")

    resample_block = max(1, math.ceil(n ** (1 / 3))) if block_size is None else block_size
    excess = {
        name: [
            float(value) - float(benchmark_value)
            for value, benchmark_value in zip(series, benchmark_values, strict=True)
        ]
        for name, series in strategies.items()
    }
    means = {name: fmean(values) for name, values in excess.items()}
    observed = {name: float(math.sqrt(n) * mean) for name, mean in means.items()}
    best_strategy = max(observed, key=observed.__getitem__)
    best_statistic = float(observed[best_strategy])

    rng = Random(seed)
    joint_exceedances = 0
    individual_exceedances = dict.fromkeys(strategies, 0)

    for _ in range(n_resamples):
        indices = _draw_indices(rng, n, resample_block)
        centered = {
            name: float(math.sqrt(n) * (fmean(values[index] for index in indices) - means[name]))
            for name, values in excess.items()
        }
        if max(centered.values()) > best_statistic:
            joint_exceedances += 1
        for name, value in centered.items():
            if value > observed[name]:
                individual_exceedances[name] += 1

    return RealityCheckResult(
        best_strategy=best_strategy,
        best_statistic=best_statistic,
        p_value=float(joint_exceedances / n_resamples),
        per_strategy_p_values={
            name: float(count / n_resamples) for name, count in individual_exceedances.items()
        },
        n_strategies=len(strategies),
        n_observations=n,
        n_resamples=n_resamples,
        block_size=resample_block,
    )
