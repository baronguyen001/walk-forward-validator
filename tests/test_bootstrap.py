from __future__ import annotations

import statistics

import pytest

from walkforward import block_bootstrap


def _series() -> list[float]:
    # deterministic, mildly autocorrelated-looking series
    return [((-1) ** i) * 0.01 + 0.005 * (i % 7) for i in range(120)]


def test_estimate_equals_statistic_on_full_sample() -> None:
    data = _series()
    result = block_bootstrap(data, statistics.mean, n_resamples=200, seed=1)
    assert result.estimate == pytest.approx(statistics.mean(data))


def test_interval_brackets_estimate() -> None:
    result = block_bootstrap(_series(), statistics.mean, n_resamples=500, seed=1)
    assert result.lower <= result.estimate <= result.upper
    assert result.confidence == 0.95


def test_deterministic_for_same_seed() -> None:
    a = block_bootstrap(_series(), n_resamples=300, seed=7)
    b = block_bootstrap(_series(), n_resamples=300, seed=7)
    assert (a.lower, a.upper, a.std_error) == (b.lower, b.upper, b.std_error)


def test_seed_changes_result() -> None:
    a = block_bootstrap(_series(), n_resamples=300, seed=1)
    b = block_bootstrap(_series(), n_resamples=300, seed=2)
    assert (a.lower, a.upper) != (b.lower, b.upper)


def test_default_block_size_is_cube_root() -> None:
    result = block_bootstrap(_series(), n_resamples=10, seed=0)
    assert result.block_size == 5  # ceil(120 ** (1/3)) == 5


def test_custom_statistic() -> None:
    result = block_bootstrap([1.0, 2.0, 3.0, 4.0], max, n_resamples=50, seed=0)
    assert result.estimate == 4.0
    assert result.upper <= 4.0


def test_validation() -> None:
    with pytest.raises(ValueError):
        block_bootstrap([1.0])  # too few
    with pytest.raises(ValueError):
        block_bootstrap([1.0, 2.0], confidence=1.5)
    with pytest.raises(ValueError):
        block_bootstrap([1.0, 2.0], block_size=0)
