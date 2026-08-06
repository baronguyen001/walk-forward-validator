import math
from statistics import fmean

import pytest

from walkforward import whites_reality_check


def test_constant_positive_excess_has_zero_p_values():
    result = whites_reality_check([[0.01] * 30], n_resamples=200)

    assert result.p_value == 0.0
    assert result.per_strategy_p_values["strategy_0"] == 0.0
    assert result.n_strategies == 1
    assert result.n_observations == 30


def test_best_strategy_and_statistic():
    performance = [
        [0.01, 0.02, 0.00, 0.01],
        [0.03, 0.02, 0.04, 0.03],
        [-0.01, 0.00, 0.01, -0.01],
    ]

    result = whites_reality_check(performance, n_resamples=200)

    assert result.best_strategy == "strategy_1"
    assert result.best_statistic == math.sqrt(4) * fmean([0.03, 0.02, 0.04, 0.03])


def test_joint_p_value_is_at_least_best_individual_p_value():
    performance = {
        "first": [0.02, -0.01, 0.03, 0.00, 0.01, -0.02] * 5,
        "second": [-0.01, 0.04, -0.02, 0.01, 0.00, 0.03] * 5,
        "third": [0.00, 0.01, -0.01, 0.02, -0.03, 0.01] * 5,
    }

    result = whites_reality_check(performance, n_resamples=200, seed=7)

    assert result.p_value >= result.per_strategy_p_values[result.best_strategy]


def test_near_zero_mean_noise_is_not_significant():
    performance = [
        [0.02, -0.01, 0.00, 0.03, -0.04, 0.01, -0.02, 0.02, 0.01, -0.03] * 3,
        [-0.01, 0.04, 0.00, -0.02, 0.01, -0.03, 0.02, -0.01, 0.03, -0.02] * 3,
        [0.00, -0.02, 0.03, 0.01, -0.01, 0.02, -0.03, 0.00, 0.01, -0.02] * 3,
    ]

    result = whites_reality_check(performance, n_resamples=200, seed=3)

    assert 0.0 <= result.p_value <= 1.0
    assert result.p_value > 0.05


def test_perfectly_periodic_series_cannot_be_resampled_apart():
    # Every circular block of size 4 over a period-2 series holds two of each
    # value, so each resample reproduces the observed mean exactly.
    result = whites_reality_check([[0.01, -0.01] * 15], n_resamples=50, block_size=4)

    assert result.best_statistic == 0.0
    assert result.p_value == 0.0


def test_results_are_deterministic():
    performance = [[0.01, -0.02, 0.03, 0.00] * 8, [0.00, 0.01, -0.01, 0.02] * 8]

    first = whites_reality_check(performance, n_resamples=200, seed=11)
    second = whites_reality_check(performance, n_resamples=200, seed=11)

    assert first.p_value == second.p_value
    assert first.best_strategy == second.best_strategy


def test_benchmark_accepts_scalar_and_sequence():
    performance = [[0.11, 0.09, 0.10, 0.12]]

    scalar = whites_reality_check(performance, benchmark=0.1, n_resamples=200)
    sequence = whites_reality_check(performance, benchmark=[0.1] * 4, n_resamples=200)

    assert scalar == sequence
    with pytest.raises(ValueError, match="benchmark must match the series length"):
        whites_reality_check(performance, benchmark=[0.1], n_resamples=200)


def test_validation_errors():
    with pytest.raises(ValueError, match="performance must contain at least one strategy"):
        whites_reality_check({})
    with pytest.raises(ValueError, match="need at least 2 observations"):
        whites_reality_check([[0.1]])
    with pytest.raises(ValueError, match="all strategies must have the same length"):
        whites_reality_check([[0.1, 0.2], [0.1]])
    with pytest.raises(ValueError, match="n_resamples must be >= 1"):
        whites_reality_check([[0.1, 0.2]], n_resamples=0)
    with pytest.raises(ValueError, match="block_size must be >= 1"):
        whites_reality_check([[0.1, 0.2]], block_size=0)


def test_input_names_are_preserved_or_generated():
    mapping_result = whites_reality_check({"alpha": [0.1, 0.2]}, n_resamples=200)
    sequence_result = whites_reality_check([[0.1, 0.2], [0.2, 0.1]], n_resamples=200)

    assert list(mapping_result.per_strategy_p_values) == ["alpha"]
    assert list(sequence_result.per_strategy_p_values) == ["strategy_0", "strategy_1"]


def test_block_size_is_reported_and_honoured():
    result = whites_reality_check([[0.01, -0.01] * 16], n_resamples=50, block_size=4)

    assert result.block_size == 4
    assert result.n_resamples == 50
