import pytest

from walkforward import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)


def _rows(train_scores, test_scores):
    rows = []
    for fold in range(3):
        for path, (train, test) in enumerate(zip(train_scores, test_scores, strict=True)):
            rows.append(
                {
                    "fold": fold,
                    "path": path,
                    "train": train + (fold * 0.001),
                    "test": test + (fold * 0.001),
                }
            )
    return rows


def test_pbo_is_bounded_between_zero_and_one():
    result = probability_of_backtest_overfitting(
        _rows([0.4, 0.8, 1.2], [0.45, 0.75, 0.35])
    )

    assert 0.0 <= result["pbo"] <= 1.0
    assert result["n_folds"] == 3
    assert len(result["logits"]) == 3


def test_pbo_is_higher_for_clearly_overfit_scores_than_robust_scores():
    robust = probability_of_backtest_overfitting(
        _rows([0.2, 0.5, 0.9], [0.25, 0.55, 0.95])
    )
    overfit = probability_of_backtest_overfitting(
        _rows([0.2, 0.5, 0.9], [0.95, 0.55, 0.10])
    )

    assert robust["pbo"] == pytest.approx(0.0)
    assert overfit["pbo"] == pytest.approx(1.0)
    assert overfit["pbo"] > robust["pbo"]
    assert robust["verdict"] == "robust"
    assert overfit["verdict"] == "overfit"


def test_probabilistic_sharpe_ratio_matches_known_value():
    result = probabilistic_sharpe_ratio(
        observed_sharpe=0.5,
        benchmark_sharpe=0.0,
        n=12,
        skew=0.0,
        kurtosis=3.0,
    )

    assert result["z_score"] == pytest.approx(1.5634719199)
    assert result["probabilistic_sharpe_ratio"] == pytest.approx(0.9410291823)


def test_deflated_sharpe_ratio_matches_known_value():
    result = deflated_sharpe_ratio(
        observed_sharpe=0.5,
        n=12,
        trials=2,
        skew=0.0,
        kurtosis=3.0,
    )

    assert result["benchmark_sharpe"] == pytest.approx(0.1662183175)
    assert result["z_score"] == pytest.approx(1.0437165759)
    assert result["deflated_sharpe_ratio"] == pytest.approx(0.8516917307)


def test_sharpe_helpers_accept_returns():
    returns = [0.01, 0.02, -0.01, 0.03, 0.015]

    psr = probabilistic_sharpe_ratio(returns)
    dsr = deflated_sharpe_ratio(returns, trials=3)

    assert 0.0 <= psr["probabilistic_sharpe_ratio"] <= 1.0
    assert 0.0 <= dsr["deflated_sharpe_ratio"] <= 1.0
