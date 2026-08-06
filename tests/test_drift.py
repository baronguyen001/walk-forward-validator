import math

import pytest

from walkforward import DriftReport, drift_table, fold_drift


def test_identical_windows_are_stable():
    values = [float(index) for index in range(50)]

    report = fold_drift(values, values)

    assert report.ks_statistic == 0.0
    assert report.ks_p_value == 1.0
    assert report.psi == pytest.approx(0.0, abs=1e-12)
    assert report.mean_shift == 0.0
    assert report.std_ratio == 1.0
    assert report.verdict == "stable"


def test_disjoint_ranges_have_maximum_ks_and_severe_drift():
    report = fold_drift([0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0])

    assert report.ks_statistic == 1.0
    assert report.verdict == "severe drift"


def test_constant_windows_have_expected_std_ratio():
    identical = fold_drift([1.0] * 10, [1.0] * 10)
    variable = fold_drift([1.0] * 10, [1.0, 2.0] * 5)

    assert identical.std_ratio == 1.0
    assert variable.std_ratio == math.inf


def test_ks_values_are_bounded_on_shifted_windows():
    train = [float(index) for index in range(20)]
    test = [float(index + 2) for index in range(20)]

    report = fold_drift(train, test)

    assert 0.0 <= report.ks_statistic <= 1.0
    assert 0.0 <= report.ks_p_value <= 1.0


def test_larger_mean_shift_has_larger_psi():
    train = [float(index) for index in range(20)]

    small_shift = fold_drift(train, [value + 0.25 for value in train])
    large_shift = fold_drift(train, [value + 5.0 for value in train])

    assert small_shift.psi < large_shift.psi


def test_psi_is_finite_with_empty_test_bins():
    report = fold_drift([float(index) for index in range(20)], [0.0, 19.0])

    assert math.isfinite(report.psi)


def test_fewer_observations_than_bins_does_not_raise():
    report = fold_drift([0.0, 1.0], [0.0, 1.0], n_bins=10)

    assert report.n_bins <= 10
    assert math.isfinite(report.psi)


def test_validation_errors():
    with pytest.raises(ValueError, match="train must contain at least one observation"):
        fold_drift([], [1.0])

    with pytest.raises(ValueError, match="test must contain at least one observation"):
        fold_drift([1.0], [])

    with pytest.raises(ValueError, match="n_bins must be >= 2"):
        fold_drift([1.0], [1.0], n_bins=1)


def test_drift_table_preserves_order_and_handles_empty_input():
    assert drift_table([]) == []

    folds = [
        ([0.0, 1.0], [0.0, 1.0]),
        ([0.0, 1.0], [1.0, 2.0]),
        ([0.0, 1.0], [2.0, 3.0]),
    ]

    reports = drift_table(folds)

    assert len(reports) == 3
    assert all(isinstance(report, DriftReport) for report in reports)
    assert reports[0].mean_shift < reports[1].mean_shift < reports[2].mean_shift


def test_verdicts_are_allowed_and_bin_count_is_bounded():
    allowed = {"stable", "moderate drift", "severe drift"}

    report = fold_drift([float(index) for index in range(10)], [20.0] * 10, n_bins=4)

    assert report.verdict in allowed
    assert report.n_bins <= 4
