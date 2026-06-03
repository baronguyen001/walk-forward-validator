import pytest

from walkforward import fold_stability


def test_fold_stability_summarizes_train_and_test_scores():
    report = fold_stability(
        {"train": [1.0, 0.9, 1.1], "test": [0.8, 0.75, 0.78]},
        overfit_threshold_pct=10,
    )

    assert report["n"] == 3
    assert report["train_mean"] == pytest.approx(1.0)
    assert report["test_mean"] == pytest.approx(0.7766666667)
    assert report["test_std"] == pytest.approx(0.0205480467)
    assert report["degradation_pct"] == pytest.approx(22.33333333)
    assert report["overfit"] is True


def test_fold_stability_accepts_test_only_scores():
    report = fold_stability([0.71, 0.73, 0.72])

    assert report["mean"] == pytest.approx(0.72)
    assert report["std"] == pytest.approx(0.0081649658)
    assert report["train_mean"] is None
    assert report["degradation_pct"] is None
    assert report["overfit"] is False


def test_fold_stability_supports_lower_is_better_metrics():
    report = fold_stability(
        [(0.4, 0.5), (0.3, 0.36)],
        higher_is_better=False,
        overfit_threshold_pct=20,
    )

    assert report["train_mean"] == pytest.approx(0.35)
    assert report["test_mean"] == pytest.approx(0.43)
    assert report["degradation_pct"] == pytest.approx(22.85714286)
    assert report["overfit"] is True


def test_fold_stability_rejects_length_mismatch():
    with pytest.raises(ValueError):
        fold_stability([0.8, 0.7], train_scores=[0.9])
