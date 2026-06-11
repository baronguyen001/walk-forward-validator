"""Overfit-aware backtest diagnostics.

The helpers in this module are intentionally small and dependency-light. They
target the question CPCV is meant to answer: did the best in-sample choice keep
its rank out-of-sample, or did the selection process mostly discover noise?

``probability_of_backtest_overfitting`` follows the Lopez de Prado PBO idea: for
each fold/path group, select the candidate with the best in-sample score, rank
that same candidate by out-of-sample score, transform the rank to a logit, and
estimate PBO as the fraction of negative logits. Values near 0 suggest the
selected candidate usually ranks above the out-of-sample median; values near 1
suggest selection is mostly overfit.

``probabilistic_sharpe_ratio`` and ``deflated_sharpe_ratio`` implement the
Bailey/Lopez de Prado Sharpe-ratio tests. They account for non-normal returns
through skew/kurtosis; DSR additionally raises the benchmark Sharpe for the
number of trials tried. These are statistical diagnostics, not proof of live
profitability: short samples, dependent returns, hidden parameter searches, and
non-stationary regimes can still make the probabilities too optimistic.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from math import erf, exp, isfinite, log, sqrt
from statistics import fmean
from typing import Any

_EULER_GAMMA = 0.5772156649015329
_SQRT_2 = sqrt(2.0)

try:  # optional speed/precision path; the stdlib fallback below is authoritative.
    import numpy as _np
except Exception:  # pragma: no cover - depends on the user's environment
    _np = None


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / _SQRT_2))


def _normal_ppf(probability: float) -> float:
    """Inverse standard-normal CDF using Peter J. Acklam's approximation."""

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between 0 and 1")
    if _np is not None:
        try:
            return float(_np.sqrt(2.0) * _np.erfinv((2.0 * probability) - 1.0))
        except AttributeError:
            pass

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    low = 0.02425
    high = 1.0 - low
    if probability < low:
        q = sqrt(-2.0 * log(probability))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if probability <= high:
        q = probability - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (
                ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r
                + 1.0
            )
        )
    q = sqrt(-2.0 * log(1.0 - probability))
    return -(
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )


def _as_float(value: Any, *, name: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _lookup(row: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for alias in aliases:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    raise KeyError(aliases[0])


def _coerce_pbo_rows(data: Iterable[Any] | Mapping[str, Iterable[Any]]) -> list[dict[str, Any]]:
    if isinstance(data, Mapping):
        train = data.get("in_sample", data.get("train", data.get("train_scores")))
        test = data.get("out_of_sample", data.get("test", data.get("test_scores")))
        if train is None or test is None:
            raise ValueError("PBO data must include in_sample/train and out_of_sample/test")
        train_values = list(train)
        test_values = list(test)
        folds = None if data.get("fold") is None else list(data["fold"])
        paths = None if data.get("path") is None else list(data["path"])
        rows = []
        for index, (train_score, test_score) in enumerate(
            zip(train_values, test_values, strict=True)
        ):
            rows.append(
                {
                    "fold": None if folds is None else folds[index],
                    "path": index if paths is None else paths[index],
                    "in_sample": train_score,
                    "out_of_sample": test_score,
                }
            )
        return rows

    rows = []
    for index, item in enumerate(data):
        if isinstance(item, Mapping):
            rows.append(
                {
                    "fold": item.get("fold", item.get("split", item.get("group", 0))),
                    "path": item.get("path", item.get("candidate", item.get("strategy", index))),
                    "in_sample": _lookup(
                        item, ("in_sample", "is_score", "train", "train_score")
                    ),
                    "out_of_sample": _lookup(
                        item, ("out_of_sample", "oos_score", "test", "test_score", "val")
                    ),
                }
            )
            continue

        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) == 2:
                rows.append(
                    {
                        "fold": 0,
                        "path": index,
                        "in_sample": item[0],
                        "out_of_sample": item[1],
                    }
                )
                continue
            if len(item) == 3:
                rows.append(
                    {
                        "fold": item[0],
                        "path": index,
                        "in_sample": item[1],
                        "out_of_sample": item[2],
                    }
                )
                continue
            if len(item) == 4:
                rows.append(
                    {
                        "path": item[0],
                        "fold": item[1],
                        "in_sample": item[2],
                        "out_of_sample": item[3],
                    }
                )
                continue
        raise TypeError("PBO rows must be mappings or 2/3/4-item sequences")
    return rows


def _rank_worst_to_best(values: Sequence[float], selected_index: int) -> float:
    selected = values[selected_index]
    less = sum(value < selected for value in values)
    equal = sum(value == selected for value in values)
    return less + ((equal + 1.0) / 2.0)


def probability_of_backtest_overfitting(
    data: Iterable[Any] | Mapping[str, Iterable[Any]],
    *,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Estimate Probability of Backtest Overfitting from IS/OOS score rows.

    Args:
        data: Rows containing in-sample and out-of-sample scores. Accepted forms
            include mappings with ``train``/``test`` or
            ``in_sample``/``out_of_sample`` keys, ``(train, test)`` pairs,
            ``(fold, train, test)`` tuples, or a column mapping.
        higher_is_better: Set ``False`` for loss/error metrics.

    Returns:
        A dictionary with ``pbo`` in ``[0, 1]``, the per-fold logit ranks, and a
        simple verdict string.
    """

    rows = _coerce_pbo_rows(data)
    if len(rows) < 2:
        raise ValueError("PBO requires at least two candidate rows")

    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["fold"]].append(row)

    logits: list[float] = []
    selected: list[dict[str, Any]] = []
    sign = 1.0 if higher_is_better else -1.0
    for fold, fold_rows in grouped.items():
        if len(fold_rows) < 2:
            continue
        in_scores = [sign * _as_float(row["in_sample"], name="in_sample") for row in fold_rows]
        out_scores = [
            sign * _as_float(row["out_of_sample"], name="out_of_sample") for row in fold_rows
        ]
        best_index = max(range(len(fold_rows)), key=lambda index: in_scores[index])
        rank = _rank_worst_to_best(out_scores, best_index)
        percentile = rank / (len(fold_rows) + 1.0)
        logit = log(percentile / (1.0 - percentile))
        logits.append(logit)
        selected.append(
            {
                "fold": fold,
                "path": fold_rows[best_index]["path"],
                "out_of_sample_rank": rank,
                "out_of_sample_percentile": percentile,
                "logit": logit,
            }
        )

    if not logits:
        raise ValueError("PBO requires at least one fold with two or more candidates")

    pbo = sum(value < 0.0 for value in logits) / len(logits)
    if pbo >= 0.7:
        verdict = "overfit"
    elif pbo <= 0.3:
        verdict = "robust"
    else:
        verdict = "mixed"
    return {
        "pbo": pbo,
        "n_folds": len(logits),
        "logit_rank_mean": fmean(logits),
        "logits": logits,
        "selected": selected,
        "verdict": verdict,
    }


def _coerce_returns(returns: Iterable[float]) -> list[float]:
    values = [_as_float(value, name="returns") for value in returns]
    if len(values) < 2:
        raise ValueError("returns must contain at least two observations")
    return values


def _moments(returns: Sequence[float]) -> tuple[float, float, float, float]:
    if _np is not None:
        arr = _np.asarray(returns, dtype=float)
        mean = float(arr.mean())
        centered = arr - mean
        std = float(arr.std(ddof=1))
        if std <= 0.0:
            raise ValueError("returns must have non-zero variance")
        skew = float(_np.mean((centered / std) ** 3))
        kurtosis = float(_np.mean((centered / std) ** 4))
        return mean, std, skew, kurtosis

    mean = fmean(returns)
    n = len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (n - 1)
    if variance <= 0.0:
        raise ValueError("returns must have non-zero variance")
    std = sqrt(variance)
    skew = sum(((value - mean) / std) ** 3 for value in returns) / n
    kurtosis = sum(((value - mean) / std) ** 4 for value in returns) / n
    return mean, std, skew, kurtosis


def _sharpe_inputs(
    returns: Iterable[float] | None,
    observed_sharpe: float | None,
    n: int | None,
    skew: float | None,
    kurtosis: float | None,
) -> tuple[float, int, float, float]:
    if returns is not None:
        values = _coerce_returns(returns)
        mean, std, sample_skew, sample_kurtosis = _moments(values)
        sr = mean / std if observed_sharpe is None else _as_float(
            observed_sharpe, name="observed_sharpe"
        )
        return (
            sr,
            len(values) if n is None else int(n),
            sample_skew if skew is None else _as_float(skew, name="skew"),
            sample_kurtosis if kurtosis is None else _as_float(kurtosis, name="kurtosis"),
        )

    if observed_sharpe is None or n is None:
        raise ValueError("provide returns or both observed_sharpe and n")
    if n < 2:
        raise ValueError("n must be >= 2")
    return (
        _as_float(observed_sharpe, name="observed_sharpe"),
        int(n),
        0.0 if skew is None else _as_float(skew, name="skew"),
        3.0 if kurtosis is None else _as_float(kurtosis, name="kurtosis"),
    )


def _sr_variance(observed_sharpe: float, n: int, skew: float, kurtosis: float) -> float:
    numerator = 1.0 - (skew * observed_sharpe) + (((kurtosis - 1.0) / 4.0) * observed_sharpe**2)
    if numerator <= 0.0:
        raise ValueError("Sharpe variance term must be positive")
    return numerator / (n - 1)


def probabilistic_sharpe_ratio(
    returns: Iterable[float] | None = None,
    *,
    observed_sharpe: float | None = None,
    benchmark_sharpe: float = 0.0,
    n: int | None = None,
    skew: float | None = None,
    kurtosis: float | None = None,
) -> dict[str, float | int]:
    """Return the Probabilistic Sharpe Ratio against a benchmark Sharpe."""

    sr, sample_count, sample_skew, sample_kurtosis = _sharpe_inputs(
        returns, observed_sharpe, n, skew, kurtosis
    )
    benchmark = _as_float(benchmark_sharpe, name="benchmark_sharpe")
    variance = _sr_variance(sr, sample_count, sample_skew, sample_kurtosis)
    z_score = (sr - benchmark) / sqrt(variance)
    return {
        "probabilistic_sharpe_ratio": _normal_cdf(z_score),
        "z_score": z_score,
        "observed_sharpe": sr,
        "benchmark_sharpe": benchmark,
        "n": sample_count,
        "skew": sample_skew,
        "kurtosis": sample_kurtosis,
    }


def _expected_max_sharpe_threshold(
    observed_sharpe: float,
    n: int,
    skew: float,
    kurtosis: float,
    trials: int,
) -> float:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if trials == 1:
        return 0.0
    sigma_sr = sqrt(_sr_variance(observed_sharpe, n, skew, kurtosis))
    term_1 = (1.0 - _EULER_GAMMA) * _normal_ppf(1.0 - (1.0 / trials))
    term_2 = _EULER_GAMMA * _normal_ppf(1.0 - (1.0 / (trials * exp(1.0))))
    return sigma_sr * (term_1 + term_2)


def deflated_sharpe_ratio(
    returns: Iterable[float] | None = None,
    *,
    observed_sharpe: float | None = None,
    n: int | None = None,
    trials: int = 1,
    skew: float | None = None,
    kurtosis: float | None = None,
) -> dict[str, float | int]:
    """Return the Deflated Sharpe Ratio probability adjusted for trials."""

    sr, sample_count, sample_skew, sample_kurtosis = _sharpe_inputs(
        returns, observed_sharpe, n, skew, kurtosis
    )
    benchmark = _expected_max_sharpe_threshold(
        sr, sample_count, sample_skew, sample_kurtosis, int(trials)
    )
    psr = probabilistic_sharpe_ratio(
        observed_sharpe=sr,
        benchmark_sharpe=benchmark,
        n=sample_count,
        skew=sample_skew,
        kurtosis=sample_kurtosis,
    )
    return {
        "deflated_sharpe_ratio": psr["probabilistic_sharpe_ratio"],
        "z_score": psr["z_score"],
        "observed_sharpe": sr,
        "benchmark_sharpe": benchmark,
        "n": sample_count,
        "trials": int(trials),
        "skew": sample_skew,
        "kurtosis": sample_kurtosis,
    }
