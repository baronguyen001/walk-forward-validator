"""Command line entry points for walk-forward-validator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any

from .drift import fold_drift
from .io import load_returns, load_scores
from .overfit import deflated_sharpe_ratio, probability_of_backtest_overfitting
from .reality import whites_reality_check


def _print_kv(result: dict[str, Any]) -> None:
    for key, value in result.items():
        if key in {"logits", "selected"}:
            continue
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def _pbo(args: argparse.Namespace) -> int:
    result = probability_of_backtest_overfitting(
        load_scores(args.path),
        higher_is_better=not args.lower_is_better,
    )
    _print_kv(result)
    return 0


def _dsr(args: argparse.Namespace) -> int:
    result = deflated_sharpe_ratio(
        load_returns(args.path, column=args.column),
        trials=args.trials,
    )
    _print_kv(result)
    verdict = "passes" if result["deflated_sharpe_ratio"] >= args.threshold else "fails"
    print(f"verdict: {verdict} {args.threshold:.2f} probability threshold")
    return 0


def _group_by_strategy(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Group loaded score rows into one performance series per strategy."""

    name_key = next(
        (key for key in ("strategy", "candidate", "path") if any(key in row for row in rows)),
        None,
    )
    value_key = next(
        (key for key in ("return", "test", "out_of_sample") if any(key in row for row in rows)),
        None,
    )
    if value_key is None:
        raise ValueError("score data has no return, test, or out_of_sample column")

    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(value_key)
        if value is None:
            continue
        name = row.get(name_key) if name_key is not None else None
        grouped.setdefault("strategy_0" if name is None else str(name), []).append(float(value))
    if not grouped:
        raise ValueError("score data is empty")
    return grouped


def _reality(args: argparse.Namespace) -> int:
    result = whites_reality_check(
        _group_by_strategy(load_scores(args.path)),
        benchmark=args.benchmark,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    print(f"best_strategy: {result.best_strategy}")
    print(f"best_statistic: {result.best_statistic:.6f}")
    print(f"p_value: {result.p_value:.6f}")
    print(f"n_strategies: {result.n_strategies}")
    print(f"n_observations: {result.n_observations}")
    print(f"block_size: {result.block_size}")
    for name, value in result.per_strategy_p_values.items():
        print(f"p_value[{name}]: {value:.6f}")
    verdict = "survives" if result.p_value <= args.alpha else "fails"
    print(f"verdict: {verdict} the {args.alpha:.2f} data-snooping threshold")
    return 0


def _drift(args: argparse.Namespace) -> int:
    report = fold_drift(
        load_returns(args.train_path, column=args.column),
        load_returns(args.test_path, column=args.column),
        n_bins=args.bins,
    )
    _print_kv(asdict(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="walkforward")
    subcommands = parser.add_subparsers(dest="command", required=True)

    pbo = subcommands.add_parser("pbo", help="estimate probability of backtest overfitting")
    pbo.add_argument("path", help="CSV/JSON with train/test or in_sample/out_of_sample scores")
    pbo.add_argument(
        "--lower-is-better",
        action="store_true",
        help="treat lower scores as better, for losses/errors",
    )
    pbo.set_defaults(func=_pbo)

    dsr = subcommands.add_parser("dsr", help="compute deflated Sharpe probability")
    dsr.add_argument("path", help="CSV/JSON with a return, score, or test column")
    dsr.add_argument("--column", help="explicit return column to use")
    dsr.add_argument("--trials", type=int, default=1, help="number of tried strategies")
    dsr.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="probability threshold for the printed verdict",
    )
    dsr.set_defaults(func=_dsr)

    reality = subcommands.add_parser(
        "reality",
        help="run White's Reality Check across candidate strategies",
    )
    reality.add_argument("path", help="CSV/JSON with per-period scores, one row per observation")
    reality.add_argument(
        "--benchmark",
        type=float,
        default=0.0,
        help="constant benchmark subtracted from every series",
    )
    reality.add_argument(
        "--resamples",
        type=int,
        default=1000,
        help="number of bootstrap resamples",
    )
    reality.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    reality.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="p-value threshold for the printed verdict",
    )
    reality.set_defaults(func=_reality)

    drift = subcommands.add_parser(
        "drift",
        help="compare the train and test distributions of one fold",
    )
    drift.add_argument("train_path", help="CSV/JSON holding the training window values")
    drift.add_argument("test_path", help="CSV/JSON holding the test window values")
    drift.add_argument("--column", help="explicit value column to use in both files")
    drift.add_argument("--bins", type=int, default=10, help="number of equal-frequency PSI bins")
    drift.set_defaults(func=_drift)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
