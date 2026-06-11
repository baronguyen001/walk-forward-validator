"""Command line entry points for walk-forward-validator."""

from __future__ import annotations

import argparse
from typing import Any

from .io import load_returns, load_scores
from .overfit import deflated_sharpe_ratio, probability_of_backtest_overfitting


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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
