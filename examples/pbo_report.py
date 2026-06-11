"""Synthetic PBO/DSR example that writes a self-contained HTML report."""

from __future__ import annotations

import csv
from pathlib import Path

from walkforward import (
    CombinatorialPurgedSplit,
    deflated_sharpe_ratio,
    load_returns,
    load_scores,
    probability_of_backtest_overfitting,
    write_report,
)


def write_synthetic_scores(path: Path) -> None:
    rows = [
        {"fold": 0, "path": "stable", "in_sample": 0.82, "out_of_sample": 0.78, "return": 0.012},
        {"fold": 0, "path": "fragile", "in_sample": 1.15, "out_of_sample": 0.31, "return": -0.004},
        {"fold": 0, "path": "plain", "in_sample": 0.70, "out_of_sample": 0.69, "return": 0.008},
        {"fold": 1, "path": "stable", "in_sample": 0.80, "out_of_sample": 0.76, "return": 0.011},
        {"fold": 1, "path": "fragile", "in_sample": 1.12, "out_of_sample": 0.28, "return": -0.006},
        {"fold": 1, "path": "plain", "in_sample": 0.72, "out_of_sample": 0.70, "return": 0.007},
        {"fold": 2, "path": "stable", "in_sample": 0.84, "out_of_sample": 0.79, "return": 0.013},
        {"fold": 2, "path": "fragile", "in_sample": 1.18, "out_of_sample": 0.33, "return": -0.003},
        {"fold": 2, "path": "plain", "in_sample": 0.71, "out_of_sample": 0.68, "return": 0.006},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fold", "path", "in_sample", "out_of_sample", "return"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    example_dir = Path(__file__).resolve().parent
    score_path = example_dir / "pbo_scores.csv"
    write_synthetic_scores(score_path)

    scores = load_scores(score_path)
    pbo = probability_of_backtest_overfitting(scores)
    dsr = deflated_sharpe_ratio(load_returns(score_path), trials=3)

    splitter = CombinatorialPurgedSplit(n_groups=6, test_groups=2, purge=2, embargo=1)
    splits = list(splitter.split(range(120)))
    write_report(
        str(example_dir / "pbo_report.html"),
        splits,
        title="PBO demo report",
        path_distribution={
            "n_splits": splitter.get_n_splits(range(120)),
            "n_paths": splitter.get_n_paths(),
            "dsr": dsr["deflated_sharpe_ratio"],
        },
        pbo_summary=pbo,
    )
    print(f"PBO: {pbo['pbo']:.3f} ({pbo['verdict']})")
    print(f"DSR: {dsr['deflated_sharpe_ratio']:.3f}")
    print("wrote examples/pbo_report.html")
