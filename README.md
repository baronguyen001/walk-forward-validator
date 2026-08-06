# walk-forward-validator

Version 0.6.0.

Walk-forward splits that will not let your backtest cheat.

![Fold preview](screenshots/folds_preview.svg)

Most time-series backtests leak by validating on data that touches or precedes the training
window. This package does one small thing: generate rolling or expanding pandas folds where
`train.index.max() < val.index.min()` is asserted before anything is yielded.

## Install

```bash
pip install walk-forward-validator
```

Plotting is optional:

```bash
pip install "walk-forward-validator[viz]"
```

## 30-second usage

```python
import pandas as pd
from walkforward import WalkForward

wf = WalkForward(train_days=365, val_days=90, step_days=30)
df = pd.read_csv("example.csv", parse_dates=["timestamp"], index_col="timestamp")

for train, val in wf.split(df):
    assert train.index.max() < val.index.min()
```

## Purged CV

For overlapping labels or features that look back across several bars, use purged splits.
`purge` removes bars between the train and test windows. `embargo` skips bars after each
test window before the next test window can start.

```python
from walkforward import PurgedWalkForwardSplit

splitter = PurgedWalkForwardSplit(train_size=365, test_size=90, purge=5, embargo=2)

for train_idx, test_idx in splitter.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    assert train.index.max() < test.index.min()
```

`PurgedKFold` is the scikit-learn-style version: `n_splits` contiguous test folds,
each trained on everything else minus a purge band around it and an embargo band
after it.

```python
from walkforward import PurgedKFold

kf = PurgedKFold(n_splits=5, purge=5, embargo=2)
for train_idx, test_idx in kf.split(df):
    ...
```

## Nested CV (tune inside, score once outside)

Picking hyper-parameters on the same folds you report is leakage with extra
steps. `NestedWalkForwardSplit` keeps the purged outer loop and carves expanding
inner folds out of each *training* window, so selection never sees the outer test
block. Inner indices are positions into the original series, so you can index the
same frame with both.

```python
from walkforward import NestedWalkForwardSplit

splitter = NestedWalkForwardSplit(train_size=365, test_size=90, inner_splits=3, purge=5)

for split in splitter.split(df):
    best = None
    for inner_train, inner_val in split.inner:
        ...  # tune on df.iloc[inner_train] / df.iloc[inner_val]
    # then score df.iloc[split.test_idx] exactly once with the chosen parameters
```

## Reality Check (did the winner just get lucky?)

A backtest that beat a hundred siblings is weaker evidence than one that beat
none. `whites_reality_check` resamples every candidate with the **same** circular
block draw, preserving the joint distribution of "best result found", and reports
how often luck alone reproduces the winner's edge.

```python
from walkforward import whites_reality_check

result = whites_reality_check(
    {"momentum": momo_returns, "meanrev": mr_returns, "carry": carry_returns},
    benchmark=0.0,
    n_resamples=2000,
)
print(result.best_strategy, result.p_value)          # snooping-adjusted
print(result.per_strategy_p_values[result.best_strategy])  # the flattering one
```

`p_value` is never smaller than the winner's own per-strategy p-value — searching
more candidates can only weaken the evidence.

## Fold drift (overfit, or a different world?)

When a fold degrades, `fold_drift` tells you whether the test window even came
from the same distribution: two-sample KS statistic and p-value, Population
Stability Index over equal-frequency train bins, mean shift, std ratio, and a
verdict.

```python
from walkforward import drift_table

pairs = [(df.iloc[tr]["ret"], df.iloc[te]["ret"]) for tr, te in splitter.split(df)]
for report in drift_table(pairs):
    print(report.psi, report.ks_statistic, report.verdict)
```

PSI thresholds of 0.1 and 0.25 are the conventional rule of thumb. These are
descriptive diagnostics, not a verdict on profitability.

## Confidence intervals (block bootstrap)

A single backtest number hides its own uncertainty. `block_bootstrap` resamples
contiguous blocks of the return series (preserving short-range autocorrelation)
to put a confidence interval around any statistic.

```python
import statistics
from walkforward import block_bootstrap

result = block_bootstrap(daily_returns, statistics.mean, n_resamples=2000, seed=0)
print(result.estimate, (result.lower, result.upper))
```

## Plotting

`fold_plot` returns a matplotlib `Figure`. The preview SVG in this README is generated
with the same helper.

```python
from walkforward import fold_plot, walk_forward_split

splits = walk_forward_split(df, train_days=365, val_days=90, step_days=90)
fig = fold_plot(splits)
fig.savefig("screenshots/folds_preview.svg")
```

## Metrics

`fold_stability` summarizes fold score mean/std, train-vs-test degradation percentage,
and a simple overfit flag.

```python
from walkforward import fold_stability

report = fold_stability({"train": train_scores, "test": test_scores})
print(report["degradation_pct"], report["overfit"])
```

## PBO and deflated Sharpe

For CPCV path scores or model-selection sweeps, `probability_of_backtest_overfitting`
estimates whether the best in-sample candidate tends to fall below the out-of-sample
median. The helper ranks the selected candidate out-of-sample per fold/path group,
applies the Lopez de Prado logit-rank transform, and reports the fraction of negative
logits as PBO.

```python
from walkforward import probability_of_backtest_overfitting

scores = [
    {"fold": 0, "path": "a", "train": 1.2, "test": 0.3},
    {"fold": 0, "path": "b", "train": 0.9, "test": 0.8},
    {"fold": 1, "path": "a", "train": 1.1, "test": 0.4},
    {"fold": 1, "path": "b", "train": 0.8, "test": 0.9},
]
print(probability_of_backtest_overfitting(scores)["pbo"])
```

`probabilistic_sharpe_ratio` and `deflated_sharpe_ratio` add skew/kurtosis-aware
Sharpe diagnostics. DSR deflates the hurdle for the number of strategies tried, which
keeps the package aligned with its anti-overfit purpose: a good backtest should survive
selection pressure, not just look good in one path.

```python
from walkforward import deflated_sharpe_ratio

returns = [0.01, 0.02, -0.01, 0.015, 0.005, 0.03]
print(deflated_sharpe_ratio(returns, trials=10)["deflated_sharpe_ratio"])
```

## CSV / JSON score loader

Use `load_scores` when you already have fold/path scores from your own research stack.
It reads stdlib-only CSV/JSON and normalizes common columns such as `train`/`test`,
`in_sample`/`out_of_sample`, and `is_score`/`oos_score` into the shape consumed by
`fold_stability` and the PBO helper.

```csv
fold,path,in_sample,out_of_sample,return
0,a,1.2,0.3,0.01
0,b,0.9,0.8,0.02
```

```python
from walkforward import load_scores, probability_of_backtest_overfitting

scores = load_scores("scores.csv")
print(probability_of_backtest_overfitting(scores)["verdict"])
```

The CLI exposes the same diagnostics:

```bash
walkforward pbo scores.csv
walkforward dsr returns.csv --trials 10
walkforward reality scores.csv --resamples 2000    # one row per (strategy, period)
walkforward drift train.csv test.csv --bins 10
```

## Combinatorial purged CV (CPCV)

A single walk-forward backtest gives you **one** out-of-sample path, so its
Sharpe/return estimate is noisy and easy to overfit. `CombinatorialPurgedSplit`
(López de Prado's CPCV) cuts the series into `n_groups` contiguous blocks, holds
out every combination of `test_groups` blocks as a test set, and reuses the same
purge + embargo logic to kill leakage. That yields `C(n_groups, test_groups)`
splits and `C(n_groups, test_groups) * test_groups / n_groups` reconstructed
backtest **paths** — so you can study the *distribution* of a metric instead of
trusting one number.

```python
from walkforward import CombinatorialPurgedSplit

splitter = CombinatorialPurgedSplit(n_groups=6, test_groups=2, purge=5, embargo=2)

# C(6, 2) = 15 purged splits, all leak-free.
for train_idx, test_idx in splitter.split(df):
    train, test = df.iloc[train_idx], df.iloc[test_idx]

# 5 backtest paths that each tile the whole series exactly once.
for path in splitter.paths(df):
    for train_idx, test_idx in path:
        ...  # score this block, then aggregate per path
```

## HTML report

`build_report` returns a single self-contained HTML string (pure string
templating, no JavaScript, no external links) with the fold/path bands, the
`fold_stability` metrics table, and a CPCV path-distribution summary. The
matplotlib chart is embedded inline as a base64 PNG when the `[viz]` extra is
installed, and degrades to a note without it — so the file always works offline.

```python
from walkforward import CombinatorialPurgedSplit, fold_stability, load_scores, write_report

splitter = CombinatorialPurgedSplit(n_groups=6, test_groups=2, purge=5)
pairs = [(df.iloc[tr], df.iloc[te]) for tr, te in splitter.split(df)]
path_scores = load_scores("scores.csv")  # optional: adds a PBO summary section

write_report(
    "report.html",
    pairs,
    title="CPCV run",
    metrics=fold_stability(test_scores),
    path_distribution={"n_paths": splitter.get_n_paths()},
    path_scores=path_scores,
)
```

## Arguments

| Arg | Meaning |
|---|---|
| `train_days` | Training window length. |
| `val_days` | Validation window length. |
| `step_days` | Days to advance between folds. |
| `expanding` | Keep `train_start` pinned to the series start. |

The public API is intentionally tiny: `Fold`, `walk_forward_split`, `WalkForward`, and
`classify_robustness`, plus `PurgedWalkForwardSplit`, `PurgedKFold`,
`CombinatorialPurgedSplit`, `NestedWalkForwardSplit`, `fold_plot`,
`fold_stability`, `load_scores`, `load_returns`, `block_bootstrap`,
`probability_of_backtest_overfitting`, `probabilistic_sharpe_ratio`,
`deflated_sharpe_ratio`, `whites_reality_check`, `fold_drift` / `drift_table`,
and `build_report` / `write_report`.

Used inside **[confluence-scanner](https://github.com/barobaonguyen/confluence-scanner)**.

Built by [barobaonguyen](https://github.com/barobaonguyen). Want the full **scrape -> AI -> alert** bot, not just this piece? -> **[Trawlkit](https://github.com/barobaonguyen)** (one-time kit).
