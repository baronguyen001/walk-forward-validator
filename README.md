# walk-forward-validator

Version 0.2.0.

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

## Arguments

| Arg | Meaning |
|---|---|
| `train_days` | Training window length. |
| `val_days` | Validation window length. |
| `step_days` | Days to advance between folds. |
| `expanding` | Keep `train_start` pinned to the series start. |

The public API is intentionally tiny: `Fold`, `walk_forward_split`, `WalkForward`, and
`classify_robustness`, plus `PurgedWalkForwardSplit`, `fold_plot`, and `fold_stability`.

Used inside **[confluence-scanner](https://github.com/barobaonguyen/confluence-scanner)**.

Built by [barobaonguyen](https://github.com/barobaonguyen). Want the full **scrape -> AI -> alert** bot, not just this piece? -> **[Trawlkit](https://github.com/barobaonguyen)** (one-time kit).
