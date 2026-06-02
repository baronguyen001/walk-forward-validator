# walk-forward-validator

Walk-forward splits that will not let your backtest cheat.

![Fold preview](screenshots/folds_preview.svg)

Most time-series backtests leak by validating on data that touches or precedes the training
window. This package does one small thing: generate rolling or expanding pandas folds where
`train.index.max() < val.index.min()` is asserted before anything is yielded.

## Install

```bash
pip install walk-forward-validator
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

## Arguments

| Arg | Meaning |
|---|---|
| `train_days` | Training window length. |
| `val_days` | Validation window length. |
| `step_days` | Days to advance between folds. |
| `expanding` | Keep `train_start` pinned to the series start. |

The public API is intentionally tiny: `Fold`, `walk_forward_split`, `WalkForward`, and
`classify_robustness`.

Used inside **[confluence-scanner](https://github.com/barobaonguyen/confluence-scanner)**.

Built by [barobaonguyen](https://github.com/barobaonguyen). Want the full **scrape -> AI -> alert** bot, not just this piece? -> **[Trawlkit](https://github.com/barobaonguyen)** (one-time kit).
