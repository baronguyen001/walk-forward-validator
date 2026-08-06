# Changelog

## 0.6.0 - 2026-08-06

- Added `NestedWalkForwardSplit`, which wraps the purged outer walk-forward loop
  with expanding inner folds carved out of each training window, so
  hyper-parameters are chosen without ever touching the outer test block. Each
  `NestedSplit` carries the outer `train_idx`/`test_idx` plus the inner
  `(train, val)` pairs, all as positions into the original series.
- Added `whites_reality_check`, White's Reality Check for data-snooping bias. It
  resamples every candidate strategy with the *same* circular block draw, so the
  distribution of "best result found" is preserved, and reports both the
  snooping-adjusted p-value and the naive per-strategy p-values. The joint
  p-value is never smaller than the winner's own, which is the whole point.
- Added `fold_drift` and `drift_table`: per-fold train/test distribution
  diagnostics with the two-sample Kolmogorov-Smirnov statistic, its asymptotic
  p-value, the Population Stability Index over equal-frequency train bins, mean
  shift, standard-deviation ratio, and a stable/moderate/severe verdict. They
  answer "did this fold degrade because the strategy was overfit, or because the
  test window came from a different world?".
- Added `walkforward reality` and `walkforward drift` CLI subcommands over the
  existing stdlib CSV/JSON loaders.
- All three modules are pure standard library and deterministic; no new
  dependency.

### Maintenance

- Fixed the README version line, which still advertised 0.4.0 after the 0.5.0
  release.

## 0.5.0 - 2026-06-21

- Added `PurgedKFold`, a scikit-learn-style cross-validator with a purge band
  around and an embargo band after each contiguous test fold (same
  `split`/`get_n_splits` API as the other splitters), to remove
  overlapping-label leakage from naive K-fold on time series.
- Added `block_bootstrap` (circular block bootstrap) for a confidence interval
  around any time-series statistic, respecting short-range autocorrelation.
  Default block size `ceil(n ** 1/3)`, deterministic for a given seed, pure
  stdlib. Returns a `BootstrapResult` (estimate, lower, upper, std_error).

## 0.4.0 - 2026-06-11

- Added `probability_of_backtest_overfitting` for Lopez de Prado-style PBO from
  per-fold/per-path in-sample and out-of-sample score rows.
- Added `probabilistic_sharpe_ratio` and `deflated_sharpe_ratio` with stdlib
  fallbacks and optional NumPy acceleration when available.
- Added stdlib CSV/JSON score loaders (`load_scores`, `load_returns`) plus
  `walkforward pbo` and `walkforward dsr` CLI subcommands.
- Added an optional PBO summary section to HTML reports when `path_scores` or a
  precomputed `pbo_summary` is provided.
- Added PBO/DSR loader tests and a synthetic `examples/pbo_report.py` workflow.

## 0.3.0 - 2026-06-10

- Added `CombinatorialPurgedSplit` (CPCV): all `C(n_groups, test_groups)` purged
  combinations and a `paths()` helper that reconstructs the backtest paths for
  lower-variance metric estimates.
- Added `build_report` / `write_report`: a self-contained HTML report (no JS, no
  external resources) with fold bands, the fold-stability table, and a CPCV
  path-distribution summary. Embeds the matplotlib PNG when the `[viz]` extra is
  installed and degrades gracefully without it.
- Added `tests/test_cpcv.py`, `tests/test_report.py`, and `examples/cpcv.py`.

## 0.2.0 - 2026-06-03

- Added `PurgedWalkForwardSplit` for purge and embargo bar gaps.
- Added optional matplotlib fold plotting via `fold_plot`.
- Added fold stability metrics for mean/std, degradation, and overfit flags.
- Added purged CV and metrics tests plus a purged CV example.

## 0.1.0 - 2026-06-02

- Initial release with rolling and expanding walk-forward splits.
- Added robustness labels and leak-focused tests.
