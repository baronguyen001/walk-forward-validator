# Changelog

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
