"""Self-contained HTML report for walk-forward / CPCV runs.

Pure string templating: no JavaScript, no templating engine, no CDN links. The
returned HTML is a single string you can write to disk and open offline. When
matplotlib is installed (the optional ``[viz]`` extra) the fold/path band figure
from :func:`walkforward.plot.fold_plot` is embedded inline as a base64 PNG data
URI, so the file stays fully self-contained. Without ``[viz]`` the report still
renders; the figure section degrades to a short note instead.
"""

from __future__ import annotations

import base64
import html
import io
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

_STYLE = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
margin:0;padding:2rem;background:#f6f8fa;color:#1f2328;}
.wrap{max-width:880px;margin:0 auto;}
h1{font-size:1.5rem;margin:0 0 .25rem;}
.sub{color:#57606a;margin:0 0 1.5rem;font-size:.9rem;}
section{background:#fff;border:1px solid #d0d7de;border-radius:8px;
padding:1.25rem 1.5rem;margin-bottom:1.25rem;}
h2{font-size:1.05rem;margin:0 0 .75rem;}
table{border-collapse:collapse;width:100%;font-size:.9rem;}
th,td{text-align:left;padding:.4rem .6rem;border-bottom:1px solid #eaeef2;}
th{color:#57606a;font-weight:600;}
td.num{text-align:right;font-variant-numeric:tabular-nums;}
.flag-ok{color:#1a7f37;font-weight:600;}
.flag-bad{color:#cf222e;font-weight:600;}
img{max-width:100%;height:auto;border:1px solid #eaeef2;border-radius:6px;}
.note{color:#57606a;font-size:.85rem;}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:.3rem 1.25rem;
font-size:.9rem;}
.kv dt{color:#57606a;}
.kv dd{margin:0;font-variant-numeric:tabular-nums;}
footer{color:#57606a;font-size:.8rem;text-align:center;margin-top:1.5rem;}
""".strip()


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return _esc(value)


class _Indexed:
    """Minimal carrier exposing a list ``index`` so ``fold_plot`` can read it.

    ``CombinatorialPurgedSplit`` yields bare integer-position lists. ``fold_plot``
    reads an object's ``.index`` (pandas-style), so we wrap those lists; ``Fold``
    objects and ``(train, test)`` DataFrame pairs pass through untouched.
    """

    __slots__ = ("index",)

    def __init__(self, positions: Sequence[Any]) -> None:
        self.index = list(positions)


def _is_scalar_sequence(obj: Any) -> bool:
    if isinstance(obj, (list, tuple, range)) and not isinstance(obj, str):
        return all(isinstance(item, (int, float)) for item in obj)
    return False


def _normalize_split(split: Any) -> Any:
    if isinstance(split, Sequence) and not isinstance(split, str) and len(split) == 2:
        left, right = split
        if _is_scalar_sequence(left) and _is_scalar_sequence(right):
            return _Indexed(left), _Indexed(right)
    return split


def _figure_data_uri(splits: Sequence[Any]) -> str | None:
    """Render ``fold_plot`` to a base64 PNG data URI, or ``None`` if unavailable."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .plot import fold_plot
    except Exception:
        return None

    try:
        fig = fold_plot([_normalize_split(split) for split in splits])
    except Exception:
        return None

    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=110)
    finally:
        plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _metrics_table(metrics: Mapping[str, Any]) -> str:
    rows = []
    for key, value in metrics.items():
        css = ""
        if key == "overfit":
            css = ' class="flag-bad"' if value else ' class="flag-ok"'
        rows.append(
            f"<tr><th>{_esc(key)}</th>"
            f'<td class="num"><span{css}>{_fmt(value)}</span></td></tr>'
        )
    return (
        "<table><thead><tr><th>metric</th><th>value</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _path_summary_block(path_distribution: Mapping[str, Any]) -> str:
    items = "".join(
        f"<dt>{_esc(key)}</dt><dd>{_fmt(value)}</dd>"
        for key, value in path_distribution.items()
    )
    return f'<dl class="kv">{items}</dl>'


def _pbo_summary_block(summary: Mapping[str, Any]) -> str:
    keys = ("pbo", "verdict", "n_folds", "logit_rank_mean")
    visible = {key: summary[key] for key in keys if key in summary}
    return _path_summary_block(visible)


def build_report(
    splits: Iterable[Any],
    *,
    title: str = "Walk-forward validation report",
    metrics: Mapping[str, Any] | None = None,
    path_distribution: Mapping[str, Any] | None = None,
    path_scores: Iterable[Any] | Mapping[str, Iterable[Any]] | None = None,
    pbo_summary: Mapping[str, Any] | None = None,
    subtitle: str | None = None,
) -> str:
    """Return a complete, self-contained HTML document as a string.

    Args:
        splits: Folds to draw as train/test bands. Each item is a ``Fold`` or a
            ``(train, test)`` pair, matching :func:`walkforward.plot.fold_plot`.
        title: Document and ``<h1>`` title.
        metrics: Fold-stability summary (e.g. the dict from
            :func:`walkforward.metrics.fold_stability`); rendered as a table.
        path_distribution: CPCV path-distribution summary (e.g. number of paths,
            mean/std of a per-path metric); rendered as a key/value block.
        path_scores: Optional per-path/per-fold in-sample and out-of-sample score
            rows. When provided, a PBO section is computed and embedded.
        pbo_summary: Optional precomputed summary from
            :func:`walkforward.overfit.probability_of_backtest_overfitting`.
        subtitle: Optional line shown under the title; defaults to a UTC stamp.

    The figure is embedded inline when matplotlib is available and degrades to a
    note otherwise, so the output never references external resources.
    """

    parsed = list(splits)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    sub = subtitle if subtitle is not None else f"Generated {stamp}"

    data_uri = _figure_data_uri(parsed) if parsed else None
    if data_uri is not None:
        figure_html = f'<img alt="Walk-forward fold bands" src="{data_uri}" />'
    elif parsed:
        figure_html = (
            '<p class="note">Fold figure unavailable. Install the plotting extra '
            "(<code>pip install &quot;walk-forward-validator[viz]&quot;</code>) "
            "to embed the band chart.</p>"
        )
    else:
        figure_html = '<p class="note">No folds provided.</p>'

    sections = [
        f'<section><h2>Folds ({len(parsed)})</h2>{figure_html}</section>',
    ]
    if metrics:
        sections.append(
            f"<section><h2>Fold stability</h2>{_metrics_table(metrics)}</section>"
        )
    if path_distribution:
        sections.append(
            "<section><h2>CPCV path distribution</h2>"
            f"{_path_summary_block(path_distribution)}</section>"
        )
    if pbo_summary is None and path_scores is not None:
        from .overfit import probability_of_backtest_overfitting

        pbo_summary = probability_of_backtest_overfitting(path_scores)
    if pbo_summary:
        sections.append(
            "<section><h2>Probability of backtest overfitting</h2>"
            f"{_pbo_summary_block(pbo_summary)}</section>"
        )

    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{_STYLE}</style></head>"
        '<body><div class="wrap">'
        f"<h1>{_esc(title)}</h1>"
        f'<p class="sub">{_esc(sub)}</p>'
        f"{''.join(sections)}"
        "<footer>Generated by walk-forward-validator</footer>"
        "</div></body></html>"
    )


def write_report(path: str, splits: Iterable[Any], **kwargs: Any) -> str:
    """Build the report and write it to ``path`` (UTF-8). Returns the HTML string."""

    document = build_report(splits, **kwargs)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)
    return document
