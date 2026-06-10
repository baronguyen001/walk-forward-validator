import importlib.util

import pytest

from walkforward import CombinatorialPurgedSplit, build_report, fold_stability, write_report

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _example_splits():
    splitter = CombinatorialPurgedSplit(n_groups=5, test_groups=2, purge=2, embargo=1)
    return list(splitter.split(range(50)))


def test_report_is_valid_self_contained_html():
    html = build_report(_example_splits(), title="Demo report")

    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert html.count("<html") == 1 and html.count("</html>") == 1
    assert html.count("<body") == 1 and html.count("</body>") == 1
    # Self-contained: no external resources, no JavaScript.
    assert "http://" not in html
    assert "https://" not in html
    assert "<script" not in html.lower()
    assert "src=\"http" not in html


def test_report_embeds_title_and_section_headers():
    metrics = fold_stability({"train": [1.0, 0.9, 1.1], "test": [0.8, 0.7, 0.78]})
    distribution = {"n_paths": 4, "mean_sharpe": 1.23, "std_sharpe": 0.31}

    html = build_report(
        _example_splits(),
        title="My CPCV Run",
        metrics=metrics,
        path_distribution=distribution,
    )

    assert "My CPCV Run" in html
    assert "Fold stability" in html
    assert "CPCV path distribution" in html
    # Metric keys and a path-distribution field are rendered.
    assert "degradation_pct" in html
    assert "overfit" in html
    assert "n_paths" in html
    assert "mean_sharpe" in html


def test_report_escapes_user_supplied_title():
    html = build_report(_example_splits(), title="<script>alert(1)</script>")

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib (viz extra) not installed")
def test_report_embeds_png_when_matplotlib_available():
    html = build_report(_example_splits(), title="With figure")

    # PNG embedded inline as a base64 data URI; no external image reference.
    assert "data:image/png;base64," in html
    assert "<img" in html


def test_report_degrades_without_matplotlib(monkeypatch):
    # Force the figure helper to report no plotting backend.
    import walkforward.report as report_module

    monkeypatch.setattr(report_module, "_figure_data_uri", lambda splits: None)
    html = report_module.build_report(_example_splits(), title="No figure")

    assert html.startswith("<!doctype html>")
    assert "data:image/png" not in html
    assert "[viz]" in html  # mentions the optional extra in the degraded note


def test_report_handles_empty_splits():
    html = build_report([], title="Empty")

    assert html.startswith("<!doctype html>")
    assert "No folds provided." in html


def test_write_report_persists_file(tmp_path):
    out = tmp_path / "report.html"
    returned = write_report(str(out), _example_splits(), title="Saved")

    written = out.read_text(encoding="utf-8")
    assert written == returned
    assert written.startswith("<!doctype html>")
