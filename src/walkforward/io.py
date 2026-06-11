"""Small stdlib loaders for user-supplied walk-forward scores."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

_TRAIN_ALIASES = ("train", "train_score", "in_sample", "is_score")
_TEST_ALIASES = ("test", "test_score", "out_of_sample", "oos_score", "val", "validation")
_RETURN_ALIASES = ("return", "returns", "pnl", "score")


def _number_or_text(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def _first(row: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    lowered = {key.lower(): value for key, value in row.items()}
    for alias in aliases:
        if alias in lowered and lowered[alias] not in (None, ""):
            return lowered[alias]
    return None


def _canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    parsed = {key.strip().lstrip("\ufeff"): _number_or_text(value) for key, value in row.items()}
    canonical: dict[str, Any] = {}
    for key in ("fold", "path", "candidate", "strategy"):
        value = _first(parsed, (key,))
        if value is not None:
            canonical[key] = value
    train = _first(parsed, _TRAIN_ALIASES)
    test = _first(parsed, _TEST_ALIASES)
    ret = _first(parsed, _RETURN_ALIASES)
    if train is not None:
        canonical["train"] = train
        canonical["in_sample"] = train
    if test is not None:
        canonical["test"] = test
        canonical["out_of_sample"] = test
    if ret is not None:
        canonical["return"] = ret
    return canonical


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [_canonical_row(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        rows = payload.get("rows", payload.get("scores", payload.get("data")))
        if rows is None:
            rows = [payload]
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError("JSON score data must be a list of objects or a dict with rows")
    return [_canonical_row(row) for row in rows]


def load_scores(path: str | Path) -> list[dict[str, Any]]:
    """Load CSV/JSON fold or path scores into canonical row dictionaries.

    Recognized score columns include ``train``/``test``,
    ``in_sample``/``out_of_sample``, and ``is_score``/``oos_score``. The returned
    rows expose both ``train``/``test`` for :func:`walkforward.fold_stability`
    and ``in_sample``/``out_of_sample`` for
    :func:`walkforward.probability_of_backtest_overfitting`.
    """

    source = Path(path)
    if source.suffix.lower() == ".csv":
        return _read_csv(source)
    if source.suffix.lower() == ".json":
        return _read_json(source)
    raise ValueError("score data must be a .csv or .json file")


def load_returns(path: str | Path, *, column: str | None = None) -> list[float]:
    """Load a return/score column from CSV/JSON as floats for Sharpe diagnostics."""

    rows = load_scores(path)
    if not rows:
        raise ValueError("score data is empty")
    candidates = (column,) if column is not None else ("return", "test", "out_of_sample")
    values = []
    for row in rows:
        for candidate in candidates:
            if candidate in row and row[candidate] is not None:
                values.append(float(row[candidate]))
                break
        else:
            raise ValueError(f"row is missing return column candidates: {candidates}")
    return values
