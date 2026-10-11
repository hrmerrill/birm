"""Tests for the ragged :class:`birm.data.Panel` ingestion layer."""

import numpy as np
import pandas as pd

from birm.data import Panel


def _long() -> pd.DataFrame:
    """Return a small long-form table with hrv + speed metrics and ragged dates.

    Returns
    -------
    pd.DataFrame
        Long table covering load ``w``, daily-ish ``hrv`` (with ``trimp`` exog), and sparse
        ``speed`` (with ``temp``/``indoor`` exog).
    """
    rows = [
        ("2025-01-03", "w", 210.0, np.nan, np.nan, np.nan),
        ("2025-01-03", "hrv", 64.0, np.nan, np.nan, 210.0),
        ("2025-01-04", "hrv", 61.0, np.nan, np.nan, 185.0),
        ("2025-01-05", "w", 150.0, np.nan, np.nan, np.nan),
        ("2025-01-06", "speed", 4.71, 27.5, 1.0, np.nan),
    ]
    return pd.DataFrame(rows, columns=["date", "metric", "value", "temp", "indoor", "trimp"])


def _sheets() -> dict[str, pd.DataFrame]:
    """Return the equivalent wide multi-sheet workbook for the same data.

    Returns
    -------
    dict[str, pd.DataFrame]
        One sheet per metric, matching :func:`_long`.
    """
    w = pd.DataFrame({"date": ["2025-01-03", "2025-01-05"], "w": [210.0, 150.0]})
    hrv = pd.DataFrame(
        {"date": ["2025-01-03", "2025-01-04"], "hrv": [64.0, 61.0], "trimp": [210.0, 185.0]}
    )
    speed = pd.DataFrame({"date": ["2025-01-06"], "speed": [4.71], "temp": [27.5], "indoor": [1.0]})
    return {"w": w, "hrv": hrv, "speed": speed}


def test_from_long_structure():
    """Long ingestion builds the union grid, densifies load, and aligns metrics."""
    panel = Panel.from_long(
        _long(),
        date="date",
        metric="metric",
        value="value",
        load="w",
        exog={"speed": ["temp", "indoor"], "hrv": ["trimp"]},
    )
    assert panel.T == 4  # 2025-01-03 .. 2025-01-06
    assert panel.load.tolist() == [210.0, 0.0, 150.0, 0.0]  # missing day = rest
    assert panel.metrics["hrv"].idx.tolist() == [0, 1]
    assert panel.metrics["speed"].idx.tolist() == [3]
    assert panel.metrics["speed"].exog["temp"].tolist() == [27.5]
    assert panel.densest() == "hrv"


def test_long_and_sheets_agree():
    """Both constructors produce identical arrays for the same data."""
    a = Panel.from_long(
        _long(),
        date="date",
        metric="metric",
        value="value",
        load="w",
        exog={"speed": ["temp", "indoor"], "hrv": ["trimp"]},
    )
    b = Panel.from_sheets(
        _sheets(),
        date="date",
        load="w",
        exog={"speed": ["temp", "indoor"], "hrv": ["trimp"]},
    )
    assert a.grid.equals(b.grid)
    assert a.load.tolist() == b.load.tolist()
    for name in a.metrics:
        assert a.metrics[name].values.tolist() == b.metrics[name].values.tolist()
        assert a.metrics[name].idx.tolist() == b.metrics[name].idx.tolist()


def test_standardize_roundtrip():
    """Standardize z-scores responses/exog and rescales load to average-day units."""
    panel = Panel.from_long(
        _long(),
        date="date",
        metric="metric",
        value="value",
        load="w",
        exog={"hrv": ["trimp"]},
    ).standardize()
    hrv = panel.metrics["hrv"].values
    assert abs(float(hrv.mean())) < 1e-9
    center, scale = panel.scaling["hrv"]
    restored = hrv * scale + center
    assert np.allclose(restored, [64.0, 61.0])
    assert np.isclose(panel.load.mean(), 1.0)  # mean load -> 1.0


def test_negative_load_rejected():
    """Negative training load is a validation error with a helpful message."""
    df = pd.DataFrame(
        {"date": ["2025-01-03", "2025-01-04"], "metric": ["w", "w"], "value": [210.0, -5.0]}
    )
    try:
        Panel.from_long(df, date="date", metric="metric", value="value", load="w")
    except ValueError as e:
        assert ">= 0" in str(e)
    else:
        raise AssertionError("expected ValueError on negative load")


def test_bad_dates_rejected():
    """Unparseable dates raise a helpful ValueError naming the offenders."""
    df = pd.DataFrame(
        {"date": ["2025-01-03", "not-a-date"], "metric": ["w", "w"], "value": [210.0, 150.0]}
    )
    try:
        Panel.from_long(df, date="date", metric="metric", value="value", load="w")
    except ValueError as e:
        assert "not-a-date" in str(e) and "ISO" in str(e)
    else:
        raise AssertionError("expected ValueError on bad dates")


def test_duplicate_dates_rejected():
    """A metric observed twice on the same day is a validation error."""
    df = pd.DataFrame(
        {"date": ["2025-01-03", "2025-01-03"], "metric": ["hrv", "hrv"], "value": [1.0, 2.0]}
    )
    df = pd.concat([df, pd.DataFrame({"date": ["2025-01-04"], "metric": ["w"], "value": [10.0]})])
    try:
        Panel.from_long(df, date="date", metric="metric", value="value", load="w")
    except ValueError as e:
        assert "duplicate" in str(e)
    else:
        raise AssertionError("expected ValueError on duplicate dates")
