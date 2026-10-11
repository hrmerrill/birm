"""Tests for IRModel."""

import numpy as np
import pandas as pd
import pytest

from birm import IRModel, Load, Metric, Panel
from birm.data import MetricData


def _panel() -> Panel:
    """Return a tiny panel with a daily-ish ``hrv`` (``w`` exog) and sparse ``speed``."""
    grid = pd.date_range("2025-01-01", periods=5, freq="D")
    return Panel(
        grid=grid,
        load=np.ones(5),
        metrics={
            "hrv": MetricData(np.array([1.0, 2.0]), np.array([0, 2]), {"w": np.array([1.0, 1.0])}),
            "speed": MetricData(np.array([3.0]), np.array([4]), {}),
        },
    )


def test_fit_validates_then_raises_not_implemented() -> None:
    """A valid spec warns on unused metrics, then raises NotImplementedError for sampling."""
    model = IRModel(
        load=Load("w"),
        fitness="banister",
        fatigue="banister",
        metrics=(Metric("hrv", exog=("w",), ar1=True),),
    )
    with pytest.warns(UserWarning, match="speed"), pytest.raises(NotImplementedError):
        model.fit(_panel())


def test_unknown_core_rejected() -> None:
    """Unrecognized fitness/fatigue cores raise ValueError at construction."""
    with pytest.raises(ValueError, match="fitness"):
        IRModel(load=Load("w"), fitness="nope", fatigue="banister", metrics=(Metric("hrv"),))
    with pytest.raises(ValueError, match="fatigue"):
        IRModel(load=Load("w"), fitness="banister", fatigue="nope", metrics=(Metric("hrv"),))


def test_duplicate_and_empty_metrics_rejected() -> None:
    """Duplicate metric names and empty metric tuples raise ValueError."""
    with pytest.raises(ValueError, match="unique"):
        IRModel(
            load=Load("w"),
            fitness="banister",
            fatigue="banister",
            metrics=(Metric("hrv"), Metric("hrv")),
        )
    with pytest.raises(ValueError, match="at least one"):
        IRModel(load=Load("w"), fitness="banister", fatigue="banister", metrics=())


def test_bad_link_rejected() -> None:
    """An unknown link function raises ValueError."""
    with pytest.raises(ValueError, match="link"):
        Metric("hrv", link="softmax")


def test_missing_metric_and_exog_rejected() -> None:
    """A spec metric absent from the panel, or a missing exog column, raises ValueError."""
    absent = IRModel(
        load=Load("w"), fitness="banister", fatigue="banister", metrics=(Metric("power"),)
    )
    with pytest.raises(ValueError, match="not found in panel"):
        absent.fit(_panel())

    bad_exog = IRModel(
        load=Load("w"),
        fitness="banister",
        fatigue="banister",
        metrics=(Metric("speed", exog=("temp",)),),
    )
    with pytest.raises(ValueError, match="exog"):
        bad_exog.fit(_panel())
