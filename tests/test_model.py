"""Tests for IRModel."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from birm import IRModel, Load, Metric, Panel, Prior
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


def _named(model: Any) -> set[str]:
    """Return the set of named variable strings in a PyMC model."""
    return {v.name for v in model.named_vars.values()}


def test_fit_validates_and_warns_on_unused_metrics() -> None:
    """A valid spec warns on unused panel metrics while building the model."""
    model = IRModel(
        load=Load("w"),
        fitness="banister",
        fatigue="banister",
        metrics=(Metric("hrv", exog=("w",), ar1=True),),
    )
    with pytest.warns(UserWarning, match="speed"):
        built = model.build_model(_panel())
    assert "obs_hrv" in _named(built)


@pytest.mark.parametrize("fitness", ["banister", "ceddia_periodize", "ceddia_taper", "ceddia"])
@pytest.mark.parametrize("fatigue", ["banister", "busso", "none"])
def test_build_model_cores(fitness: str, fatigue: str) -> None:
    """Every fitness x fatigue core assembles with the expected reparam/admissibility vars."""
    spec = IRModel(
        load=Load("w"),
        fitness=fitness,
        fatigue=fatigue,
        metrics=(Metric("hrv", exog=("w",), ar1=True), Metric("speed")),
    )
    names = _named(spec.build_model(_panel()))
    assert {"R", "tau_F", "tau_G", "k_F", "obs_hrv", "obs_speed"} <= names
    assert ("admiss" in names) == (fatigue != "none")
    assert ("k_G" in names) == (fatigue != "none")
    assert ("tau_v" in names) == (fatigue == "busso")
    assert ("k_p" in names) == (fitness in ("ceddia_periodize", "ceddia"))
    assert ("k_a" in names) == (fitness in ("ceddia_taper", "ceddia"))
    # densest metric (hrv, 2 obs) is gamma-pinned; speed is estimated.
    assert "gamma_speed" in names
    assert "gamma_hrv" not in names
    # hrv carries the orthogonal AR(1) and an exog slope; speed carries neither.
    assert {"u_hrv", "wU_hrv", "beta_hrv", "alpha_hrv", "sigma_hrv"} <= names
    assert "u_speed" not in names and "beta_speed" not in names


def test_build_model_fixed_gamma_no_intercept_links() -> None:
    """Fixed gamma, dropped intercept, and non-identity links resolve without extra RVs."""
    spec = IRModel(
        load=Load("w"),
        fitness="banister",
        fatigue="banister",
        metrics=(
            Metric("hrv", link="log", intercept=False),
            Metric("speed", gamma=2.0),
        ),
    )
    names = _named(spec.build_model(_panel()))
    assert "alpha_hrv" not in names  # intercept dropped
    assert "gamma_speed" not in names  # fixed gamma is a constant, not an RV


def test_prior_override() -> None:
    """A ``priors`` entry replaces the default distribution for that name."""
    spec = IRModel(
        load=Load("w"),
        fitness="banister",
        fatigue="banister",
        metrics=(Metric("hrv"),),
        priors={"k_F": Prior("LogNormal", {"mu": 0.0, "sigma": 1.0})},
    )
    built = spec.build_model(_panel())
    k_f = next(v for v in built.free_RVs if v.name == "k_F")
    assert k_f.owner.op.name == "lognormal"


def test_fit_samples_inferencedata() -> None:
    """A tiny ``fit`` returns an InferenceData carrying the per-metric likelihood group."""
    spec = IRModel(
        load=Load("w"),
        fitness="banister",
        fatigue="banister",
        metrics=(Metric("hrv", ar1=True), Metric("speed")),
    )
    idata = spec.fit(
        _panel(),
        sampler={"draws": 5, "tune": 5, "chains": 1, "progressbar": False, "random_seed": 0},
    )
    assert "R" in idata.posterior
    assert "obs_hrv" in idata.observed_data


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
