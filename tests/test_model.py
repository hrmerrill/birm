"""Tests for IRModel."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from birm import IRModel, Load, Metric, Panel, Prior
from birm.data import MetricData
from birm.model import _OFFSET_CAUSAL, _OFFSET_MAX


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
        core="banister",
        metrics=(Metric("hrv", exog=("w",), ar1=True),),
    )
    with pytest.warns(UserWarning, match="speed"):
        built = model.build_model(_panel())
    assert "obs_hrv" in _named(built)


@pytest.mark.parametrize("core", ["banister", "busso", "ceddia", "fitness-only"])
def test_build_model_cores(core: str) -> None:
    """Every core assembles with the expected reparameterized vars."""
    spec = IRModel(
        load=Load("w"),
        core=core,
        metrics=(Metric("hrv", exog=("w",), ar1=True), Metric("speed")),
    )
    names = _named(spec.build_model(_panel()))
    assert {"R", "tau_F", "tau_G", "k_F", "obs_hrv", "obs_speed"} <= names
    assert "admiss" not in names
    assert ("k_G" in names) == (core != "fitness-only")
    assert ("tau_v" in names) == (core == "busso")
    ceddia = {"k_D", "rho_G", "lambda_rho", "rho_F", "tau_F1", "tau_G1"}
    assert (ceddia <= names) == (core == "ceddia")
    assert ceddia.isdisjoint(names) == (core != "ceddia")
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
        core="banister",
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
        core="banister",
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
        core="banister",
        metrics=(Metric("hrv", ar1=True), Metric("speed")),
    )
    idata = spec.fit(
        _panel(),
        sampler={"draws": 5, "tune": 5, "chains": 1, "progressbar": False, "random_seed": 0},
    )
    assert "R" in idata.posterior
    assert "obs_hrv" in idata.observed_data


def test_unknown_core_rejected() -> None:
    """An unrecognized core raises ValueError at construction."""
    with pytest.raises(ValueError, match="core"):
        IRModel(load=Load("w"), core="nope", metrics=(Metric("hrv"),))


@pytest.mark.parametrize("core", ["banister", "busso", "ceddia", "fitness-only"])
def test_direct_parameterization_samples_raw_params(core: str) -> None:
    """``direct`` mode exposes k_G/tau_F/tau_G as free RVs and drops the reparam margins."""
    spec = IRModel(
        load=Load("w"),
        core=core,
        metrics=(Metric("hrv", ar1=True), Metric("speed")),
        parameterization="direct",
    )
    built = spec.build_model(_panel())
    names = _named(built)
    free = {v.name for v in built.free_RVs}
    assert {"R", "k_F", "tau_F", "tau_G"} <= names
    # No reparameterization margins in direct mode.
    assert {"theta", "lambda_f", "lambda_g", "lambda_rho"}.isdisjoint(names)
    # tau_F/tau_G are sampled directly rather than derived.
    assert {"tau_F", "tau_G"} <= free
    assert ("k_G" in free) == (core != "fitness-only")
    assert ("tau_v" in free) == (core == "busso")
    assert ({"k_D", "rho_F", "rho_G"} <= free) == (core == "ceddia")


def test_unknown_parameterization_rejected() -> None:
    """An unrecognized parameterization raises ValueError at construction."""
    with pytest.raises(ValueError, match="parameterization"):
        IRModel(
            load=Load("w"),
            core="banister",
            metrics=(Metric("hrv"),),
            parameterization="nope",
        )


def test_duplicate_and_empty_metrics_rejected() -> None:
    """Duplicate metric names and empty metric tuples raise ValueError."""
    with pytest.raises(ValueError, match="unique"):
        IRModel(
            load=Load("w"),
            core="banister",
            metrics=(Metric("hrv"), Metric("hrv")),
        )
    with pytest.raises(ValueError, match="at least one"):
        IRModel(load=Load("w"), core="banister", metrics=())


def test_bad_link_rejected() -> None:
    """An unknown link function raises ValueError."""
    with pytest.raises(ValueError, match="link"):
        Metric("hrv", link="softmax")


def test_missing_metric_and_exog_rejected() -> None:
    """A spec metric absent from the panel, or a missing exog column, raises ValueError."""
    absent = IRModel(load=Load("w"), core="banister", metrics=(Metric("power"),))
    with pytest.raises(ValueError, match="not found in panel"):
        absent.fit(_panel())

    bad_exog = IRModel(
        load=Load("w"),
        core="banister",
        metrics=(Metric("speed", exog=("temp",)),),
    )
    with pytest.raises(ValueError, match="exog"):
        bad_exog.fit(_panel())


def test_offset_rejects_empty_support() -> None:
    """``offset`` needs rows on the common support."""
    # the tiny 5-day panel has no grid index in [_OFFSET_MAX, T-1]; the common support is empty.
    spec = IRModel(
        load=Load("w"),
        core="banister",
        metrics=(Metric("hrv", exog=("w",), ar1=True), Metric("speed", offset=True)),
    )
    with pytest.raises(ValueError, match="common support"):
        spec.build_model(_panel())


def test_offset_marginalizes_and_recovers_known_shift() -> None:
    """The marginalized offset stays continuous and matches the r_{t-d} shift + softmax by hand.

    Single model build: the per-lag responsibilities ``offset_w_speed`` are reconstructed in NumPy
    from the model's own stock at its own point, using the documented shift ``r_{t-d}`` and a flat
    prior. Guards the shift direction/sign, the common-support masking, and the softmax, and checks
    no discrete variable enters the sampler.
    """
    rng = np.random.default_rng(0)
    t = 160
    load = (rng.random(t) < 0.3) * rng.gamma(2.0, 1.0, t)
    grid = pd.date_range("2024-01-01", periods=t, freq="D")
    speed_idx = np.arange(8, 155)  # straddles the common support [14, 145]
    speed_vals = rng.normal(size=len(speed_idx))

    panel = Panel(
        grid=grid,
        load=load,
        metrics={
            "hrv": MetricData(rng.normal(size=t), np.arange(t), {"w": load}),
            "speed": MetricData(speed_vals, speed_idx, {}),
        },
    )
    spec = IRModel(
        load=Load("w"),
        core="banister",
        metrics=(Metric("hrv", exog=("w",), ar1=True), Metric("speed", offset=True)),
    )
    model = spec.build_model(panel)

    # no discrete free RV -> default NUTS path preserved; the likelihood is a Potential, not an RV.
    assert all(v.name != "offset_d_speed" for v in model.free_RVs)
    assert "offset_w_speed" in _named(model)
    assert "obs_speed" not in {v.name for v in model.free_RVs}

    point = model.initial_point()
    stock, gamma, alpha, sigma, weights = model.compile_fn(
        [
            model["R"],
            model["gamma_speed"],
            model["alpha_speed"],
            model["sigma_speed"],
            model["offset_w_speed"],
        ],
        inputs=model.value_vars,
        on_unused_input="ignore",
    )(point)

    lags = np.arange(0 if _OFFSET_CAUSAL else -_OFFSET_MAX, _OFFSET_MAX + 1)
    hi = t - 1 + int(lags.min())
    keep = (speed_idx >= _OFFSET_MAX) & (speed_idx <= hi)  # common support only
    idx_v, y = speed_idx[keep], speed_vals[keep]
    assert keep.sum() < len(speed_idx)  # the masking actually drops the out-of-support rows
    log_prior = -0.5 * (lags / 7.0) ** 2  # 0-centered discrete Gaussian, scale _OFFSET_SCALE
    log_prior -= np.log(np.exp(log_prior).sum())
    ll = (
        np.array(
            [
                np.sum(
                    -0.5 * ((y - (alpha + gamma * stock[idx_v - s])) / sigma) ** 2
                    - np.log(sigma)
                    - 0.5 * np.log(2.0 * np.pi)
                )
                for s in lags
            ]
        )
        + log_prior
    )
    expected = np.exp(ll - ll.max())
    expected /= expected.sum()

    assert weights.shape == (len(lags),)
    assert weights.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(weights, expected, atol=1e-6)


def test_offset_with_ar1_builds_and_marginalizes() -> None:
    """A shifted channel may also carry its AR(1) residual; the lag stays marginalized, no RV."""
    rng = np.random.default_rng(1)
    t = 120
    load = (rng.random(t) < 0.3) * rng.gamma(2.0, 1.0, t)
    grid = pd.date_range("2024-01-01", periods=t, freq="D")
    panel = Panel(
        grid=grid,
        load=load,
        metrics={
            "hrv": MetricData(rng.normal(size=t), np.arange(t), {"w": load}),
            "speed": MetricData(rng.normal(size=t), np.arange(t), {}),
        },
    )
    spec = IRModel(
        load=Load("w"),
        core="banister",
        metrics=(Metric("hrv", exog=("w",), ar1=True, offset=True), Metric("speed")),
    )
    model = spec.build_model(panel)

    named = _named(model)
    assert "offset_w_hrv" in named
    assert "u_hrv" in named  # the AR(1) residual is still built and added contemporaneously
    assert all(v.name != "offset_d_hrv" for v in model.free_RVs)  # lag marginalized, not sampled
    assert np.isfinite(model.compile_logp()(model.initial_point()))
