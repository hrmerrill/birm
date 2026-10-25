"""Tests for the :mod:`birm.core` fitness and fatigue stock builders."""

import numpy as np
import pytest

from birm.core.fatigue import fatigue_stock
from birm.core.fitness import fitness_stock


def _leaky(forcing: np.ndarray, tau: float) -> np.ndarray:
    """Compute the reference numpy leaky integrator ``I_t = e^{-1/tau} I_{t-1} + forcing_t``.

    Parameters
    ----------
    forcing : np.ndarray
        Per-day forcing sequence.
    tau : float
        Decay time constant in days.

    Returns
    -------
    np.ndarray
        The accumulated stock, same length as ``forcing``.
    """
    coef = np.exp(-1.0 / tau)
    out = np.empty_like(forcing, dtype=float)
    prev = 0.0
    for i, f in enumerate(forcing):
        prev = coef * prev + f
        out[i] = prev
    return out


def _leaky_causal(forcing: np.ndarray, tau: float) -> np.ndarray:
    """Strictly causal leaky integrator ``I_t = sum_{s<t} forcing_s e^{-(t-s)/tau}``.

    The response at day ``t`` depends only on days strictly before ``t`` (the same-day forcing is
    excluded), matching the documented ``sum_{s<t}`` stock convention.

    Parameters
    ----------
    forcing : np.ndarray
        Per-day forcing sequence.
    tau : float
        Decay time constant in days.

    Returns
    -------
    np.ndarray
        The strictly-past accumulated stock, same length as ``forcing``.
    """
    return _leaky(forcing, tau) - forcing


@pytest.fixture
def drive() -> np.ndarray:
    """Return a short training-load drive with rest days.

    Returns
    -------
    np.ndarray
        Daily load on the shared grid.
    """
    return np.array([1.0, 0.0, 2.0, 0.5, 0.0, 1.5, 0.0, 3.0], dtype=float)


def test_fatigue_banister_matches_oracle(drive: np.ndarray) -> None:
    """Banister fatigue equals ``k_G`` times the leaky load integral."""
    got = fatigue_stock(drive, k_G=1.5, tau_G=7.0, variant="banister").eval()
    assert np.allclose(got, 1.5 * _leaky_causal(drive, 7.0))


def test_fatigue_busso_matches_oracle(drive: np.ndarray) -> None:
    """Busso fatigue re-forces the tau_G integral with the tau_v-accumulated gain."""
    gain = _leaky_causal(drive, 3.0)
    expected = 1.5 * _leaky_causal(gain * drive, 7.0)
    got = fatigue_stock(drive, k_G=1.5, tau_G=7.0, variant="busso", tau_v=3.0).eval()
    assert np.allclose(got, expected)


def test_fatigue_busso_requires_tau_v(drive: np.ndarray) -> None:
    """Busso without a third timescale is an error."""
    with pytest.raises(ValueError, match="tau_v"):
        fatigue_stock(drive, k_G=1.5, tau_G=7.0, variant="busso")


def test_fatigue_unknown_variant(drive: np.ndarray) -> None:
    """An unrecognized fatigue variant is rejected."""
    with pytest.raises(ValueError, match="fatigue variant"):
        fatigue_stock(drive, k_G=1.5, tau_G=7.0, variant="nope")


def test_fitness_banister_matches_oracle(drive: np.ndarray) -> None:
    """Banister fitness equals ``k_F`` times the leaky load integral."""
    got = fitness_stock(drive, k_F=0.8, tau_F=40.0, variant="banister").eval()
    assert np.allclose(got, _leaky_causal(0.8 * drive, 40.0))


def _ceddia_oracle(drive: np.ndarray, forcing: np.ndarray, tau0: float, tau1: float) -> np.ndarray:
    """Brute-force ``sum_{s<t} forcing_s exp(-(t-s)/(tau0 + tau1 w_s))``.

    Parameters
    ----------
    drive : np.ndarray
        Per-day load setting each impulse's decay time.
    forcing : np.ndarray
        Per-day impulse.
    tau0 : float
        Zero-load decay time constant.
    tau1 : float
        Decay-time slope per unit load.

    Returns
    -------
    np.ndarray
        The accumulated stock.
    """
    out = np.zeros_like(drive)
    for t in range(len(drive)):
        for s in range(t):
            out[t] += forcing[s] * np.exp(-(t - s) / (tau0 + tau1 * drive[s]))
    return out


def test_diminishing_gain_matches_eq20(drive: np.ndarray) -> None:
    """M_F = k1/(k3-1) (1 - (1+w)^-(k3-1)) (Ceddia Eq. 20), Banister at 0, log at 1."""
    from birm.core.fitness import diminishing_gain

    expected = 0.8 / (0.5 - 1.0) * (1.0 - (1.0 + drive) ** -(0.5 - 1.0))
    assert np.allclose(diminishing_gain(drive, 0.8, 0.5).eval(), expected)
    assert np.allclose(diminishing_gain(drive, 0.8, 0.0).eval(), 0.8 * drive)
    assert np.allclose(diminishing_gain(drive, 0.8, 1.0).eval(), 0.8 * np.log1p(drive))


def test_diminishing_gain_gradient_finite_at_one() -> None:
    """The k_D -> 1 log limit keeps the gradient finite."""
    import pytensor
    import pytensor.tensor as pt

    from birm.core.fitness import diminishing_gain

    k_D = pt.scalar("k_D")
    grad = pytensor.function([k_D], pt.grad(diminishing_gain(np.array([2.0]), 1.0, k_D).sum(), k_D))
    assert np.isfinite(grad(1.0))


def test_fitness_ceddia_matches_oracle(drive: np.ndarray) -> None:
    """Ceddia fitness = power-law gain decaying with a load-dependent time constant."""
    forcing = 0.8 * ((1.0 + drive) ** 0.5 - 1.0) / 0.5
    expected = _ceddia_oracle(drive, forcing, 3.0, 2.0)
    got = fitness_stock(drive, 0.8, 3.0, variant="ceddia", k_D=0.5, tau_F1=2.0).eval()
    assert np.allclose(got, expected)


def test_fatigue_ceddia_matches_oracle(drive: np.ndarray) -> None:
    """Ceddia fatigue = linear gain decaying with a load-dependent time constant."""
    expected = 1.5 * _ceddia_oracle(drive, drive, 2.0, 0.5)
    got = fatigue_stock(drive, 1.5, 2.0, variant="ceddia", tau_G1=0.5).eval()
    assert np.allclose(got, expected)


def test_ceddia_null_recovers_banister(drive: np.ndarray) -> None:
    """With ``k_D = tau_F1 = tau_G1 = 0`` the Ceddia core is exactly Banister."""
    from birm.core.stock import impulse_response

    kw = {"k_F": 0.8, "tau_F": 40.0, "k_G": 1.5, "tau_G": 10.0}
    banister = impulse_response(drive, core="banister", **kw).eval()
    assert np.allclose(impulse_response(drive, core="ceddia", **kw).eval(), banister)


def test_ceddia_heavier_load_decays_slower() -> None:
    """A larger session leaves a longer-lived (relatively) fatigue trace."""
    light = fatigue_stock(np.array([1.0, 0, 0, 0, 0]), 1.0, 2.0, variant="ceddia", tau_G1=1.0)
    heavy = fatigue_stock(np.array([4.0, 0, 0, 0, 0]), 1.0, 2.0, variant="ceddia", tau_G1=1.0)
    light, heavy = light.eval(), heavy.eval() / 4.0
    assert heavy[-1] / heavy[1] > light[-1] / light[1]


def test_ceddia_no_training_is_zero() -> None:
    """With no training days the varying-decay kernel is empty and the stock is zero."""
    got = fatigue_stock(np.zeros(4), 1.0, 2.0, variant="ceddia", tau_G1=1.0).eval()
    assert np.allclose(got, 0.0)


def test_fitness_unknown_variant(drive: np.ndarray) -> None:
    """An unrecognized fitness variant is rejected."""
    with pytest.raises(ValueError, match="fitness variant"):
        fitness_stock(drive, k_F=0.8, tau_F=40.0, variant="nope")


def test_reparam_fitness_timescale() -> None:
    """tau_F = 1 + tau_G + lambda_f stays strictly above tau_G."""
    from birm.core.reparam import fitness_timescale

    assert fitness_timescale(10.0, 3.0) == 14.0
    assert fitness_timescale(10.0, 0.0) > 10.0  # +1 offset keeps F off the tau_F=tau_G ridge


def test_reparam_fatigue_timescale() -> None:
    """tau_G = tau_v + lambda_g (Busso) stays strictly above tau_v for lambda_g > 0."""
    from birm.core.reparam import fatigue_timescale

    assert fatigue_timescale(2.0, 5.0) == 7.0


def test_reparam_fatigue_gain_floors_ratio() -> None:
    """k_G = (theta + 1) k_F enforces k_G > k_F for theta >= 0."""
    from birm.core.reparam import fatigue_gain

    assert fatigue_gain(0.1, 0.0) == pytest.approx(0.1)  # theta=0 -> k_G == k_F (the floor)
    assert fatigue_gain(0.1, 3.0) == pytest.approx(0.4)
    assert fatigue_gain(0.1, 3.0) > 0.1


def test_reparam_decay_slope() -> None:
    """tau_1 = rho tau / w_ref lengthens tau by a factor (1 + rho) at the reference load."""
    from birm.core.reparam import decay_slope

    assert 10.0 + decay_slope(10.0, 0.5, 4.0) * 4.0 == pytest.approx(15.0)


def test_impulse_response_is_fitness_minus_fatigue(drive: np.ndarray) -> None:
    """R_t equals the assembled fitness stock minus the fatigue stock."""
    from birm.core.stock import impulse_response

    f = fitness_stock(drive, k_F=0.8, tau_F=40.0, variant="banister").eval()
    g = fatigue_stock(drive, k_G=1.5, tau_G=10.0, variant="banister").eval()
    got = impulse_response(drive, k_F=0.8, tau_F=40.0, k_G=1.5, tau_G=10.0, core="banister").eval()
    assert np.allclose(got, f - g)


def test_impulse_response_fitness_only(drive: np.ndarray) -> None:
    """With core='fitness-only' the stock is the fitness stock alone (g_t == 0)."""
    from birm.core.stock import impulse_response

    f = fitness_stock(drive, k_F=0.8, tau_F=40.0, variant="banister").eval()
    got = impulse_response(
        drive, k_F=0.8, tau_F=40.0, k_G=1.5, tau_G=10.0, core="fitness-only"
    ).eval()
    assert np.allclose(got, f)


def test_impulse_response_unknown_core(drive: np.ndarray) -> None:
    """An unrecognized core is rejected."""
    from birm.core.stock import impulse_response

    with pytest.raises(ValueError, match="core"):
        impulse_response(drive, k_F=0.8, tau_F=40.0, core="nope")


def test_impulse_response_busso_requires_tau_v(drive: np.ndarray) -> None:
    """Busso fatigue inside the assembled stock still requires tau_v."""
    from birm.core.stock import impulse_response

    with pytest.raises(ValueError, match="tau_v"):
        impulse_response(drive, k_F=0.8, tau_F=40.0, k_G=1.5, tau_G=10.0, core="busso")
