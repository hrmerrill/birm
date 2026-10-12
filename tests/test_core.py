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


@pytest.fixture
def drive() -> np.ndarray:
    """Return a short training-load drive with rest days.

    Returns
    -------
    np.ndarray
        Daily load on the shared grid.
    """
    return np.array([1.0, 0.0, 2.0, 0.5, 0.0, 1.5, 0.0, 3.0], dtype=float)


@pytest.fixture
def fatigue(drive: np.ndarray) -> np.ndarray:
    """Return a concrete fatigue stock to feed the periodization feedback.

    Parameters
    ----------
    drive : np.ndarray
        The daily load fixture.

    Returns
    -------
    np.ndarray
        A banister fatigue stock evaluated at ``k_G=1.5``, ``tau_G=7``.
    """
    return _leaky(1.5 * drive, 7.0)


def test_fatigue_banister_matches_oracle(drive: np.ndarray) -> None:
    """Banister fatigue equals ``k_G`` times the leaky load integral."""
    got = fatigue_stock(drive, k_G=1.5, tau_G=7.0, variant="banister").eval()
    assert np.allclose(got, 1.5 * _leaky(drive, 7.0))


def test_fatigue_none_is_zero(drive: np.ndarray) -> None:
    """The fitness-only (``none``) fatigue stock is identically zero."""
    got = fatigue_stock(drive, k_G=1.5, tau_G=7.0, variant="none").eval()
    assert np.allclose(got, 0.0)
    assert got.shape == drive.shape


def test_fatigue_busso_matches_oracle(drive: np.ndarray) -> None:
    """Busso fatigue re-forces the tau_G integral with the tau_v-accumulated gain."""
    gain = _leaky(drive, 3.0)
    expected = 1.5 * _leaky(gain * drive, 7.0)
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
    assert np.allclose(got, _leaky(0.8 * drive, 40.0))


@pytest.mark.parametrize("variant", ["ceddia_periodize", "ceddia_taper", "ceddia"])
def test_ceddia_null_recovers_banister(
    drive: np.ndarray, fatigue: np.ndarray, variant: str
) -> None:
    """With ``k_p = k_a = 0`` every Ceddia variant collapses to the Banister null."""
    banister = fitness_stock(drive, k_F=0.8, tau_F=40.0, variant="banister").eval()
    got = fitness_stock(
        drive, k_F=0.8, tau_F=40.0, variant=variant, g=fatigue, k_p=0.0, k_a=0.0
    ).eval()
    assert np.allclose(got, banister)


def test_fitness_taper_matches_oracle(drive: np.ndarray) -> None:
    """Tapering divides the gain by ``1 + k_a w`` before integrating."""
    forcing = 0.8 * drive / (1.0 + 0.3 * drive)
    got = fitness_stock(drive, k_F=0.8, tau_F=40.0, variant="ceddia_taper", k_a=0.3).eval()
    assert np.allclose(got, _leaky(forcing, 40.0))


def test_fitness_periodize_matches_oracle(drive: np.ndarray, fatigue: np.ndarray) -> None:
    """Periodization divides the gain by ``1 + k_p g_lag`` using yesterday's fatigue."""
    g_lag = np.concatenate([[0.0], fatigue[:-1]])
    forcing = 0.8 * drive / (1.0 + 0.5 * g_lag)
    got = fitness_stock(
        drive, k_F=0.8, tau_F=40.0, variant="ceddia_periodize", g=fatigue, k_p=0.5
    ).eval()
    assert np.allclose(got, _leaky(forcing, 40.0))


def test_periodize_without_fatigue_is_banister(drive: np.ndarray) -> None:
    """Periodization with no fatigue stock (``g=None``) warns and drops the feedback factor."""
    banister = fitness_stock(drive, k_F=0.8, tau_F=40.0, variant="banister").eval()
    with pytest.warns(UserWarning, match="periodization"):
        got = fitness_stock(
            drive, k_F=0.8, tau_F=40.0, variant="ceddia_periodize", g=None, k_p=0.5
        ).eval()
    assert np.allclose(got, banister)


def test_periodization_dampens_fitness(drive: np.ndarray, fatigue: np.ndarray) -> None:
    """Stronger periodization (larger ``k_p``) lowers the accumulated fitness stock."""
    weak = fitness_stock(
        drive, k_F=0.8, tau_F=40.0, variant="ceddia_periodize", g=fatigue, k_p=0.1
    ).eval()
    strong = fitness_stock(
        drive, k_F=0.8, tau_F=40.0, variant="ceddia_periodize", g=fatigue, k_p=1.0
    ).eval()
    assert (strong <= weak + 1e-9).all()
    assert strong[-1] < weak[-1]


def test_fitness_unknown_variant(drive: np.ndarray) -> None:
    """An unrecognized fitness variant is rejected."""
    with pytest.raises(ValueError, match="fitness variant"):
        fitness_stock(drive, k_F=0.8, tau_F=40.0, variant="nope")
