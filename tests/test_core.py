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
    assert np.allclose(got, 1.5 * _leaky_causal(drive, 7.0))


def test_fatigue_none_is_zero(drive: np.ndarray) -> None:
    """The fitness-only (``none``) fatigue stock is identically zero."""
    got = fatigue_stock(drive, k_G=1.5, tau_G=7.0, variant="none").eval()
    assert np.allclose(got, 0.0)
    assert got.shape == drive.shape


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
    assert np.allclose(got, _leaky_causal(forcing, 40.0))


def test_fitness_periodize_matches_oracle(drive: np.ndarray, fatigue: np.ndarray) -> None:
    """Periodization divides the gain by ``1 + k_p g_s`` using the (causal) fatigue entering s."""
    forcing = 0.8 * drive / (1.0 + 0.5 * fatigue)
    got = fitness_stock(
        drive, k_F=0.8, tau_F=40.0, variant="ceddia_periodize", g=fatigue, k_p=0.5
    ).eval()
    assert np.allclose(got, _leaky_causal(forcing, 40.0))


def test_periodize_yesterday_gates_today() -> None:
    """A session yesterday (not only two+ days ago) reduces today's fitness gain."""
    back_to_back = np.array([1.0, 1.0, 0.0])
    g = fatigue_stock(back_to_back, k_G=1.0, tau_G=1.0).eval()
    gated = fitness_stock(
        back_to_back, k_F=1.0, tau_F=40.0, variant="ceddia_periodize", g=g, k_p=1.0
    ).eval()
    banister = fitness_stock(back_to_back, k_F=1.0, tau_F=40.0, variant="banister").eval()
    assert gated[-1] < banister[-1]


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


def test_reparam_admissible_fatigue_gain_floors_at_a_c() -> None:
    """k_G = (a_c + theta) k_F enforces k_G / k_F >= a_c for theta >= 0."""
    from birm.core.admissibility import admissibility_threshold
    from birm.core.reparam import admissible_fatigue_gain

    k_F, tau_F, tau_G = 0.1, 40.0, 10.0
    a_c = admissibility_threshold(tau_F, tau_G)
    assert admissible_fatigue_gain(k_F, 0.0, tau_F, tau_G) == pytest.approx(a_c * k_F)
    assert admissible_fatigue_gain(k_F, 0.5, tau_F, tau_G) == pytest.approx((a_c + 0.5) * k_F)
    assert admissible_fatigue_gain(k_F, 0.0, tau_F, tau_G) / k_F >= a_c


def test_admissibility_matches_formula() -> None:
    """The admissibility margin matches its closed form (numpy inputs -> numpy out)."""
    from birm.core.admissibility import admissibility

    k_F, k_G, tau_F, tau_G = 0.1, 0.4, 40.0, 10.0
    expected = k_G * (1.0 + 1.0 / (np.exp(1.0 / tau_G) - 1.0)) - k_F / (np.exp(1.0 / tau_F) - 1.0)
    assert admissibility(k_F, k_G, tau_F, tau_G) == pytest.approx(expected)


def test_admissibility_symbolic_inputs_return_tensor() -> None:
    """Symbolic inputs dispatch through to a PyTensor variable that evaluates to the margin."""
    import pytensor.tensor as pt

    from birm.core.admissibility import admissibility

    k_F = pt.scalar("k_F")
    out = admissibility(k_F, 0.4, 40.0, 10.0)
    assert hasattr(out, "eval")
    assert out.eval({k_F: 0.1}) == pytest.approx(admissibility(0.1, 0.4, 40.0, 10.0))


def test_admissibility_margin_sign_tracks_threshold() -> None:
    """Margin > 0 exactly when k_G/k_F exceeds the a_c gain-ratio floor."""
    from birm.core.admissibility import admissibility, admissibility_threshold

    tau_F, tau_G, k_F = 40.0, 10.0, 0.1
    a_c = admissibility_threshold(tau_F, tau_G)
    above = admissibility(k_F, (a_c + 0.5) * k_F, tau_F, tau_G)
    below = admissibility(k_F, (a_c - 0.1 if a_c > 0.1 else a_c * 0.5) * k_F, tau_F, tau_G)
    assert above > 0.0
    assert below < 0.0


def test_admissibility_threshold_above_one_when_tau_f_exceeds() -> None:
    """a_c > 1 approximately holds when tau_F > tau_G + 1 (the reparam's +1 offset)."""
    from birm.core.admissibility import admissibility_threshold

    assert admissibility_threshold(1.0 + 10.0 + 5.0, 10.0) > 1.0


def test_impulse_response_is_fitness_minus_fatigue(drive: np.ndarray) -> None:
    """R_t equals the assembled fitness stock minus the fatigue stock."""
    from birm.core.stock import impulse_response

    f = fitness_stock(drive, k_F=0.8, tau_F=40.0, variant="banister").eval()
    g = fatigue_stock(drive, k_G=1.5, tau_G=10.0, variant="banister").eval()
    got = impulse_response(
        drive, k_F=0.8, tau_F=40.0, k_G=1.5, tau_G=10.0, fitness="banister", fatigue="banister"
    ).eval()
    assert np.allclose(got, f - g)


def test_impulse_response_none_fatigue_is_fitness_only(drive: np.ndarray) -> None:
    """With fatigue='none' the stock is the fitness stock alone (g_t == 0)."""
    from birm.core.stock import impulse_response

    f = fitness_stock(drive, k_F=0.8, tau_F=40.0, variant="banister").eval()
    got = impulse_response(
        drive, k_F=0.8, tau_F=40.0, k_G=1.5, tau_G=10.0, fitness="banister", fatigue="none"
    ).eval()
    assert np.allclose(got, f)


def test_impulse_response_periodize_feeds_fatigue(drive: np.ndarray) -> None:
    """The ceddia_periodize fitness gets the model's own fatigue stock fed back (no warning)."""
    from birm.core.stock import impulse_response

    g = fatigue_stock(drive, k_G=1.5, tau_G=10.0, variant="banister").eval()
    f = fitness_stock(drive, k_F=0.8, tau_F=40.0, variant="ceddia_periodize", g=g, k_p=0.5).eval()
    got = impulse_response(
        drive,
        k_F=0.8,
        tau_F=40.0,
        k_G=1.5,
        tau_G=10.0,
        fitness="ceddia_periodize",
        fatigue="banister",
        k_p=0.5,
    ).eval()
    assert np.allclose(got, f - g)


def test_impulse_response_periodize_none_fatigue_warns(drive: np.ndarray) -> None:
    """ceddia_periodize + none fatigue drops the feedback and warns."""
    from birm.core.stock import impulse_response

    f = fitness_stock(drive, k_F=0.8, tau_F=40.0, variant="banister").eval()
    with pytest.warns(UserWarning, match="periodization"):
        got = impulse_response(
            drive,
            k_F=0.8,
            tau_F=40.0,
            k_G=1.5,
            tau_G=10.0,
            fitness="ceddia_periodize",
            fatigue="none",
            k_p=0.5,
        ).eval()
    assert np.allclose(got, f)


def test_impulse_response_busso_requires_tau_v(drive: np.ndarray) -> None:
    """Busso fatigue inside the assembled stock still requires tau_v."""
    from birm.core.stock import impulse_response

    with pytest.raises(ValueError, match="tau_v"):
        impulse_response(
            drive, k_F=0.8, tau_F=40.0, k_G=1.5, tau_G=10.0, fitness="banister", fatigue="busso"
        )
