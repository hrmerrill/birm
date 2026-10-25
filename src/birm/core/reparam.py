r"""Identifiability reparameterizations for the latent-stock timescales and gains.

The raw impulse-response parameter space is ill-conditioned: fitness and fatigue are antagonistic
filters on the same load, so the likelihood has a sharp ridge near $\tau_F = \tau_G$ and a
gain-ratio direction that is barely identified (Hellard 2006, Vermeire 2022). BIRM samples in a
reparameterized space that is one-to-one with the physiologically reasonable region and keeps the
model off the singular ridge:

- [fitness_timescale][] — $\tau_F = 1 + \tau_G + \lambda_F$ pins fitness strictly slower
  than fatigue (by at least a day) and off the $\tau_F = \tau_G$ singularity.
- [fatigue_timescale][] — $\tau_G = \tau_v + \lambda_G$ keeps the Busso accumulation
  timescale strictly fastest, giving the ordered $\tau_v < \tau_G < \tau_F$.
- [fatigue_gain][] — $k_G = (\theta + 1) k_F$ floors the gain ratio above one, hard
  enforcing $k_G > k_F$ (Peng 2024).
- [decay_slope][] — $\tau_1 = \rho\,\tau / \bar w$ makes the Ceddia decay-time slopes
  unit-free, so fitness outlasts fatigue at every load.

These are pure arithmetic transforms of already-sampled parameters: the non-negative margins
($\lambda_F, \lambda_G, \theta, \rho$) carry non-negative priors in the model layer, which is
what makes the constraints hold with certainty. Each works on plain floats (for tests and
diagnostics) or on PyTensor variables (inside the model graph).
"""

from __future__ import annotations

from typing import Any


def fitness_timescale(tau_G: Any, lambda_f: Any) -> Any:
    r"""Reparameterize the fitness timescale as $\tau_F = 1 + \tau_G + \lambda_F$.

    Parameters
    ----------
    tau_G : Any
        Fatigue decay time constant $\tau_G$ in days (float or scalar tensor).
    lambda_f : Any
        Non-negative fitness-slower-than-fatigue margin $\lambda_F$ in days.

    Returns
    -------
    Any
        The fitness decay time constant $\tau_F$, of the same kind as the inputs.
    """
    return 1.0 + tau_G + lambda_f


def fatigue_timescale(tau_v: Any, lambda_g: Any) -> Any:
    r"""Reparameterize the fatigue timescale as $\tau_G = \tau_v + \lambda_G$ (Busso).

    Parameters
    ----------
    tau_v : Any
        Busso accumulation time constant $\tau_v$ in days (float or scalar tensor).
    lambda_g : Any
        Non-negative fatigue-slower-than-accumulation margin $\lambda_G$ in days.

    Returns
    -------
    Any
        The fatigue decay time constant $\tau_G$, of the same kind as the inputs.
    """
    return tau_v + lambda_g


def fatigue_gain(k_F: Any, theta: Any) -> Any:
    r"""Reparameterize the fatigue gain as $k_G = (\theta + 1) k_F$.

    Parameters
    ----------
    k_F : Any
        Fitness gain $k_F$ (float or scalar tensor).
    theta : Any
        Non-negative gain-ratio driver $\theta$; $k_G / k_F = \theta + 1$.

    Returns
    -------
    Any
        The fatigue gain $k_G$, of the same kind as the inputs.
    """
    return (theta + 1.0) * k_F


def decay_slope(tau: Any, rho: Any, w_ref: float) -> Any:
    r"""Reparameterize a Ceddia decay-time slope as $\tau_1 = \rho\,\tau / \bar w$.

    The slope $\tau_1$ (days per unit load) in $\tau(w) = \tau + \tau_1 w$ depends on the load
    units. Sampling the dimensionless $\rho \ge 0$ instead makes the prior unit-free:
    $\tau(\bar w) = (1 + \rho)\,\tau$, so $\rho$ is the fractional lengthening of the decay time
    after a session of typical load $\bar w$. With $\rho_F = \rho_G + \lambda_\rho$ and
    $\tau_F > \tau_G$ this also gives $\tau_{F1} \ge \tau_{G1}$, so
    $\tau_F(w) > \tau_G(w)$ at every load.

    Parameters
    ----------
    tau : Any
        Zero-load decay time constant $\tau$ in days (float or scalar tensor).
    rho : Any
        Non-negative fractional lengthening $\rho$ at the reference load.
    w_ref : float
        Reference (typical training-day) load $\bar w > 0$.

    Returns
    -------
    Any
        The slope $\tau_1$ in days per unit load, of the same kind as the inputs.
    """
    return rho * tau / w_ref
