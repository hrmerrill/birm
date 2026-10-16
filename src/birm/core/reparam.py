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
- [admissible_fatigue_gain][] — $k_G = (a_c + \theta) k_F$ floors the gain ratio at the
  Ceddia (2026) admissibility metric $a_c$ instead of one, hard enforcing
  $P(k_G/k_F > a_c) = 1$.

These are pure arithmetic transforms of already-sampled parameters: the non-negative margins
($\lambda_F, \lambda_G, \theta$) carry ``HalfNormal`` priors in the model layer, which is
what makes the constraints hold with certainty. Each works on plain floats (for tests and
diagnostics) or on PyTensor variables (inside the model graph).
"""

from __future__ import annotations

from typing import Any

from birm.core.admissibility import admissibility_threshold


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


def admissible_fatigue_gain(k_F: Any, theta: Any, tau_F: Any, tau_G: Any) -> Any:
    r"""Reparameterize the fatigue gain as $k_G = (a_c + \theta) k_F$.

    Identical in structure to [fatigue_gain][], but floors the gain ratio at the Ceddia (2026)
    admissibility metric $a_c = \frac{1 - e^{-1/\tau_G}}{e^{1/\tau_F} - 1}$ rather than at one.
    With a non-negative margin $\theta$ this enforces $k_G / k_F = a_c + \theta \ge a_c$, so the
    posterior probability of admissibility $P(k_G / k_F > a_c) = 1$. The fitness timescale keeps
    the same $\tau_F = 1 + \tau_G + \lambda_F$ reparameterization.

    Parameters
    ----------
    k_F : Any
        Fitness gain $k_F$ (float or scalar tensor).
    theta : Any
        Non-negative gain-ratio margin $\theta$; $k_G / k_F = a_c + \theta$.
    tau_F : Any
        Fitness decay time constant $\tau_F$ in days (float or scalar tensor).
    tau_G : Any
        Fatigue decay time constant $\tau_G$ in days (float or scalar tensor).

    Returns
    -------
    Any
        The fatigue gain $k_G$, of the same kind as the inputs.
    """
    return (admissibility_threshold(tau_F, tau_G) + theta) * k_F
