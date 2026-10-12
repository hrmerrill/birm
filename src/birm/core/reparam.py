r"""Identifiability reparameterizations for the latent-stock timescales and gains.

The raw impulse-response parameter space is ill-conditioned: fitness and fatigue are antagonistic
filters on the same load, so the likelihood has a sharp ridge near :math:`\tau_F = \tau_G` and a
gain-ratio direction that is barely identified (Hellard 2006, Vermeire 2022). BIRM samples in a
reparameterized space that is one-to-one with the physiologically reasonable region and keeps the
model off the singular ridge:

- :func:`fitness_timescale` — :math:`\tau_F = 1 + \tau_G + \lambda_F` pins fitness strictly slower
  than fatigue (by at least a day) and off the :math:`\tau_F = \tau_G` singularity.
- :func:`fatigue_timescale` — :math:`\tau_G = \tau_v + \lambda_G` keeps the Busso accumulation
  timescale strictly fastest, giving the ordered :math:`\tau_v < \tau_G < \tau_F`.
- :func:`fatigue_gain` — :math:`k_G = (\theta + 1) k_F` floors the gain ratio above one, hard
  enforcing :math:`k_G > k_F` (Peng 2024).

These are pure arithmetic transforms of already-sampled parameters: the non-negative margins
(:math:`\lambda_F, \lambda_G, \theta`) carry ``HalfNormal`` priors in the model layer, which is
what makes the constraints hold with certainty. Each works on plain floats (for tests and
diagnostics) or on PyTensor variables (inside the model graph).
"""

from __future__ import annotations

from typing import Any


def fitness_timescale(tau_G: Any, lambda_f: Any) -> Any:
    r"""Reparameterize the fitness timescale as :math:`\tau_F = 1 + \tau_G + \lambda_F`.

    Parameters
    ----------
    tau_G : Any
        Fatigue decay time constant :math:`\tau_G` in days (float or scalar tensor).
    lambda_f : Any
        Non-negative fitness-slower-than-fatigue margin :math:`\lambda_F` in days.

    Returns
    -------
    Any
        The fitness decay time constant :math:`\tau_F`, of the same kind as the inputs.
    """
    return 1.0 + tau_G + lambda_f


def fatigue_timescale(tau_v: Any, lambda_g: Any) -> Any:
    r"""Reparameterize the fatigue timescale as :math:`\tau_G = \tau_v + \lambda_G` (Busso).

    Parameters
    ----------
    tau_v : Any
        Busso accumulation time constant :math:`\tau_v` in days (float or scalar tensor).
    lambda_g : Any
        Non-negative fatigue-slower-than-accumulation margin :math:`\lambda_G` in days.

    Returns
    -------
    Any
        The fatigue decay time constant :math:`\tau_G`, of the same kind as the inputs.
    """
    return tau_v + lambda_g


def fatigue_gain(k_F: Any, theta: Any) -> Any:
    r"""Reparameterize the fatigue gain as :math:`k_G = (\theta + 1) k_F`.

    Parameters
    ----------
    k_F : Any
        Fitness gain :math:`k_F` (float or scalar tensor).
    theta : Any
        Non-negative gain-ratio driver :math:`\theta`; :math:`k_G / k_F = \theta + 1`.

    Returns
    -------
    Any
        The fatigue gain :math:`k_G`, of the same kind as the inputs.
    """
    return (theta + 1.0) * k_F
