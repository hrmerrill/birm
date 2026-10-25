r"""Fitness stock $f_t$: ``banister`` | ``ceddia``.

Each variant turns the training-load drive $w_t$ into the fitness component of the latent
impulse-response stock on the shared daily grid. ``banister`` is linear in the load with a fixed
decay time constant. ``ceddia`` is the recommended form of Ceddia (2026, Fig. 10): a power-law
diminishing-returns gain (Eq. 20) and a decay time constant that grows linearly with the load
of the session that produced it (Eq. 30 with $m = 1$).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytensor.tensor as pt

from birm.core.integrator import _ar1, _leaky_varying

FITNESS = ("banister", "ceddia")
_K_D_EPS = 1e-6


def diminishing_gain(drive: Any, k_F: Any, k_D: Any) -> Any:
    r"""Ceddia (2026, Eq. 20) power-law diminishing-returns fitness gain $M_F(w)$.

    $$M_F(w) = k_F \frac{(1 + w)^{1 - k_D} - 1}{1 - k_D},$$

    which has slope $k_F$ at $w = 0$, reduces to the Banister $k_F w$ at $k_D = 0$, and tends
    to $k_F \log(1 + w)$ as $k_D \to 1$ (evaluated in that limit within ``1e-6`` of one, so
    the gradient stays finite). Ceddia writes this as $k_1/(k_3 - 1)\,(1 - (1+w)^{-(k_3-1)})$;
    here $k_1 = k_F$ and $k_3 = k_D$.

    Parameters
    ----------
    drive : Any
        Training load $w \ge 0$ (array-like or tensor).
    k_F : Any
        Fitness gain $k_F$ (marginal gain at zero load), a float or scalar tensor.
    k_D : Any
        Diminishing-returns exponent $k_D \ge 0$, a float or scalar tensor.

    Returns
    -------
    Any
        The per-day fitness impulse $M_F(w)$, same shape as ``drive``.
    """
    log1p_w = pt.log1p(pt.as_tensor_variable(drive))
    a = 1.0 - pt.as_tensor_variable(k_D)
    near_one = pt.abs(a) < _K_D_EPS
    a_safe = pt.switch(near_one, _K_D_EPS, a)
    return k_F * pt.switch(near_one, log1p_w, pt.expm1(a_safe * log1p_w) / a_safe)


def fitness_stock(
    drive: Any,
    k_F: Any,
    tau_F: Any,
    *,
    variant: str = "banister",
    k_D: Any = 0.0,
    tau_F1: Any = 0.0,
) -> pt.TensorVariable:
    r"""Build the fitness stock $f_t$ for the chosen variant.

    - ``banister`` — $f_t = k_F \sum_{s<t} w_s e^{-(t-s)/\tau_F}$.
    - ``ceddia`` — $f_t = \sum_{s<t} M_F(w_s) e^{-(t-s)/\tau_F(w_s)}$ with the power-law gain
      $M_F$ of [diminishing_gain][birm.core.fitness.diminishing_gain] and the load-dependent
      time constant $\tau_F(w) = \tau_F + \tau_{F1} w$ (Ceddia's $\tau_{11}$, $\tau_{12}$).
      Setting $k_D = \tau_{F1} = 0$ recovers ``banister`` exactly.

    Parameters
    ----------
    drive : Any
        Training-load drive $w_t$ on the daily grid, length ``T``. Must be a concrete array for
        ``ceddia`` (the training days index the decay kernel).
    k_F : Any
        Fitness gain $k_F$, a float or scalar tensor.
    tau_F : Any
        Fitness decay time constant $\tau_F$ in days (at zero load for ``ceddia``).
    variant : str
        One of ``banister``, ``ceddia``.
    k_D : Any
        Diminishing-returns exponent $k_D \ge 0$ (``ceddia`` only).
    tau_F1 : Any
        Days of extra fitness decay time per unit load, $\tau_{F1} \ge 0$ (``ceddia`` only).

    Returns
    -------
    pytensor.tensor.TensorVariable
        The fitness stock $f_t$, length ``T``.

    Raises
    ------
    ValueError
        If ``variant`` is unrecognized.
    """
    if variant not in FITNESS:
        raise ValueError(f"fitness variant must be one of {FITNESS}, got {variant!r}")
    if variant == "ceddia":
        w = np.asarray(drive, dtype=float)
        return _leaky_varying(diminishing_gain(w, k_F, k_D), tau_F + tau_F1 * w, w)
    return k_F * _ar1(pt.exp(-1.0 / tau_F), drive, exclude_current=True)
