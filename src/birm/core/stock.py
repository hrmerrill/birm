r"""Latent stock assembly: $R_t = f_t - g_t$.

Assembles the latent stock for a chosen impulse-response ``core`` via [impulse_response][],
wiring together the fitness and fatigue building blocks (which share the leaky integrators in
[birm.core.integrator][]). The reparameterization that keeps the parameter space identifiable
($\tau_F = 1 + \tau_G + \lambda_F$, $k_G = (\theta + 1) k_F$) lives in [birm.core.reparam][];
this layer consumes the already-resolved $(k_F, \tau_F, k_G, \tau_G, \dots)$.
"""

from __future__ import annotations

from typing import Any

import pytensor.tensor as pt

from birm.core.fatigue import fatigue_stock
from birm.core.fitness import fitness_stock

CORES = ("banister", "busso", "ceddia", "fitness-only")


def impulse_response(
    drive: Any,
    *,
    k_F: Any,
    tau_F: Any,
    k_G: Any = 0.0,
    tau_G: Any = 1.0,
    core: str = "banister",
    tau_v: Any = None,
    k_D: Any = 0.0,
    tau_F1: Any = 0.0,
    tau_G1: Any = 0.0,
) -> pt.TensorVariable:
    r"""Assemble the latent impulse-response stock $R_t = f_t - g_t$.

    The cores are

    - ``banister`` — linear fitness and fatigue with fixed time constants (Banister 1975).
    - ``busso`` — Banister fitness; fatigue gain grows with recent load over ``tau_v``
      (Busso 2003).
    - ``ceddia`` — power-law diminishing fitness gain (``k_D``) and fitness/fatigue decay time
      constants that grow linearly with the session load (``tau_F1``, ``tau_G1``), the form
      recommended by Ceddia (2026, Fig. 10).
    - ``fitness-only`` — Banister fitness with no fatigue ($g_t \equiv 0$).

    Parameters
    ----------
    drive : Any
        Training-load drive $w_t$ on the daily grid, length ``T``. Must be a concrete array for
        ``ceddia``.
    k_F : Any
        Fitness gain $k_F$ (float or scalar tensor).
    tau_F : Any
        Fitness decay time constant $\tau_F$ in days.
    k_G : Any
        Fatigue gain $k_G$ (float or scalar tensor); ignored for ``fitness-only``.
    tau_G : Any
        Fatigue decay time constant $\tau_G$ in days; ignored for ``fitness-only``.
    core : str
        One of ``banister``, ``busso``, ``ceddia``, ``fitness-only``.
    tau_v : Any, optional
        Busso third (fastest) timescale $\tau_v$ in days; required for ``busso``.
    k_D : Any
        Ceddia diminishing-returns exponent $k_D \ge 0$.
    tau_F1 : Any
        Ceddia fitness decay-time slope $\tau_{F1} \ge 0$ (days per unit load).
    tau_G1 : Any
        Ceddia fatigue decay-time slope $\tau_{G1} \ge 0$ (days per unit load).

    Returns
    -------
    pytensor.tensor.TensorVariable
        The latent impulse-response stock $R_t$, length ``T``.

    Raises
    ------
    ValueError
        If ``core`` is unrecognized, or ``busso`` is requested without ``tau_v``.
    """
    if core not in CORES:
        raise ValueError(f"core must be one of {CORES}, got {core!r}")
    if core == "ceddia":
        f = fitness_stock(drive, k_F, tau_F, variant="ceddia", k_D=k_D, tau_F1=tau_F1)
        return f - fatigue_stock(drive, k_G, tau_G, variant="ceddia", tau_G1=tau_G1)
    f = fitness_stock(drive, k_F, tau_F)
    if core == "fitness-only":
        return f
    return f - fatigue_stock(drive, k_G, tau_G, variant=core, tau_v=tau_v)
