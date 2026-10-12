r"""Latent stock assembly: :math:`R_t = f_t - g_t`.

Assembles the latent stock for a chosen ``fitness`` :math:`\times` ``fatigue`` cell via
:func:`impulse_response`, wiring together the fitness and fatigue cores (which share the
:func:`birm.core.integrator._ar1` leaky recursion). The reparameterization that keeps the
parameter space identifiable (:math:`\tau_F = 1 + \tau_G + \lambda_F`, :math:`k_G = (\theta + 1)
k_F`) lives in :mod:`birm.core.reparam`; this layer consumes the already-resolved
:math:`(k_F, \tau_F, k_G, \tau_G, \dots)`.

Note: when assembling the ``f`` x ``g`` grid, ``fitness="ceddia"`` + ``fatigue="busso"`` is a
legal cell but likely the most fragile one. Busso makes *fatigue* gain grow with accumulated load
while Ceddia's ``k_p`` makes the *fitness* gain shrink with accumulated fatigue -- two suppression
channels driven by the same load history, so ``k_p`` and the Busso ``tau_v``/gain trade off and
sample poorly (posterior ridges, divergences), and three time constants (``tau_v < tau_G < tau_F``)
plus the Ceddia nonlinearity must be separated. It is physiologically coherent but redundant, not
inconsistent. Do not forbid it: lean on the identifiability reparam and let the per-channel LOO
/ convergence / Pareto-k gates demote it when the data will not support both.
"""

from __future__ import annotations

from typing import Any

import pytensor.tensor as pt

from birm.core.fatigue import fatigue_stock
from birm.core.fitness import fitness_stock


def impulse_response(
    drive: Any,
    *,
    k_F: Any,
    tau_F: Any,
    k_G: Any,
    tau_G: Any,
    fitness: str = "banister",
    fatigue: str = "banister",
    k_p: Any = 0.0,
    k_a: Any = 0.0,
    tau_v: Any = None,
) -> pt.TensorVariable:
    r"""Assemble the latent impulse-response stock :math:`R_t = f_t - g_t`.

    Builds the fatigue stock :math:`g_t` for the chosen ``fatigue`` variant and the fitness stock
    :math:`f_t` for the chosen ``fitness`` variant, then returns their difference on the shared
    daily grid. The periodization variants (``ceddia_periodize``, ``ceddia``) feed the fatigue
    stock back into the fitness gain; when ``fatigue="none"`` there is no fatigue to feed back, so
    the periodization factor is dropped and :func:`birm.core.fitness.fitness_stock` warns. The
    reparameterization that produces identifiable :math:`(\tau_F, k_G, \dots)` lives in
    :mod:`birm.core.reparam`; this function consumes the already-resolved values.

    Parameters
    ----------
    drive : Any
        Training-load drive :math:`w_t` on the daily grid, length ``T`` (array-like or tensor).
    k_F : Any
        Fitness gain :math:`k_F` (float or scalar tensor).
    tau_F : Any
        Fitness decay time constant :math:`\tau_F` in days.
    k_G : Any
        Fatigue gain :math:`k_G` (float or scalar tensor); ignored when ``fatigue="none"``.
    tau_G : Any
        Fatigue decay time constant :math:`\tau_G` in days.
    fitness : str
        Fitness variant; one of ``banister``, ``ceddia_periodize``, ``ceddia_taper``, ``ceddia``.
    fatigue : str
        Fatigue variant; one of ``banister``, ``busso``, ``none``.
    k_p : Any
        Periodization strength :math:`k_p` (``>= 0``) for the Ceddia fitness variants.
    k_a : Any
        Taper strength :math:`k_a` (``>= 0``) for the Ceddia fitness variants.
    tau_v : Any, optional
        Busso third (fastest) timescale :math:`\tau_v` in days; required when ``fatigue="busso"``.

    Returns
    -------
    pytensor.tensor.TensorVariable
        The latent impulse-response stock :math:`R_t`, length ``T``.
    """
    g = fatigue_stock(drive, k_G, tau_G, variant=fatigue, tau_v=tau_v)
    g_feedback = None if fatigue == "none" else g
    f = fitness_stock(drive, k_F, tau_F, variant=fitness, g=g_feedback, k_p=k_p, k_a=k_a)
    return f - g
