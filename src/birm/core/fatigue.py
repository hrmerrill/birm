r"""Fatigue stock $g_t$: ``banister`` | ``busso`` | ``none``.

Each variant turns the training-load drive $w_t$ into the fatigue component of the latent
impulse-response stock on the shared daily grid. The fatigue impulse stays linear in the load
across all variants; the only variation is whether the gain is constant (``banister``),
time-varying through a third accumulated-load timescale (``busso``), or absent (``none``, a
fitness-only model).
"""

from __future__ import annotations

from typing import Any

import pytensor.tensor as pt

from birm.core.integrator import _ar1

FATIGUE = ("banister", "busso", "none")


def fatigue_stock(
    drive: Any,
    k_G: Any,
    tau_G: Any,
    *,
    variant: str = "banister",
    tau_v: Any = None,
) -> pt.TensorVariable:
    r"""Build the fatigue stock $g_t$ for the chosen variant.

    The three variants are

    - ``banister`` — constant gain: $g_t = k_G \sum_{s<t} w_s e^{-(t-s)/\tau_G}$.
    - ``busso`` — variable dose-response: the gain is itself a leaky integral of recent load,
      $k_G(s) = k_G \sum_{u<s} w_u e^{-(s-u)/\tau_v}$, and
      $g_t = \sum_{s<t} k_G(s)\, w_s e^{-(t-s)/\tau_G}$. ``tau_v`` is the third (fastest)
      timescale and must be supplied; the ordering $\tau_v < \tau_G$ is the caller's
      (model layer) responsibility.
    - ``none`` — fitness-only model: $g_t \equiv 0$.

    Parameters
    ----------
    drive : Any
        Training-load drive $w_t$ on the daily grid, length ``T`` (array-like or tensor).
    k_G : Any
        Fatigue gain $k_G$, a float or scalar tensor.
    tau_G : Any
        Fatigue decay time constant $\tau_G$ in days, a float or scalar tensor.
    variant : str
        One of ``banister``, ``busso``, ``none``.
    tau_v : Any, optional
        Busso third timescale $\tau_v$ in days; required when ``variant="busso"``.

    Returns
    -------
    pytensor.tensor.TensorVariable
        The fatigue stock $g_t$, length ``T``.

    Raises
    ------
    ValueError
        If ``variant`` is unrecognized, or ``variant="busso"`` without ``tau_v``.
    """
    if variant not in FATIGUE:
        raise ValueError(f"fatigue variant must be one of {FATIGUE}, got {variant!r}")

    drive = pt.as_tensor_variable(drive)

    if variant == "none":
        return pt.zeros_like(drive)

    if variant == "busso":
        if tau_v is None:
            raise ValueError("busso fatigue requires tau_v (the third, fastest timescale)")
        # k_G(s)/k_G accumulated over tau_v, then re-forced into the tau_G integral as k_G(s)*w_s.
        # Both integrals are strictly causal (sum_{s<t}): the stock responds only to prior days.
        gain = _ar1(pt.exp(-1.0 / tau_v), drive, exclude_current=True)
        return k_G * _ar1(pt.exp(-1.0 / tau_G), gain * drive, exclude_current=True)

    return k_G * _ar1(pt.exp(-1.0 / tau_G), drive, exclude_current=True)
