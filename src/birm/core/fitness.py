r"""Fitness stock $f_t$: ``banister`` | ``ceddia_periodize`` | ``ceddia_taper`` | ``ceddia``.

Each variant turns the training-load drive $w_t$ into the fitness component of the latent
impulse-response stock on the shared daily grid. All the nonlinearity lives in the fitness-gain
function $M_F(w_s, g_s)$ (Ceddia 2026): the plain Banister null is $M_F = k_F w_s$,
and the Ceddia variants divide it by a **periodization** factor $1 + k_p g_s$ (gain fades as
fatigue accumulates), a **taper** factor $1 + k_a w_s$ (diminishing returns within a single
session), or both. Setting ``k_p = k_a = 0`` recovers the Banister linear model exactly.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytensor.tensor as pt

from birm.core.integrator import _ar1

FITNESS = ("banister", "ceddia_periodize", "ceddia_taper", "ceddia")
_NEEDS_FATIGUE = ("ceddia_periodize", "ceddia")


def fitness_stock(
    drive: Any,
    k_F: Any,
    tau_F: Any,
    *,
    variant: str = "banister",
    g: Any = None,
    k_p: Any = 0.0,
    k_a: Any = 0.0,
) -> pt.TensorVariable:
    r"""Build the fitness stock $f_t$ for the chosen variant.

    The gain $M_F(w_s, g_s)$ per variant is

    - ``banister`` — $k_F w_s$.
    - ``ceddia_periodize`` — $k_F w_s / (1 + k_p g_s)$ (periodization via fatigue feedback).
    - ``ceddia_taper`` — $k_F w_s / (1 + k_a w_s)$ (tapering via diminishing returns).
    - ``ceddia`` — $k_F w_s / \big((1 + k_p g_s)(1 + k_a w_s)\big)$ (both).

    and $f_t = \sum_{s<t} M_F(w_s, g_s) e^{-(t-s)/\tau_F}$. The periodization factor uses the
    fatigue stock $g_s$ entering day $s$, which is strictly causal
    ($g_s = k_G \sum_{u<s} w_u e^{-(s-u)/\tau_G}$), so yesterday's load gates today's gain but
    today's own load cannot instantly gate itself. When a periodization variant is paired with
    ``g=None`` (i.e. a ``none`` fatigue model) there is no fatigue to feed back, so the
    periodization factor is dropped and the variant degenerates to the taper/Banister form; a
    [UserWarning][] is emitted because the requested spec is effectively overridden.

    Parameters
    ----------
    drive : Any
        Training-load drive $w_t$ on the daily grid, length ``T`` (array-like or tensor).
    k_F : Any
        Fitness gain $k_F$, a float or scalar tensor.
    tau_F : Any
        Fitness decay time constant $\tau_F$ in days, a float or scalar tensor.
    variant : str
        One of ``banister``, ``ceddia_periodize``, ``ceddia_taper``, ``ceddia``.
    g : Any, optional
        Fatigue stock $g_t$, length ``T``; required for ``ceddia_periodize`` and ``ceddia``.
        Passing ``None`` (a ``none`` fatigue model) drops the periodization factor and warns.
    k_p : Any
        Periodization strength $k_p$ (``>= 0``); ``0`` recovers no periodization.
    k_a : Any
        Taper strength $k_a$ (``>= 0``); ``0`` recovers no tapering.

    Returns
    -------
    pytensor.tensor.TensorVariable
        The fitness stock $f_t$, length ``T``.

    Raises
    ------
    ValueError
        If ``variant`` is unrecognized.

    Warns
    -----
    UserWarning
        If ``variant`` needs fatigue feedback but ``g`` is ``None``, the periodization factor is
        silently dropped and the spec is effectively overridden.
    """
    if variant not in FITNESS:
        raise ValueError(f"fitness variant must be one of {FITNESS}, got {variant!r}")

    drive = pt.as_tensor_variable(drive)
    forcing = k_F * drive

    if variant in _NEEDS_FATIGUE:
        if g is None:
            warnings.warn(
                f"fitness={variant!r} needs a fatigue stock for its periodization feedback, but "
                "the fatigue model is 'none' (g=None). Dropping the periodization factor: this "
                f"fitness core reduces to "
                f"{'ceddia_taper' if variant == 'ceddia' else 'banister'}. Pair a non-'none' "
                "fatigue model to keep periodization.",
                UserWarning,
                stacklevel=2,
            )
        else:
            # g_s already excludes day s (fatigue_stock is strictly causal), so no extra shift.
            forcing = forcing / (1.0 + k_p * pt.as_tensor_variable(g))

    if variant in ("ceddia_taper", "ceddia"):
        forcing = forcing / (1.0 + k_a * drive)

    return _ar1(pt.exp(-1.0 / tau_F), forcing, exclude_current=True)
