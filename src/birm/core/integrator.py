r"""Shared leaky integrators: [_ar1][] (fixed decay) and [_leaky_varying][] (per-impulse decay)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytensor
import pytensor.tensor as pt


def _ar1(
    coef: Any, forcing: Any, x_init: float = 0.0, *, exclude_current: bool = False
) -> pt.TensorVariable:
    r"""Realise the leaky recursion $x_t = \text{coef}\,x_{t-1} + \text{forcing}_t$.

    Uses an ``O(T)`` ``pytensor.scan`` rather than a dense ``O(T^2)`` geometric operator so the
    gradient stays cheap on long (multi-year) daily grids. ``coef`` is a scalar (a float or a
    random-variable tensor); ``forcing`` is a length-``T`` sequence.

    Parameters
    ----------
    coef : Any
        Scalar decay coefficient $e^{-1/\tau}$, a float or scalar tensor.
    forcing : Any
        Per-day forcing sequence, length ``T`` (array-like or tensor).
    x_init : float
        Initial state $x_{-1}$ before the first step.
    exclude_current : bool
        When ``True`` the same-day forcing is removed from each step, giving the strictly causal
        $x_t = \sum_{s<t} \text{forcing}_s\, \text{coef}^{\,t-s}$ (the response at day $t$ depends
        only on days *before* $t$). The default ``False`` keeps the inclusive $\sum_{s\le t}$ form
        required by a genuine AR(1) latent process, whose current innovation must enter.

    Returns
    -------
    pytensor.tensor.TensorVariable
        The accumulated stock, length ``T``.
    """

    def step(f_t: Any, x_prev: Any, c: Any) -> Any:
        """Advance one leaky-integrator step.

        Parameters
        ----------
        f_t : Any
            Forcing at the current day.
        x_prev : Any
            Stock carried from the previous day.
        c : Any
            Decay coefficient.

        Returns
        -------
        Any
            Updated stock for the current day.
        """
        return c * x_prev + f_t

    forcing = pt.as_tensor_variable(forcing)
    stock = pytensor.scan(
        step,
        sequences=[forcing],
        outputs_info=[pt.constant(float(x_init)).astype(forcing.dtype)],
        non_sequences=[coef],
        return_updates=False,
    )
    # Dropping the same-day term x_t - forcing_t turns the inclusive sum_{s<=t} into sum_{s<t}
    # while preserving the e^{-(t-s)/tau} kernel (nearest past day t-1 keeps weight e^{-1/tau}).
    return stock - forcing if exclude_current else stock


def _leaky_varying(forcing: Any, tau: Any, drive: np.ndarray) -> pt.TensorVariable:
    r"""Strictly causal sum with a per-impulse decay $x_t = \sum_{s<t} F_s e^{-(t-s)/\tau_s}$.

    Each impulse $F_s$ decays with its own time constant $\tau_s$ (Ceddia 2026, Eq. 17), so no
    single recursion applies. The kernel is built densely over the training days only (days with
    $w_s > 0$; the forcing must vanish on rest days), costing ``O(T * S)`` for ``S`` training days.

    Parameters
    ----------
    forcing : Any
        Per-day impulse $F_s$, length ``T`` (array-like or tensor); zero wherever ``drive`` is.
    tau : Any
        Per-day decay time constant $\tau_s$ in days, length ``T`` (array-like or tensor).
    drive : np.ndarray
        Concrete training load $w_t$, length ``T``, whose positive days index the kernel columns.

    Returns
    -------
    pytensor.tensor.TensorVariable
        The accumulated stock, length ``T``.
    """
    days = np.flatnonzero(drive > 0)
    lag = np.arange(len(drive))[:, None] - days[None, :]
    causal = lag > 0
    lag = np.where(causal, lag, 0).astype(float)
    tau_s = pt.as_tensor_variable(tau)[days]
    kernel = pt.exp(-lag / tau_s[None, :]) * causal
    return pt.dot(kernel, pt.as_tensor_variable(forcing)[days])
