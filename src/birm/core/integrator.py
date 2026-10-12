r"""Shared leaky-integrator [_ar1][] reused by fitness, fatigue, and the Busso gain."""

from __future__ import annotations

from typing import Any

import pytensor
import pytensor.tensor as pt


def _ar1(coef: Any, forcing: Any, x_init: float = 0.0) -> pt.TensorVariable:
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
    return pytensor.scan(
        step,
        sequences=[forcing],
        outputs_info=[pt.constant(float(x_init)).astype(forcing.dtype)],
        non_sequences=[coef],
        return_updates=False,
    )
