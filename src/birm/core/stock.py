r"""Latent stock assembly: :math:`R_t = f_t - g_t`.

Placeholder for the stock layer. For now it owns only the shared leaky-integrator recursion
:func:`_ar1` used by both the fitness and fatigue cores; the reparameterization wiring
(:math:`\tau_F = 1 + \tau_G + \lambda_f`, :math:`k_G = r k_F`) and the ``f`` :math:`\times` ``g``
assembly of :math:`R_t` land here later.

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

import pytensor
import pytensor.tensor as pt


def _ar1(coef: Any, forcing: Any, x_init: float = 0.0) -> pt.TensorVariable:
    r"""Realise the leaky recursion :math:`x_t = \text{coef}\,x_{t-1} + \text{forcing}_t`.

    Uses an ``O(T)`` ``pytensor.scan`` rather than a dense ``O(T^2)`` geometric operator so the
    gradient stays cheap on long (multi-year) daily grids. ``coef`` is a scalar (a float or a
    random-variable tensor); ``forcing`` is a length-``T`` sequence.

    Parameters
    ----------
    coef : Any
        Scalar decay coefficient :math:`e^{-1/\tau}`, a float or scalar tensor.
    forcing : Any
        Per-day forcing sequence, length ``T`` (array-like or tensor).
    x_init : float
        Initial state :math:`x_{-1}` before the first step.

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
    out, _ = pytensor.scan(
        step,
        sequences=[forcing],
        outputs_info=[pt.constant(float(x_init)).astype(forcing.dtype)],
        non_sequences=[coef],
    )
    return out
