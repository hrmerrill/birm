r"""Per-channel observation layer: the measurement map from the shared stock to one metric.

Each response metric :math:`Y^{(j)}` is a noisy, transformed readout of the single shared latent
stock :math:`R_t` (``docs/model.md``):

.. math::

    Y^{(j)}_t = \alpha^{(j)} + \gamma^{(j)} l^{(j)}(R_t) + X^{(j)}_t \beta^{(j)} + u^{(j)}_t
                + \epsilon^{(j)}_t .

This module builds the deterministic part of that map -- everything except the white noise
:math:`\epsilon` -- as a pure PyTensor graph, mirroring :mod:`birm.core.stock`: it consumes
already-resolved parameters (``alpha``, ``gamma``, ``beta``, and a prebuilt AR(1) latent ``u``) and
never creates priors or the ``obs_<metric>`` likelihood. The surrounding ``build_model`` owns the
``pm.*`` priors (:math:`\alpha, \gamma, \beta, \sigma`, and the AR(1) ``phi_u``/``sigma_u``) and the
Normal likelihood, exactly as it owns the stock's gain/decay priors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytensor.tensor as pt

LINKS: dict[str, tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {
    "identity": (lambda x: x, lambda x: x),
    "log": (np.log, np.exp),
    "logit": (lambda x: np.log(x / (1.0 - x)), lambda x: 1.0 / (1.0 + np.exp(-x))),
}


def apply_link(x: Any, name: str = "identity") -> Any:
    r"""Apply the forward link :math:`l^{(j)}` to the latent stock.

    Parameters
    ----------
    x : Any
        Value to transform, typically the latent stock :math:`R_t` (float, array, or tensor).
    name : str
        Link name; one of the keys of :data:`LINKS` (``identity``, ``log``, ``logit``).

    Returns
    -------
    Any
        The transformed value, numpy for numpy inputs and a PyTensor variable for symbolic inputs.

    Raises
    ------
    ValueError
        If ``name`` is not a recognized link.
    """
    if name not in LINKS:
        raise ValueError(f"link must be one of {tuple(LINKS)}, got {name!r}")
    return LINKS[name][0](x)


def apply_inverse_link(x: Any, name: str = "identity") -> Any:
    r"""Apply the inverse link :math:`{l^{(j)}}^{-1}` (the forecast back-transform).

    Parameters
    ----------
    x : Any
        Value to back-transform (float, array, or tensor).
    name : str
        Link name; one of the keys of :data:`LINKS` (``identity``, ``log``, ``logit``).

    Returns
    -------
    Any
        The back-transformed value, numpy for numpy inputs and a PyTensor variable for symbolic
        inputs.

    Raises
    ------
    ValueError
        If ``name`` is not a recognized link.
    """
    if name not in LINKS:
        raise ValueError(f"link must be one of {tuple(LINKS)}, got {name!r}")
    return LINKS[name][1](x)


def orthogonalize(signal: Any, design: Any, ridge: float = 1e-6) -> pt.TensorVariable:
    r"""Project ``signal`` into the null space of the ``design`` columns.

    Returns :math:`u - X (X^\top X + \text{ridge}\,I)^{-1} X^\top u`, the component of ``signal``
    orthogonal to every column of ``design`` (``X^T u = 0`` up to the ridge). Used to sweep the
    AR(1) residual off the load-driven subspace :math:`\text{span}\{1, f, g, \dots\}` so it cannot
    re-absorb fitness/fatigue variance the stock parameters should explain. The small ``ridge``
    keeps the near-collinear leaky-integrator normal equations well-posed.

    Parameters
    ----------
    signal : Any
        The length-``T`` residual to orthogonalize (array-like or tensor).
    design : Any
        The ``(T, k)`` design matrix whose column span is swept out (array-like or tensor).
    ridge : float
        Tikhonov ridge added to the Gram diagonal for numerical stability.

    Returns
    -------
    pytensor.tensor.TensorVariable
        The load-orthogonal residual, length ``T``.
    """
    signal = pt.as_tensor_variable(signal)
    design = pt.as_tensor_variable(design)
    gram = design.T @ design + ridge * pt.eye(design.shape[1])
    coef = pt.linalg.solve(gram, design.T @ signal, assume_a="pos")
    return signal - design @ coef


def observation_mean(
    stock: Any,
    *,
    link: str = "identity",
    alpha: Any = 0.0,
    gamma: Any = 1.0,
    exog: Any = None,
    beta: Any = None,
    u: Any = None,
) -> Any:
    r"""Assemble the deterministic channel mean :math:`\alpha + \gamma l(R) + X\beta + u`.

    Composes the measurement map for one metric on its own observed rows (every input is row-aligned
    to the metric's observations). Every term but the linked stock is optional: the intercept
    defaults to ``0``, the scale to ``1``, and the exogenous and AR(1) contributions are dropped
    when their inputs are ``None``. The white noise :math:`\epsilon` is not added here -- it is the
    likelihood scale owned by ``build_model``.

    Parameters
    ----------
    stock : Any
        The latent stock :math:`R_t` evaluated on the metric's observed rows (array-like or tensor).
    link : str
        Link :math:`l^{(j)}` applied to ``stock``; one of the keys of :data:`LINKS`.
    alpha : Any
        Intercept :math:`\alpha^{(j)}` (scalar). ``0`` drops the intercept.
    gamma : Any
        Scale :math:`\gamma^{(j)}` on the linked stock (scalar). Pinned to ``1`` on the densest
        metric by the model layer.
    exog : Any, optional
        The ``(n_j, p)`` exogenous design :math:`X^{(j)}` on the metric's rows; ``None`` drops it.
    beta : Any, optional
        The length-``p`` coefficients :math:`\beta^{(j)}`; required when ``exog`` is given.
    u : Any, optional
        The orthogonalized AR(1) residual :math:`u^{(j)}` on the metric's rows; ``None`` drops it.

    Returns
    -------
    Any
        The row-aligned channel mean (everything but the noise :math:`\epsilon`); numpy for numpy
        inputs and a PyTensor variable when any input is symbolic.

    Raises
    ------
    ValueError
        If ``exog`` is given without ``beta`` (or vice versa).
    """
    if (exog is None) != (beta is None):
        raise ValueError("exog and beta must be given together (both or neither)")

    mean = alpha + gamma * apply_link(stock, link)
    if exog is not None:
        mean = mean + pt.as_tensor_variable(exog) @ pt.as_tensor_variable(beta)
    if u is not None:
        mean = mean + u
    return mean
