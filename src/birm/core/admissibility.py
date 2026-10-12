r"""Ceddia (2026) admissibility quantity for the latent stock -- recorded, never enforced.

Ceddia 2026 derive an admissibility condition implied by bounded infinite-horizon optimal training
policies (finite human performance): a unit load impulse must leave a net *gain* rather than a net
cost. In the notation of ``docs/model.md`` this is

.. math::

    k_G \left(1 + \frac{1}{e^{1/\tau_G} - 1}\right) - \frac{k_F}{e^{1/\tau_F} - 1} > 0 ,

equivalently a floor on the gain ratio,

.. math::

    \frac{k_G}{k_F} > \frac{1 - e^{-1/\tau_G}}{e^{1/\tau_F} - 1} \equiv a_c .

BIRM does **not** impose :math:`k_G/k_F > a_c`: that stricter floor can bias estimates for athletes
who have not yet reached peak performance. The reparameterization enforces the weaker
:math:`k_G/k_F > 1` (and, via :math:`\tau_F > \tau_G + 1`, the approximate :math:`a_c > 1`), while
the admissibility *margin* here is recorded as a deterministic so ``P(margin > 0)`` can be read off
the posterior as a diagnostic -- exactly as in the R&D ``banister.py`` head.

Both helpers use ``numpy.exp``, which dispatches on PyTensor variables (returning a tensor for the
in-graph deterministic) and on plain floats/arrays (returning a float/array for diagnostics on
posterior draws), so the same function serves both callers. The time constants must satisfy
:math:`\tau_F > \tau_G` as the reparameterization guarantees.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def admissibility(k_F: Any, k_G: Any, tau_F: Any, tau_G: Any) -> Any:
    r"""Ceddia (2026) admissibility margin; positive means admissible.

    Parameters
    ----------
    k_F : Any
        Fitness gain :math:`k_F` (float, array, or scalar tensor).
    k_G : Any
        Fatigue gain :math:`k_G` (float, array, or scalar tensor).
    tau_F : Any
        Fitness decay time constant :math:`\tau_F` in days.
    tau_G : Any
        Fatigue decay time constant :math:`\tau_G` in days.

    Returns
    -------
    Any
        The admissibility margin (``> 0`` is admissible), numpy for numpy inputs and a PyTensor
        variable for symbolic inputs.
    """
    return k_G * (1.0 + 1.0 / (np.exp(1.0 / tau_G) - 1.0)) - k_F / (np.exp(1.0 / tau_F) - 1.0)


def admissibility_threshold(tau_F: Any, tau_G: Any) -> Any:
    r"""Gain-ratio floor :math:`a_c` above which the stock is admissible.

    Parameters
    ----------
    tau_F : Any
        Fitness decay time constant :math:`\tau_F` in days (float, array, or scalar tensor).
    tau_G : Any
        Fatigue decay time constant :math:`\tau_G` in days (float, array, or scalar tensor).

    Returns
    -------
    Any
        The gain-ratio floor :math:`a_c`, numpy for numpy inputs and a PyTensor variable for
        symbolic inputs.
    """
    return (1.0 - np.exp(-1.0 / tau_G)) / (np.exp(1.0 / tau_F) - 1.0)
