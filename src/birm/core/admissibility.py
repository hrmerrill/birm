r"""Ceddia (2026) admissibility quantity for the latent stock.

Ceddia 2026 derive an admissibility condition implied by bounded infinite-horizon optimal training
policies: a unit load impulse must leave a net *gain* rather than a net cost:

$$
k_G \left(1 + \frac{1}{e^{1/\tau_G} - 1}\right) - \frac{k_F}{e^{1/\tau_F} - 1} > 0 ,
$$

equivalently on the gain ratio,

$$
\frac{k_G}{k_F} > \frac{1 - e^{-1/\tau_G}}{e^{1/\tau_F} - 1} \equiv a_c .
$$

The reparameterization enforces the weaker $k_G/k_F > 1$ (and, via
$\tau_F > \tau_G + 1$, the approximate $a_c > 1$), while the admissibility margin here
is recorded so ``P(margin > 0)`` can be read off the posterior as a diagnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def admissibility(k_F: Any, k_G: Any, tau_F: Any, tau_G: Any) -> Any:
    r"""Ceddia (2026) admissibility margin; positive means admissible.

    Parameters
    ----------
    k_F : Any
        Fitness gain $k_F$ (float, array, or scalar tensor).
    k_G : Any
        Fatigue gain $k_G$ (float, array, or scalar tensor).
    tau_F : Any
        Fitness decay time constant $\tau_F$ in days.
    tau_G : Any
        Fatigue decay time constant $\tau_G$ in days.

    Returns
    -------
    Any
        The admissibility margin (``> 0`` is admissible), numpy for numpy inputs and a PyTensor
        variable for symbolic inputs.
    """
    return k_G * (1.0 + 1.0 / (np.exp(1.0 / tau_G) - 1.0)) - k_F / (np.exp(1.0 / tau_F) - 1.0)


def admissibility_threshold(tau_F: Any, tau_G: Any) -> Any:
    r"""Gain-ratio floor $a_c$ above which the stock is admissible.

    Parameters
    ----------
    tau_F : Any
        Fitness decay time constant $\tau_F$ in days (float, array, or scalar tensor).
    tau_G : Any
        Fatigue decay time constant $\tau_G$ in days (float, array, or scalar tensor).

    Returns
    -------
    Any
        The gain-ratio floor $a_c$, numpy for numpy inputs and a PyTensor variable for
        symbolic inputs.
    """
    return (1.0 - np.exp(-1.0 / tau_G)) / (np.exp(1.0 / tau_F) - 1.0)
