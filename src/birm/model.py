"""The impulse-response model specification."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from birm.data import Panel

FITNESS = ("banister", "ceddia_periodize", "ceddia_taper", "ceddia")
FATIGUE = ("banister", "busso", "none")
LINKS = ("identity", "log", "logit")


@dataclass(frozen=True)
class Prior:
    """A named prior distribution with its keyword parameters.

    Parameters
    ----------
    dist : str
        PyMC distribution name (e.g. ``"Normal"``, ``"HalfNormal"``).
    params : dict[str, Any]
        Keyword arguments passed through to the distribution.
    """

    dist: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Load:
    """The training-load driver :math:`w_t` of the latent stock shared across metrics.

    Parameters
    ----------
    name : str
        Series name identifying the load in the :class:`~birm.data.Panel`.
    """

    name: str


@dataclass(frozen=True)
class Metric:
    r"""One response series :math:`Y^{(j)}` as a noisy readout of the shared stock.

    Parameters
    ----------
    name : str
        Metric name; must match a key in :attr:`birm.data.Panel.metrics`.
    link : str
        Link function :math:`l^{(j)}`; one of ``identity``, ``log``, ``logit``.
    exog : tuple[str, ...]
        Exogenous predictor columns :math:`X^{(j)}` (``-> \beta``), read off this metric's rows.
    ar1 : bool
        Whether to include the orthogonalized lag-1 process :math:`u^{(j)}`.
    intercept : bool
        Whether to include the intercept :math:`\alpha^{(j)}`.
    gamma : float, optional
        Fixed scale :math:`\gamma^{(j)}`. If ``None`` the densest metric is auto-pinned to
        :math:`\gamma \equiv 1` and the rest are estimated.
    """

    name: str
    link: str = "identity"
    exog: tuple[str, ...] = ()
    ar1: bool = False
    intercept: bool = True
    gamma: float | None = None

    def __post_init__(self) -> None:
        """Validate the link choice.

        Raises
        ------
        ValueError
            If ``link`` is not a recognized link function.
        """
        if self.link not in LINKS:
            raise ValueError(
                f"metric {self.name!r}: link must be one of {LINKS}, got {self.link!r}"
            )


@dataclass(frozen=True)
class IRModel:
    r"""Declarative spec of the shared-stock multivariate impulse-response model.

    Selects one impulse-response core :math:`R_t` via ``fitness`` :math:`\times` ``fatigue`` and
    attaches one observation layer per :class:`Metric`. Call :meth:`fit` with a
    :class:`~birm.data.Panel` to sample.

    Parameters
    ----------
    load : Load
        The training-load driver :math:`w_t`.
    fitness : str
        Fitness core :math:`M_F`; one of ``banister``, ``ceddia_periodize``, ``ceddia_taper``,
        ``ceddia``.
    fatigue : str
        Fatigue core; one of ``banister``, ``busso``, ``none``.
    metrics : tuple[Metric, ...]
        Response metrics sharing the single latent stock.
    priors : dict[str, Prior]
        Optional overrides for named model parameters (e.g. ``k_p``, ``k_a``, ``tau_v``).
    """

    load: Load
    fitness: str
    fatigue: str
    metrics: tuple[Metric, ...]
    priors: dict[str, Prior] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the core selection and metric uniqueness.

        Raises
        ------
        ValueError
            If ``fitness``/``fatigue`` are unrecognized, no metrics are given, or metric names
            are not unique.
        """
        if self.fitness not in FITNESS:
            raise ValueError(f"fitness must be one of {FITNESS}, got {self.fitness!r}")
        if self.fatigue not in FATIGUE:
            raise ValueError(f"fatigue must be one of {FATIGUE}, got {self.fatigue!r}")
        if not self.metrics:
            raise ValueError("at least one metric is required")
        names = [m.name for m in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError(f"metric names must be unique, got {names}")

    def fit(self, panel: Panel, sampler: dict[str, Any] | None = None) -> Any:
        """Validate this spec against ``panel`` and sample the posterior.

        Checks that every spec metric exists in the panel with resolvable exog columns, warns on
        unused panel metrics, then samples.

        Parameters
        ----------
        panel : Panel
            Observations to fit, with metric names matching this spec's :class:`Metric` names.
        sampler : dict[str, Any], optional
            Sampler keyword arguments (e.g. ``draws``, ``tune``, ``seed``).

        Returns
        -------
        Any
            An ``arviz.InferenceData`` with the posterior and per-metric likelihoods.

        Raises
        ------
        ValueError
            If a spec metric is missing from the panel, or references an exog column that metric
            does not carry.
        NotImplementedError
            Sampling is not wired up yet (the ``core/`` and ``observation`` layers are pending).
        """
        self._validate(panel)
        raise NotImplementedError(
            "IRModel.fit sampling is not implemented yet; the core stock and observation layers "
            "are still pending. The spec validated cleanly against the panel."
        )

    def _validate(self, panel: Panel) -> None:
        """Check metric/exog alignment between this spec and ``panel``.

        Parameters
        ----------
        panel : Panel
            The panel to bind against.

        Raises
        ------
        ValueError
            If a spec metric is missing from the panel, or references a missing exog column.
        """
        for metric in self.metrics:
            if metric.name not in panel.metrics:
                raise ValueError(
                    f"spec metric {metric.name!r} not found in panel metrics "
                    f"{sorted(panel.metrics)}"
                )
            available = set(panel.metrics[metric.name].exog)
            missing = [c for c in metric.exog if c not in available]
            if missing:
                raise ValueError(
                    f"metric {metric.name!r} references exog {missing} not carried by the panel "
                    f"(has {sorted(available)})"
                )

        unused = set(panel.metrics) - {m.name for m in self.metrics}
        if unused:
            warnings.warn(f"panel metrics {sorted(unused)} are not used by the spec", stacklevel=2)
