"""The impulse-response model specification."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from birm.core import (
    admissibility,
    fatigue_gain,
    fatigue_timescale,
    fitness_timescale,
    impulse_response,
    observation_mean,
    orthogonalize,
)
from birm.core.integrator import _ar1
from birm.data import Panel

FITNESS = ("banister", "ceddia_periodize", "ceddia_taper", "ceddia")
FATIGUE = ("banister", "busso", "none")
LINKS = ("identity", "log", "logit")

# Default prior hyperparameters. Scales marked ``* spread`` are multiplied by the densest
# or per-metric response spread so gains stay O(1) whether or not the panel was standardized.
# Any of these is overridable per-name via ``IRModel.priors``.
_TAU_G = (11.0, 0.3)  # LogNormal(log mu, sigma) days, fatigue timescale (non-busso)
_TAU_V = (3.0, 0.3)  # LogNormal(log mu, sigma) days, Busso accumulation timescale
_LAMBDA_F = (33.0, 0.4)  # LogNormal(log mu, sigma) days, margin: tau_F = 1 + tau_G + lambda_f
_LAMBDA_G = (8.0, 0.4)  # LogNormal(log mu, sigma) margin: tau_G = tau_v + lambda_g (busso)
_THETA = (4.0, 0.5)  # LogNormal(log mu, sigma) driver: k_G = (theta + 1) k_F, r = theta + 1
_K_F_FACTOR = 0.1  # HalfNormal(k_F) sigma = factor * densest spread
_K_SIGMA = 1.0  # HalfNormal for the Ceddia periodization/taper strengths k_p, k_a
_ALPHA_FACTOR = 3.0  # Normal(alpha) sigma = factor * metric spread
_GAMMA = (1.0, 1.0)  # Normal(mu, sigma) for estimated readout scales
_PHI_U = (0.8, 0.15, 0.4, 0.98)  # TruncatedNormal(mu, sigma, lower, upper) for the AR(1) coef

# Default sampler settings. The near-unit-root AR(1) residual gives a funnel-like geometry that
# needs the nutpie NUTS backend with long tuning and a high target-accept to mix; ``progressbar`` is
# off so the default path stays headless (nutpie's rich progress bar imports matplotlib). Override
# any of these via ``IRModel.fit(..., sampler={...})``.
_SAMPLER_DEFAULTS: dict[str, Any] = {
    "draws": 1000,
    "tune": 2500,
    "chains": 4,
    "target_accept": 0.99,
    "nuts_sampler": "nutpie",
    "progressbar": False,
    "random_seed": 0,
}


def _rv(priors: dict[str, Prior], name: str, dist: str, **params: Any) -> Any:
    """Create a named PyMC random variable, honoring a user override.

    Parameters
    ----------
    priors : dict[str, Prior]
        Per-name overrides; if ``name`` is present its distribution and params replace the defaults.
    name : str
        Random-variable name registered in the active model.
    dist : str
        Default PyMC distribution class name (e.g. ``"HalfNormal"``).
    **params : Any
        Default keyword arguments for the distribution.

    Returns
    -------
    Any
        The constructed PyMC random variable.
    """
    override = priors.get(name)
    if override is not None:
        return getattr(pm, override.dist)(name, **override.params)
    return getattr(pm, dist)(name, **params)


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
        """Validate this spec against ``panel``, build the model, and sample the posterior.

        Checks that every spec metric exists in the panel with resolvable exog columns, warns on
        unused panel metrics, assembles the PyMC model via :meth:`build_model`, then samples.

        Parameters
        ----------
        panel : Panel
            Observations to fit, with metric names matching this spec's :class:`Metric` names.
        sampler : dict[str, Any], optional
            Overrides merged over the R&D defaults (:data:`_SAMPLER_DEFAULTS`: nutpie NUTS,
            ``draws=1000``, ``tune=2500``, ``chains=4``, ``target_accept=0.99``), passed through to
            :func:`pymc.sample`.

        Returns
        -------
        Any
            An ``arviz.InferenceData`` with the posterior and per-metric ``obs_<name>`` likelihoods.

        Raises
        ------
        ValueError
            If a spec metric is missing from the panel, or references an exog column that metric
            does not carry.
        """
        with self.build_model(panel):
            return pm.sample(**{**_SAMPLER_DEFAULTS, **(sampler or {})})

    def build_model(self, panel: Panel) -> pm.Model:
        """Assemble the PyMC model binding this spec to ``panel``.

        Validates the spec against the panel, builds the shared latent stock :math:`R_t` from the
        selected cores, then attaches one observation layer and ``obs_<name>`` likelihood per
        metric.

        Parameters
        ----------
        panel : Panel
            Observations to bind, with metric names matching this spec's :class:`Metric` names.

        Returns
        -------
        pymc.Model
            The assembled model, ready to sample.
        """
        self._validate(panel)
        drive = np.asarray(panel.load, dtype=float)
        dense = panel.densest()
        scale = float(np.std(panel.metrics[dense].values)) or 1.0
        model = pm.Model()
        with model:
            stock, ortho = self._stock(drive, scale)
            for metric in self.metrics:
                self._metric(panel.metrics[metric.name], metric, stock, ortho, panel.T, dense)
        return model

    def _stock(self, drive: np.ndarray, scale: float) -> tuple[Any, Any]:
        r"""Build the shared latent stock :math:`R_t` with reparameterized, admissible priors.

        Parameters
        ----------
        drive : np.ndarray
            The daily training-load driver :math:`w_t` on the full grid.
        scale : float
            Spread of the densest metric's response, used to scale the gain priors.

        Returns
        -------
        tuple[Any, Any]
            The ``R`` deterministic (fitness minus fatigue) on the full grid, and the AR(1)
            orthogonalization design :math:`[\mathbf{1}, i_1, i_2]` built from the **gain-free**
            base integrals (so the residual cannot soak up either dose-response direction).
        """
        k_F = _rv(self.priors, "k_F", "HalfNormal", sigma=_K_F_FACTOR * scale)
        tau_v = None
        if self.fatigue == "busso":
            tau_v = _rv(self.priors, "tau_v", "LogNormal", mu=np.log(_TAU_V[0]), sigma=_TAU_V[1])
            lambda_g = _rv(
                self.priors, "lambda_g", "LogNormal", mu=np.log(_LAMBDA_G[0]), sigma=_LAMBDA_G[1]
            )
            tau_G = pm.Deterministic("tau_G", fatigue_timescale(tau_v, lambda_g))
        else:
            tau_G = _rv(self.priors, "tau_G", "LogNormal", mu=np.log(_TAU_G[0]), sigma=_TAU_G[1])
        lambda_f = _rv(
            self.priors, "lambda_f", "LogNormal", mu=np.log(_LAMBDA_F[0]), sigma=_LAMBDA_F[1]
        )
        tau_F = pm.Deterministic("tau_F", fitness_timescale(tau_G, lambda_f))

        k_G: Any = 0.0
        if self.fatigue != "none":
            theta = _rv(self.priors, "theta", "LogNormal", mu=np.log(_THETA[0]), sigma=_THETA[1])
            k_G = pm.Deterministic("k_G", fatigue_gain(k_F, theta))
            pm.Deterministic("admiss", admissibility(k_F, k_G, tau_F, tau_G))

        k_p: Any = 0.0
        if self.fitness in ("ceddia_periodize", "ceddia"):
            k_p = _rv(self.priors, "k_p", "HalfNormal", sigma=_K_SIGMA)
        k_a: Any = 0.0
        if self.fitness in ("ceddia_taper", "ceddia"):
            k_a = _rv(self.priors, "k_a", "HalfNormal", sigma=_K_SIGMA)

        stock = impulse_response(
            drive,
            k_F=k_F,
            tau_F=tau_F,
            k_G=k_G,
            tau_G=tau_G,
            fitness=self.fitness,
            fatigue=self.fatigue,
            k_p=k_p,
            k_a=k_a,
            tau_v=tau_v,
        )
        # Orthogonalization design for the AR(1) residual: the gain-free base integrals at the
        # current timescales. Sweeping the fitness (and fatigue) integral directions out of u forces
        # the stock -- not the near-unit-root residual -- to carry the dose-response signal, so the
        # gains and timescales stay identified. Net R alone is insufficient: u could still absorb
        # the fatigue direction orthogonal to R and strand tau_G at its prior.
        grid_len = len(drive)
        cols = [pt.ones(grid_len), _ar1(pt.exp(-1.0 / tau_F), drive)]
        if self.fatigue != "none":
            cols.append(_ar1(pt.exp(-1.0 / tau_G), drive))
        ortho = pt.stack(cols, axis=1)
        return pm.Deterministic("R", stock), ortho

    def _metric(
        self, data: Any, metric: Metric, stock: Any, ortho: Any, grid_len: int, dense: str
    ) -> None:
        r"""Attach one metric's observation mean and ``obs_<name>`` likelihood.

        Parameters
        ----------
        data : Any
            The :class:`~birm.data.MetricData` for this metric (values, grid indices, exog).
        metric : Metric
            The metric spec (link, exog, ar1, intercept, gamma).
        stock : Any
            The shared ``R`` deterministic on the full grid.
        ortho : Any
            The AR(1) orthogonalization design :math:`[\mathbf{1}, i_1, i_2]` from :meth:`_stock`.
        grid_len : int
            Number of grid timesteps :math:`T`.
        dense : str
            Name of the densest metric, auto-pinned to :math:`\gamma \equiv 1`.
        """
        priors = self.priors
        values = data.values
        center = float(np.median(values))
        spread = float(np.std(values)) or 1.0

        alpha: Any = 0.0
        if metric.intercept:
            alpha = _rv(
                priors, f"alpha_{metric.name}", "Normal", mu=center, sigma=_ALPHA_FACTOR * spread
            )

        if metric.gamma is not None:
            gamma: Any = metric.gamma
        elif metric.name == dense:
            gamma = 1.0
        else:
            gamma = _rv(priors, f"gamma_{metric.name}", "Normal", mu=_GAMMA[0], sigma=_GAMMA[1])

        exog = beta = None
        if metric.exog:
            exog = np.column_stack([data.exog[c] for c in metric.exog])
            beta = _rv(
                priors,
                f"beta_{metric.name}",
                "Normal",
                mu=0.0,
                sigma=spread,
                shape=len(metric.exog),
            )

        u_obs = None
        if metric.ar1:
            u_obs = self._ar1_residual(metric.name, ortho, grid_len, spread)[data.idx]

        mean = observation_mean(
            stock[data.idx],
            link=metric.link,
            alpha=alpha,
            gamma=gamma,
            exog=exog,
            beta=beta,
            u=u_obs,
        )
        sigma = _rv(priors, f"sigma_{metric.name}", "HalfNormal", sigma=spread)
        pm.Normal(f"obs_{metric.name}", mu=mean, sigma=sigma, observed=values)

    def _ar1_residual(self, name: str, ortho: Any, grid_len: int, spread: float) -> Any:
        r"""Build the orthogonalized AR(1) residual :math:`u^{(j)}` on the full grid.

        The raw lag-1 process is swept against the base-integral design :math:`[\mathbf{1}, i_1,
        i_2]` so it cannot re-absorb either dose-response direction, keeping the stock gains and
        timescales identified.

        Parameters
        ----------
        name : str
            Metric name, used to namespace the ``phi_u_``/``sigma_u_``/``wU_``/``u_`` variables.
        ortho : Any
            The orthogonalization design :math:`[\mathbf{1}, i_1, i_2]` from :meth:`_stock`.
        grid_len : int
            Number of grid timesteps :math:`T`.
        spread : float
            Spread of the metric's response, used to scale the innovation prior.

        Returns
        -------
        Any
            The ``u_<name>`` deterministic: the load-orthogonal residual on the full grid.
        """
        phi = _rv(
            self.priors,
            f"phi_u_{name}",
            "TruncatedNormal",
            mu=_PHI_U[0],
            sigma=_PHI_U[1],
            lower=_PHI_U[2],
            upper=_PHI_U[3],
        )
        sigma_u = _rv(self.priors, f"sigma_u_{name}", "HalfNormal", sigma=spread)
        wU = pm.Normal(f"wU_{name}", 0.0, 1.0, shape=grid_len)
        u_raw = _ar1(phi, sigma_u * wU)
        return pm.Deterministic(f"u_{name}", orthogonalize(u_raw, ortho))

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
