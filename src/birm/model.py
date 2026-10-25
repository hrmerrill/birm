"""The impulse-response model specification."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from birm.core import (
    CORES,
    apply_link,
    decay_slope,
    fatigue_gain,
    fatigue_timescale,
    fitness_timescale,
    impulse_response,
    observation_mean,
    orthogonalize,
)
from birm.core.integrator import _ar1, _leaky_varying
from birm.data import Panel

LINKS = ("identity", "log", "logit")
PARAMETERIZATIONS = ("informed", "direct")

# Default prior hyperparameters. Scales marked ``* spread`` are multiplied by the densest
# or per-metric response spread so gains stay O(1) whether or not the panel was standardized.
# Any of these is overridable per-name via ``IRModel.priors``.
_TAU_G = (11.0, 0.3)  # LogNormal(log mu, sigma) days, fatigue timescale (non-busso)
_TAU_V = (3.0, 0.3)  # LogNormal(log mu, sigma) days, Busso accumulation timescale
_LAMBDA_F = (33.0, 0.3)  # LogNormal(log mu, sigma) days, margin: tau_F = 1 + tau_G + lambda_f
_LAMBDA_G = (8.0, 0.4)  # LogNormal(log mu, sigma) margin: tau_G = tau_v + lambda_g (busso)
_THETA = (4.0, 0.5)  # LogNormal(log mu, sigma) driver: k_G = (theta + 1) k_F, r = theta + 1
_K_F_FACTOR = 0.1  # HalfNormal(k_F) sigma = factor * densest spread
# Ceddia (2026) priors are deliberately informative and shrink toward Banister: Ceddia's
# Section 5.1 expects these to be weakly identified from sparse performance tests.
_K_D_SIGMA = 1.0  # HalfNormal for the Ceddia diminishing-returns exponent k_D
_RHO_SIGMA = 0.5  # HalfNormal for rho_G, lambda_rho (informed) or rho_F, rho_G (direct)

# Weakly-informative defaults for the ``direct`` (unconstrained) parameterization: the gains and
# both timescales are sampled independently over wide ranges, with no ordering or gain-ratio
# constraint. This is the ill-conditioned baseline that the ``informed`` reparameterization exists
# to fix, so expect ridges/divergences near tau_F = tau_G.
_K_DIRECT_FACTOR = 1.0  # HalfNormal(k_F), HalfNormal(k_G) sigma = factor * densest spread
_TAU_F_DIRECT = (20.0, 1.0)  # LogNormal(log mu, sigma) days, fitness timescale (direct)
_TAU_G_DIRECT = (5.0, 1.0)  # LogNormal(log mu, sigma) days, fatigue timescale (direct)
_TAU_V_DIRECT = (3.0, 1.0)  # LogNormal(log mu, sigma) days, Busso accumulation timescale (direct)
_ALPHA_FACTOR = 3.0  # Normal(alpha) sigma = factor * metric spread
_GAMMA = (1.0, 1.0)  # Normal(mu, sigma) for estimated readout scales
_PHI_U = (0.0, 1.0)  # Uniform(lower, upper) for the AR(1) coef

# Discrete readout-shift (``offset``) test: a metric's stock is read at r_{t-d}, with ``d`` an
# integer in [-_OFFSET_MAX, _OFFSET_MAX] days marginalized out via logsumexp (so the continuous
# sampler is untouched). The prior over the lags is a 0-centered discrete Gaussian with scale
# _OFFSET_SCALE days, which shrinks the large-|d| basin a flat prior would leave open. See
# IRModel._metric_offset.
_OFFSET_MAX = 14
_OFFSET_SCALE = 7.0  # days, sd of the 0-centered discrete Gaussian prior on the lag
# Causal restriction: the shared stock r_t is a causal convolution of past loads, so reading it at
# r_{t-d} with d<0 (i.e. r_{t+|d|}) lets a response depend on *future* training. With
# _OFFSET_CAUSAL the lag support is clipped to d >= 0 (a half-Gaussian prior), keeping the readout
# strictly causal; set False to restore the symmetric [-_OFFSET_MAX, _OFFSET_MAX] test.
_OFFSET_CAUSAL = True

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
    """The training-load driver $w_t$ of the latent stock shared across metrics.

    Parameters
    ----------
    name : str
        Series name identifying the load in the [Panel][birm.data.Panel].
    """

    name: str


@dataclass(frozen=True)
class Metric:
    r"""One response series $Y^{(j)}$ as a noisy readout of the shared stock.

    Parameters
    ----------
    name : str
        Metric name; must match a key in `Panel.metrics`.
    link : str
        Link function $l^{(j)}$; one of ``identity``, ``log``, ``logit``.
    exog : tuple[str, ...]
        Exogenous predictor columns $X^{(j)}$ (``-> \beta``), read off this metric's rows.
    ar1 : bool
        Whether to include the orthogonalized lag-1 process $u^{(j)}$.
    intercept : bool
        Whether to include the intercept $\alpha^{(j)}$.
    gamma : float, optional
        Fixed scale $\gamma^{(j)}$. If ``None`` the densest metric is auto-pinned to
        $\gamma \equiv 1$ and the rest are estimated.
    offset : bool
        Whether to read this metric's stock at a shifted time $r_{t-d}$ with an integer lag
        $d$ (support $\{0,\dots,14\}$ if causal, else $[-14,14]$) under a 0-centered discrete
        Gaussian prior, marginalized out via ``logsumexp``. The shift applies to the stock term
        only; any ``ar1`` residual and exogenous terms stay contemporaneous.
    """

    name: str
    link: str = "identity"
    exog: tuple[str, ...] = ()
    ar1: bool = False
    intercept: bool = True
    gamma: float | None = None
    offset: bool = False

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

    Selects one impulse-response ``core`` $R_t$ and attaches one observation layer per
    [Metric][birm.model.Metric]. Call [fit][birm.model.IRModel.fit] with a
    [Panel][birm.data.Panel] to sample.

    Parameters
    ----------
    load : Load
        The training-load driver $w_t$.
    core : str
        Impulse-response core; one of ``banister`` (Banister 1975), ``busso`` (fatigue gain
        grows with recent load), ``ceddia`` (diminishing-returns fitness gain and decay times
        that grow with session load, Ceddia 2026), or ``fitness-only`` (no fatigue).
    metrics : tuple[Metric, ...]
        Response metrics sharing the single latent stock.
    priors : dict[str, Prior]
        Optional overrides for named model parameters (e.g. ``k_D``, ``rho_G``, ``tau_v``).
    parameterization : str
        Latent-stock prior geometry; one of ``informed`` or ``direct``.
        ``informed`` (default) samples the identifiable reparameterized margins
        ($\theta, \lambda_F, \lambda_G, \lambda_\rho$) so the gain ratio and timescale ordering
        $\tau_v < \tau_G < \tau_F$ (at every load, for Ceddia) and $k_G > k_F$ hold with
        certainty (off the $\tau_F = \tau_G$ ridge). ``direct`` instead samples $k_F$, $k_G$,
        $\tau_F$, $\tau_G$ (and Busso $\tau_v$, Ceddia $\rho_F, \rho_G$) independently from
        weakly-informative ``HalfNormal``/``LogNormal`` priors over wide ranges, with no
        ordering or gain-ratio constraint -- the ill-conditioned baseline the
        reparameterization is designed to fix.
    """

    load: Load
    core: str
    metrics: tuple[Metric, ...]
    priors: dict[str, Prior] = field(default_factory=dict)
    parameterization: str = "informed"

    def __post_init__(self) -> None:
        """Validate the core selection and metric uniqueness.

        Raises
        ------
        ValueError
            If ``core``/``parameterization`` are unrecognized, no metrics are given,
            or metric names are not unique.
        """
        if self.core not in CORES:
            raise ValueError(f"core must be one of {CORES}, got {self.core!r}")
        if self.parameterization not in PARAMETERIZATIONS:
            raise ValueError(
                f"parameterization must be one of {PARAMETERIZATIONS}, "
                f"got {self.parameterization!r}"
            )
        if not self.metrics:
            raise ValueError("at least one metric is required")
        names = [m.name for m in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError(f"metric names must be unique, got {names}")

    def fit(self, panel: Panel, sampler: dict[str, Any] | None = None) -> Any:
        """Validate this spec against ``panel``, build the model, and sample the posterior.

        Checks that every spec metric exists in the panel with resolvable exog columns, warns on
        unused panel metrics, assembles the PyMC model via
        [build_model][birm.model.IRModel.build_model], then samples.

        Parameters
        ----------
        panel : Panel
            Observations to fit, with metric names matching this spec's
            [Metric][birm.model.Metric] names.
        sampler : dict[str, Any], optional
            Overrides merged over the R&D defaults (`_SAMPLER_DEFAULTS`: nutpie NUTS,
            ``draws=1000``, ``tune=2500``, ``chains=4``, ``target_accept=0.99``), passed through to
            `pymc.sample`.

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

        Validates the spec against the panel, builds the shared latent stock $R_t$ from the
        selected cores, then attaches one observation layer and ``obs_<name>`` likelihood per
        metric.

        Parameters
        ----------
        panel : Panel
            Observations to bind, with metric names matching this spec's
            [Metric][birm.model.Metric] names.

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
        r"""Build the shared latent stock $R_t$ with reparameterized priors.

        Parameters
        ----------
        drive : np.ndarray
            The daily training-load driver $w_t$ on the full grid.
        scale : float
            Spread of the densest metric's response, used to scale the gain priors.

        Returns
        -------
        tuple[Any, Any]
            The ``R`` deterministic (fitness minus fatigue) on the full grid, and the AR(1)
            orthogonalization design $[\mathbf{1}, i_1, i_2]$ built from the **gain-free**
            base integrals (so the residual cannot soak up either dose-response direction).
        """
        if self.parameterization == "direct":
            k_F, tau_F, k_G, tau_G, tau_v = self._stock_params_direct(scale)
        else:
            k_F, tau_F, k_G, tau_G, tau_v = self._stock_params_informed(scale)

        k_D: Any = 0.0
        tau_F1: Any = 0.0
        tau_G1: Any = 0.0
        if self.core == "ceddia":
            k_D, tau_F1, tau_G1 = self._ceddia_params(drive, tau_F, tau_G)

        stock = impulse_response(
            drive,
            k_F=k_F,
            tau_F=tau_F,
            k_G=k_G,
            tau_G=tau_G,
            core=self.core,
            tau_v=tau_v,
            k_D=k_D,
            tau_F1=tau_F1,
            tau_G1=tau_G1,
        )
        # Orthogonalization design for the AR(1) residual: the gain-free base integrals at the
        # current timescales. Sweeping the fitness (and fatigue) integral directions out of u forces
        # the stock -- not the near-unit-root residual -- to carry the dose-response signal, so the
        # gains and timescales stay identified. Net R alone is insufficient: u could still absorb
        # the fatigue direction orthogonal to R and strand tau_G at its prior.
        grid_len = len(drive)
        cols = [pt.ones(grid_len), self._base_integral(drive, tau_F, tau_F1)]
        if self.core != "fitness-only":
            cols.append(self._base_integral(drive, tau_G, tau_G1))
        ortho = pt.stack(cols, axis=1)
        return pm.Deterministic("R", stock), ortho

    def _base_integral(self, drive: np.ndarray, tau: Any, tau_1: Any) -> Any:
        r"""Gain-free integral $\sum_{s<t} w_s e^{-(t-s)/\tau(w_s)}$ for the AR(1) design.

        Parameters
        ----------
        drive : np.ndarray
            The daily training-load driver $w_t$ on the full grid.
        tau : Any
            Decay time constant $\tau$ in days (at zero load for ``ceddia``).
        tau_1 : Any
            Ceddia decay-time slope (days per unit load); unused for the other cores.

        Returns
        -------
        Any
            The gain-free integral, length ``T``.
        """
        if self.core == "ceddia":
            return _leaky_varying(drive, tau + tau_1 * drive, drive)
        return _ar1(pt.exp(-1.0 / tau), drive, exclude_current=True)

    def _ceddia_params(self, drive: np.ndarray, tau_F: Any, tau_G: Any) -> tuple[Any, Any, Any]:
        r"""Sample the Ceddia (2026) diminishing-returns exponent and decay-time slopes.

        Samples $k_D$ and the dimensionless lengthening rates $\rho_F, \rho_G$ (with
        $\rho_F = \rho_G + \lambda_\rho$ under ``informed``, independent under ``direct``), then
        records the slopes $\tau_{F1}, \tau_{G1}$ (days per unit load) via
        ``birm.core.reparam.decay_slope`` at the reference load $\bar w$, the mean
        load over training days. The priors are informative and shrink toward Banister.

        Parameters
        ----------
        drive : np.ndarray
            The daily training-load driver $w_t$ on the full grid.
        tau_F : Any
            Zero-load fitness decay time constant $\tau_F$.
        tau_G : Any
            Zero-load fatigue decay time constant $\tau_G$.

        Returns
        -------
        tuple[Any, Any, Any]
            The ``(k_D, tau_F1, tau_G1)`` tensors.
        """
        positive = drive[drive > 0]
        w_ref = float(positive.mean()) if positive.size else 1.0
        k_D = _rv(self.priors, "k_D", "HalfNormal", sigma=_K_D_SIGMA)
        rho_G = _rv(self.priors, "rho_G", "HalfNormal", sigma=_RHO_SIGMA)
        if self.parameterization == "direct":
            rho_F = _rv(self.priors, "rho_F", "HalfNormal", sigma=_RHO_SIGMA)
        else:
            lambda_rho = _rv(self.priors, "lambda_rho", "HalfNormal", sigma=_RHO_SIGMA)
            rho_F = pm.Deterministic("rho_F", rho_G + lambda_rho)
        tau_F1 = pm.Deterministic("tau_F1", decay_slope(tau_F, rho_F, w_ref))
        tau_G1 = pm.Deterministic("tau_G1", decay_slope(tau_G, rho_G, w_ref))
        return k_D, tau_F1, tau_G1

    def _stock_params_informed(self, scale: float) -> tuple[Any, Any, Any, Any, Any]:
        r"""Sample the identifiable reparameterized stock parameters.

        Samples $k_F$ and the non-negative margins ($\theta, \lambda_F, \lambda_G$), then derives
        $k_G = (\theta + 1) k_F$, $\tau_F = 1 + \tau_G + \lambda_F$, and (Busso)
        $\tau_G = \tau_v + \lambda_G$ as ``Deterministic``s so the ordering
        $\tau_v < \tau_G < \tau_F$ and $k_G > k_F$ hold with certainty.

        Parameters
        ----------
        scale : float
            Spread of the densest metric's response, used to scale the gain prior.

        Returns
        -------
        tuple[Any, Any, Any, Any, Any]
            The ``(k_F, tau_F, k_G, tau_G, tau_v)`` tensors; ``k_G`` is ``0.0`` for
            ``core="fitness-only"`` and ``tau_v`` is ``None`` unless ``core="busso"``.
        """
        k_F = _rv(self.priors, "k_F", "HalfNormal", sigma=_K_F_FACTOR * scale)
        tau_v = None
        if self.core == "busso":
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
        if self.core != "fitness-only":
            theta = _rv(self.priors, "theta", "LogNormal", mu=np.log(_THETA[0]), sigma=_THETA[1])
            k_G = pm.Deterministic("k_G", fatigue_gain(k_F, theta))
        return k_F, tau_F, k_G, tau_G, tau_v

    def _stock_params_direct(self, scale: float) -> tuple[Any, Any, Any, Any, Any]:
        r"""Sample the unconstrained (baseline) stock parameters directly.

        Draws $k_F$, $k_G$, $\tau_F$, $\tau_G$ (and Busso $\tau_v$) from independent
        weakly-informative ``HalfNormal``/``LogNormal`` priors over wide ranges, with no gain-ratio
        or timescale-ordering constraint. This is the ill-conditioned baseline the ``informed``
        reparameterization is designed to fix.

        Parameters
        ----------
        scale : float
            Spread of the densest metric's response, used to scale the gain priors.

        Returns
        -------
        tuple[Any, Any, Any, Any, Any]
            The ``(k_F, tau_F, k_G, tau_G, tau_v)`` tensors; ``k_G`` is ``0.0`` when
            ``core="fitness-only"`` and ``tau_v`` is ``None`` unless ``core="busso"``.
        """
        k_F = _rv(self.priors, "k_F", "HalfNormal", sigma=_K_DIRECT_FACTOR * scale)
        tau_F = _rv(
            self.priors, "tau_F", "LogNormal", mu=np.log(_TAU_F_DIRECT[0]), sigma=_TAU_F_DIRECT[1]
        )
        tau_G = _rv(
            self.priors, "tau_G", "LogNormal", mu=np.log(_TAU_G_DIRECT[0]), sigma=_TAU_G_DIRECT[1]
        )
        tau_v = None
        if self.core == "busso":
            tau_v = _rv(
                self.priors,
                "tau_v",
                "LogNormal",
                mu=np.log(_TAU_V_DIRECT[0]),
                sigma=_TAU_V_DIRECT[1],
            )
        k_G: Any = 0.0
        if self.core != "fitness-only":
            k_G = _rv(self.priors, "k_G", "HalfNormal", sigma=_K_DIRECT_FACTOR * scale)
        return k_F, tau_F, k_G, tau_G, tau_v

    def _metric(
        self, data: Any, metric: Metric, stock: Any, ortho: Any, grid_len: int, dense: str
    ) -> None:
        r"""Attach one metric's observation mean and ``obs_<name>`` likelihood.

        Parameters
        ----------
        data : Any
            The [MetricData][birm.data.MetricData] for this metric (values, grid indices, exog).
        metric : Metric
            The metric spec (link, exog, ar1, intercept, gamma).
        stock : Any
            The shared ``R`` deterministic on the full grid.
        ortho : Any
            The AR(1) orthogonalization design $[\mathbf{1}, i_1, i_2]$ from [_stock][].
        grid_len : int
            Number of grid timesteps $T$.
        dense : str
            Name of the densest metric, auto-pinned to $\gamma \equiv 1$.
        """
        priors = self.priors
        values = data.values
        center = float(np.median(values))
        spread = float(np.std(values)) or 1.0

        if metric.offset:
            self._metric_offset(data, metric, stock, ortho, grid_len, center, spread, dense)
            return

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

    def _metric_offset(
        self,
        data: Any,
        metric: Metric,
        stock: Any,
        ortho: Any,
        grid_len: int,
        center: float,
        spread: float,
        dense: str,
    ) -> None:
        r"""Attach a readout-shift metric: $r_{t-d}$ with the integer lag $d$ marginalized out.

        For each candidate lag $d$ the channel mean
        reads the shared stock at $r_{t-d}$, and the per-lag Gaussian log-likelihoods are combined
        with a 0-centered discrete Gaussian prior (scale `_OFFSET_SCALE` days) via ``logsumexp``
        into one ``pm.Potential`` -- so no discrete variable enters the sampler and the default
        NUTS path is preserved. The lag support is $d \in \{0, \dots, M\}$ when `_OFFSET_CAUSAL`
        (reading $r_{t+|d|}$ with $d<0$ would make the response depend on future loads, since the
        stock is a causal convolution), else the symmetric $\{-M, \dots, M\}$ ($M$ = `_OFFSET_MAX`).
        To avoid an edge/support bias
        (shifts that drop more boundary rows would otherwise look artificially better), the
        likelihood is evaluated on the **common support**: only rows whose shifted index stays on
        the grid for *every* candidate lag. The per-lag
        responsibilities $P(d \mid y, \theta)$ and their mean $\mathbb{E}[d \mid y, \theta]$ are
        recorded as deterministics for posterior recovery.

        Parameters
        ----------
        data : Any
            The [MetricData][birm.data.MetricData] for this metric (values, grid indices, exog).
        metric : Metric
            The metric spec (link, exog, ar1, intercept, gamma).
        stock : Any
            The shared ``R`` deterministic on the full grid.
        ortho : Any
            The AR(1) orthogonalization design $[\mathbf{1}, i_1, i_2]$ from [_stock][]; only used
            when ``metric.ar1`` (the residual is orthogonalized against the *unshifted* design).
        grid_len : int
            Number of grid timesteps $T$.
        center : float
            Median of the metric's response, used to center the intercept prior.
        spread : float
            Spread of the metric's response, used to scale the priors.
        dense : str
            Name of the densest metric, auto-pinned to $\gamma \equiv 1$.
        """
        priors = self.priors
        name = metric.name
        idx = np.asarray(data.idx)
        shifts = np.arange(0 if _OFFSET_CAUSAL else -_OFFSET_MAX, _OFFSET_MAX + 1)
        lo = _OFFSET_MAX  # largest positive lag reads idx - _OFFSET_MAX, must stay >= 0
        hi = grid_len - 1 + int(shifts.min())  # most negative lag reads idx - shifts.min()
        keep = (idx >= lo) & (idx <= hi)
        idx_v = idx[keep]
        values_v = data.values[keep]
        if len(idx_v) == 0:
            raise ValueError(
                f"metric {name!r}: offset=True leaves no rows on the common support [{lo}, {hi}]"
            )

        alpha: Any = 0.0
        if metric.intercept:
            alpha = _rv(priors, f"alpha_{name}", "Normal", mu=center, sigma=_ALPHA_FACTOR * spread)
        if metric.gamma is not None:
            gamma: Any = metric.gamma
        elif name == dense:
            gamma = 1.0
        else:
            gamma = _rv(priors, f"gamma_{name}", "Normal", mu=_GAMMA[0], sigma=_GAMMA[1])

        exog_term: Any = 0.0
        if metric.exog:
            exog = np.column_stack([data.exog[c] for c in metric.exog])[keep]
            beta = _rv(
                priors, f"beta_{name}", "Normal", mu=0.0, sigma=spread, shape=len(metric.exog)
            )
            exog_term = pt.as_tensor_variable(exog) @ beta
        sigma = _rv(priors, f"sigma_{name}", "HalfNormal", sigma=spread)

        # AR(1) residual (if any) is contemporaneous with the observation, so it is added at the
        # observed rows and is shared across all candidate lags; only the stock read is shifted.
        u_term: Any = 0.0
        if metric.ar1:
            u_term = self._ar1_residual(name, ortho, grid_len, spread)[idx_v]

        # (n_shifts, n_rows); every entry is on-grid by the common-support filter above
        gathered = idx_v[None, :] - shifts[:, None]
        linked = apply_link(stock[gathered], metric.link)
        mean = alpha + gamma * linked + (exog_term if metric.exog else 0.0) + u_term
        resid = (values_v[None, :] - mean) / sigma
        loglik = -0.5 * resid**2 - pt.log(sigma) - 0.5 * np.log(2.0 * np.pi)
        log_prior = -0.5 * (shifts / _OFFSET_SCALE) ** 2  # 0-centered discrete Gaussian over lags
        log_prior = log_prior - np.log(np.exp(log_prior).sum())
        ll_shift = loglik.sum(axis=1) + log_prior
        logm = pm.math.logsumexp(ll_shift)
        pm.Potential(f"obs_{name}", logm)

        resp = pt.exp(ll_shift - logm)
        pm.Deterministic(f"offset_w_{name}", resp)
        lags = pt.as_tensor_variable(shifts.astype(float))
        pm.Deterministic(f"offset_d_{name}", (lags * resp).sum())

    def _ar1_residual(self, name: str, ortho: Any, grid_len: int, spread: float) -> Any:
        r"""Build the orthogonalized AR(1) residual $u^{(j)}$ on the full grid.

        The raw lag-1 process is swept against the base-integral design $[\mathbf{1}, i_1,
        i_2]$ so it cannot re-absorb either dose-response direction, keeping the stock gains and
        timescales identified.

        Parameters
        ----------
        name : str
            Metric name, used to namespace the ``phi_u_``/``sigma_u_``/``wU_``/``u_`` variables.
        ortho : Any
            The orthogonalization design $[\mathbf{1}, i_1, i_2]$ from [_stock][].
        grid_len : int
            Number of grid timesteps $T$.
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
            "Uniform",
            lower=_PHI_U[0],
            upper=_PHI_U[1],
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
