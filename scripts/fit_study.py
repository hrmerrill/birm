r"""Fit the three simulation-study models to the cached datasets and stream metrics to CSV.

Reads ``cache/simulation_study.npz`` (see ``scripts/simulate_study.py``) and fits, for every
``(model, cadence, replicate)`` cell:

- **Baseline** — unconstrained Banister (``parameterization="direct"``) on performance channel 1.
- **Constrained** — constrained Banister (``parameterization="informed"``) on channel 1.
- **Joint** — constrained joint model on both channels.

That is ``3 models x 4 cadences x 30 reps = 360`` fits. Each fit is on the **raw** (unstandardized)
load/performance so the estimates and credible contours are directly comparable to the generative
truth ($k_F=0.01,\ k_G=0.02,\ \tau_F=30,\ \tau_G=5$). The densest channel is auto-pinned to
$\gamma\equiv1$, so the gains stay on the generative scale.

Per-fit metrics (Results / "Simulation study") are appended to ``cache/sim_study_results.csv``:
posterior correlation of $(k_F,k_G)$ and $(\tau_F,\tau_G)$, the 95% HDR area of each pair, whether
the true pair falls inside that contour, posterior-mean estimates with 95% CIs for every impulse
parameter and the derived $t_r$/$t_p$, the Bayesian $R^2$ (mean + 95% CI) of each channel, and
convergence diagnostics. The full MCMC draws are **not** cached.

The run is resumable: existing ``(model, cadence, rep)`` rows are skipped, so the script can be
stopped and restarted without wasting compute. Pass ``--restart`` to wipe the CSV and refit
everything.

Run::

    uv run --with netCDF4 python scripts/fit_study.py
    uv run --with netCDF4 python scripts/fit_study.py --restart
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from birm import IRModel, Load, Metric, Panel

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
DATA = CACHE / "simulation_study.npz"
RESULTS = CACHE / "sim_study_results.csv"

SEED_BASE = 20260101
TRUE = {"k_F": 0.01, "k_G": 0.02, "tau_F": 30.0, "tau_G": 5.0}
PARAMS = ("k_F", "k_G", "tau_F", "tau_G")
CADENCES = ("every_two_weeks", "weekly", "twice_weekly", "four_weekly")
MODELS = ("baseline", "constrained", "joint")

SAMPLER = {"draws": 1000, "tune": 1000, "chains": 4}


def _model(name: str) -> IRModel:
    """Build one of the three study model specifications.

    Parameters
    ----------
    name : str
        One of ``baseline``, ``constrained``, ``joint``.

    Returns
    -------
    IRModel
        The model spec: Banister fitness/fatigue, identity readout, ``direct`` for the baseline
        and ``informed`` otherwise, on one channel (``baseline``/``constrained``) or two
        (``joint``).
    """
    metrics = (Metric("perf1"), Metric("perf2")) if name == "joint" else (Metric("perf1"),)
    param = "direct" if name == "baseline" else "informed"
    return IRModel(
        load=Load("w"),
        fitness="banister",
        fatigue="banister",
        metrics=metrics,
        parameterization=param,
    )


def _panel(data: dict[str, np.ndarray], rep: int, cadence: str, channels: tuple[str, ...]) -> Panel:
    """Build the raw (unstandardized) panel for one replicate and cadence.

    Load is dense on all days; each performance channel is observed only on the cadence days.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Arrays loaded from ``simulation_study.npz``.
    rep : int
        Replicate index.
    cadence : str
        Observation-cadence name.
    channels : tuple[str, ...]
        Performance channels to include (only those the model reads).

    Returns
    -------
    Panel
        Panel over a synthetic daily grid, not standardized (raw gains).
    """
    day = data["day"]
    dates = pd.date_range("2020-01-01", periods=len(day), freq="D")
    idx = data[f"cadence_{cadence}"]

    rows = [pd.DataFrame({"date": dates, "metric": "w", "value": data["load"][rep]})]
    for ch in channels:
        rows.append(pd.DataFrame({"date": dates[idx], "metric": ch, "value": data[ch][rep][idx]}))
    long = pd.concat(rows, ignore_index=True)
    return Panel.from_long(long, date="date", metric="metric", value="value", load="w")


def _flat(idata: Any, name: str) -> np.ndarray:
    """Flatten a posterior variable's draws over chains and draws.

    Parameters
    ----------
    idata : Any
        ``arviz.InferenceData`` posterior.
    name : str
        Variable name.

    Returns
    -------
    np.ndarray
        Draws flattened to ``(n_draws, ...)``.
    """
    arr = idata.posterior[name].to_numpy()
    return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])


def _ci(draws: np.ndarray) -> tuple[float, float, float]:
    """Return ``(mean, 2.5%, 97.5%)`` of a draw vector, ignoring non-finite draws.

    Parameters
    ----------
    draws : np.ndarray
        Posterior draws of a scalar quantity.

    Returns
    -------
    tuple[float, float, float]
        Posterior mean and 95% credible-interval bounds.
    """
    d = draws[np.isfinite(draws)]
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _area_capture(x: np.ndarray, y: np.ndarray, true_x: float, true_y: float) -> tuple[float, int]:
    """Area of the 2D 95% highest-density region and whether the truth lies inside it.

    A Gaussian KDE on the joint draws defines the density; the 95% HDR is its super-level set at
    the 5th percentile of the draws' own densities. Area is that region's measure on a fine grid;
    capture is whether the KDE density at ``(true_x, true_y)`` clears the same level.

    Parameters
    ----------
    x, y : np.ndarray
        Posterior draws of the two parameters.
    true_x, true_y : float
        Generative truth for the pair.

    Returns
    -------
    tuple[float, int]
        The 95% HDR area and a ``1``/``0`` capture indicator.
    """
    kde = gaussian_kde(np.vstack([x, y]))
    level = float(np.percentile(kde(np.vstack([x, y])), 5))

    px, py = 0.15 * (x.max() - x.min()), 0.15 * (y.max() - y.min())
    gx = np.linspace(x.min() - px, x.max() + px, 240)
    gy = np.linspace(y.min() - py, y.max() + py, 240)
    mesh_x, mesh_y = np.meshgrid(gx, gy)
    dens = kde(np.vstack([mesh_x.ravel(), mesh_y.ravel()])).reshape(mesh_x.shape)
    cell = (gx[1] - gx[0]) * (gy[1] - gy[0])
    area = float((dens >= level).sum() * cell)

    captured = int(float(kde(np.array([[true_x], [true_y]]))[0]) >= level)
    return area, captured


def _r2(idata: Any, panel: Panel, name: str) -> tuple[float, float, float]:
    r"""Bayesian $R^2$ (mean, 2.5%, 97.5%) for one channel across draws.

    $R^2 = \operatorname{var}(\mu) / (\operatorname{var}(\mu) + \sigma^2)$ per draw, where
    $\mu = \alpha + \gamma R$ on the channel's observed days (no exog, no AR(1) in the sim).

    Parameters
    ----------
    idata : Any
        Posterior inference data.
    panel : Panel
        The fitted panel (for the channel's grid indices).
    name : str
        Channel name.

    Returns
    -------
    tuple[float, float, float]
        Posterior mean and 95% CI of the Bayesian $R^2$.
    """
    md = panel.metrics[name]
    r_stock = _flat(idata, "R")[:, md.idx]
    n = r_stock.shape[0]
    alpha = _flat(idata, f"alpha_{name}") if f"alpha_{name}" in idata.posterior else np.zeros(n)
    gamma = _flat(idata, f"gamma_{name}") if f"gamma_{name}" in idata.posterior else np.ones(n)
    mu = alpha[:, None] + gamma[:, None] * r_stock
    var_res = _flat(idata, f"sigma_{name}") ** 2
    var_mu = mu.var(axis=1, ddof=1)
    r2 = var_mu / (var_mu + var_res)
    return _ci(r2)


def _recovery_peak(draws: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    r"""Derive $t_r$ (recovery) and $t_p$ (peak) draws from the gain/timescale draws.

    $t_r = \frac{\tau_F\tau_G}{\tau_F-\tau_G}\log(k_G/k_F)$ and
    $t_p = \frac{\tau_F\tau_G}{\tau_F-\tau_G}\log\!\frac{k_G\tau_F}{k_F\tau_G}$.

    Parameters
    ----------
    draws : dict[str, np.ndarray]
        Flattened draws for ``k_F``, ``k_G``, ``tau_F``, ``tau_G``.

    Returns
    -------
    dict[str, np.ndarray]
        Draws for ``t_r`` and ``t_p`` (non-finite where $\tau_F=\tau_G$).
    """
    kF, kG, tF, tG = (draws[p] for p in PARAMS)
    with np.errstate(divide="ignore", invalid="ignore"):
        pref = (tF * tG) / (tF - tG)
        t_r = pref * np.log(kG / kF)
        t_p = pref * np.log((kG * tF) / (kF * tG))
    return {"t_r": t_r, "t_p": t_p}


def _metrics(idata: Any, panel: Panel, model: str, runtime: float) -> dict[str, float]:
    """Collapse one fit's posterior into the flat row of study metrics.

    Parameters
    ----------
    idata : Any
        Posterior inference data.
    panel : Panel
        The fitted panel.
    model : str
        Model name (controls whether channel-2 $R^2$ is recorded).
    runtime : float
        Sampling wall-clock seconds.

    Returns
    -------
    dict[str, float]
        All per-fit metric columns.
    """
    draws = {p: _flat(idata, p) for p in PARAMS}
    row: dict[str, float] = {}

    for p in PARAMS:
        row[f"{p}_mean"], row[f"{p}_lo"], row[f"{p}_hi"] = _ci(draws[p])
    for name, d in _recovery_peak(draws).items():
        row[f"{name}_mean"], row[f"{name}_lo"], row[f"{name}_hi"] = _ci(d)

    row["corr_k"] = float(np.corrcoef(draws["k_F"], draws["k_G"])[0, 1])
    row["corr_tau"] = float(np.corrcoef(draws["tau_F"], draws["tau_G"])[0, 1])
    row["area_k"], row["capture_k"] = _area_capture(
        draws["k_F"], draws["k_G"], TRUE["k_F"], TRUE["k_G"]
    )
    row["area_tau"], row["capture_tau"] = _area_capture(
        draws["tau_F"], draws["tau_G"], TRUE["tau_F"], TRUE["tau_G"]
    )

    row["r2_perf1_mean"], row["r2_perf1_lo"], row["r2_perf1_hi"] = _r2(idata, panel, "perf1")
    if model == "joint":
        row["r2_perf2_mean"], row["r2_perf2_lo"], row["r2_perf2_hi"] = _r2(idata, panel, "perf2")
    else:
        row["r2_perf2_mean"] = row["r2_perf2_lo"] = row["r2_perf2_hi"] = np.nan

    row["admiss_prob"] = float((_flat(idata, "admiss") > 0).mean())

    summ = az.summary(idata, var_names=list(PARAMS))
    row["max_rhat"] = float(summ["r_hat"].max())
    row["min_ess"] = float(summ["ess_bulk"].min())
    row["divergences"] = int(idata.sample_stats["diverging"].sum())
    row["runtime_s"] = float(runtime)
    return row


def _done(restart: bool) -> set[tuple[str, str, int]]:
    """Return the set of already-computed ``(model, cadence, rep)`` keys.

    Parameters
    ----------
    restart : bool
        If ``True``, delete any existing results CSV and start fresh.

    Returns
    -------
    set[tuple[str, str, int]]
        Completed cells to skip.
    """
    if restart and RESULTS.exists():
        RESULTS.unlink()
    if not RESULTS.exists():
        return set()
    prev = pd.read_csv(RESULTS)
    return set(zip(prev["model"], prev["cadence"], prev["rep"].astype(int), strict=True))


def _append(row: dict[str, Any]) -> None:
    """Append one result row to the CSV, writing the header only when the file is new.

    Parameters
    ----------
    row : dict[str, Any]
        A single fit's keys and metrics.
    """
    df = pd.DataFrame([row])
    df.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)


def main() -> None:
    """Fit every outstanding ``(model, cadence, rep)`` cell and stream metrics to the CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart", action="store_true", help="wipe results and refit everything")
    args = parser.parse_args()

    data = dict(np.load(DATA))
    n_reps = data["load"].shape[0]
    done = _done(args.restart)

    for mi, model in enumerate(MODELS):
        spec = _model(model)
        channels = ("perf1", "perf2") if model == "joint" else ("perf1",)
        for ci, cadence in enumerate(CADENCES):
            for rep in range(n_reps):
                if (model, cadence, rep) in done:
                    continue
                panel = _panel(data, rep, cadence, channels)
                seed = SEED_BASE + mi * 100_000 + ci * 1_000 + rep
                t0 = time.perf_counter()
                idata = spec.fit(panel, sampler={**SAMPLER, "random_seed": seed})
                runtime = time.perf_counter() - t0

                row = {"model": model, "cadence": cadence, "rep": rep, "seed": seed}
                row.update(_metrics(idata, panel, model, runtime))
                _append(row)
                print(f"done: {model:<11} {cadence:<15} rep {rep:>2}  ({runtime:5.1f}s)")


if __name__ == "__main__":
    main()
