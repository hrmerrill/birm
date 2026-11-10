"""Test the fatigue component: full ``banister`` vs ``fitness-only`` on the runner application.

Both cores use the paper's Section 3.2 observation layers (HRV: ``load_prev`` exog + AR(1);
speed: ``wetbulb`` exog, no AR(1)) and sampler settings from ``fit_runner_application.py``.
Each core is also fit without the HRV AR(1) term, since without fatigue the orthogonalized
AR(1) residual is less constrained and may absorb the missing variability.
Reports the speed-channel Bayesian $R^2$ (as in Table 2) and PSIS-LOO ELPD per metric.

    uv run --with netCDF4 python scripts/fatigue_significance.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from birm import IRModel, Load, Metric, Panel

CACHE = Path(__file__).resolve().parents[1] / "cache"
CORES = ("banister", "fitness-only")
VARIANTS = [(c, ar1) for ar1 in (True, False) for c in CORES]
SAMPLER: dict[str, Any] = {
    "draws": 1000,
    "tune": 5000,
    "chains": 4,
    "target_accept": 0.995,
    "nuts_sampler": "nutpie",
    "progressbar": False,
    "random_seed": 0,
}


def flat(idata, name: str) -> np.ndarray:
    """Posterior draws of ``name`` with chain and draw stacked on axis 0."""
    arr = idata.posterior[name].values
    return arr.reshape(-1, *arr.shape[2:])


def r2_speed(idata, panel: Panel) -> np.ndarray:
    """Per-draw speed $R^2$, matching ``plot_posterior_grid._r2_performance``."""
    speed = panel.metrics["speed"]
    mu = (
        flat(idata, "alpha_speed")[:, None]
        + flat(idata, "gamma_speed")[:, None] * flat(idata, "R")[:, speed.idx]
    )
    mu = (
        mu
        + flat(idata, "beta_speed").reshape(mu.shape[0], -1)
        @ np.column_stack([speed.exog["wetbulb"]]).T
    )
    v = mu.var(axis=1, ddof=1)
    return v / (v + flat(idata, "sigma_speed") ** 2)


def fit(core: str, ar1: bool, panel: Panel):
    """Fit one core/AR(1) variant (or load its cache) with pointwise log-likelihood attached."""
    path = CACHE / f"fatigue_test_{core}{'' if ar1 else '_noar1'}.nc"
    if path.exists():
        return az.from_netcdf(path)
    model = IRModel(
        load=Load("w"),
        core=core,
        metrics=(
            Metric("hrv", exog=("load_prev",), ar1=ar1),
            Metric("speed", exog=("wetbulb",), ar1=False),
        ),
    )
    with model.build_model(panel):
        idata = pm.sample(**SAMPLER)
        pm.compute_log_likelihood(idata)
    idata.to_netcdf(str(path))
    return idata


def main() -> None:
    """Fit both cores and print R^2 and LOO comparisons."""
    long = pd.read_csv(CACHE / "model_ready_long.csv", parse_dates=["date"])
    panel = Panel.from_long(
        long,
        date="date",
        metric="metric",
        value="value",
        load="w",
        exog={"hrv": ["load_prev"], "speed": ["wetbulb"]},
    ).standardize()
    fits = {f"{c}{'' if ar1 else ' noAR1'}": fit(c, ar1, panel) for c, ar1 in VARIANTS}

    for c, idata in fits.items():
        r2 = r2_speed(idata, panel)
        lo, hi = np.percentile(r2, [2.5, 97.5])
        rhat = az.summary(idata, var_names=["k_F", "tau_F", "gamma_speed"])["r_hat"].max()
        print(
            f"{c:19s} speed R^2 = {r2.mean():.3f} (95% CI {lo:.3f}, {hi:.3f})  max r_hat={rhat:.3f}"
        )

    for var in ("obs_speed", "obs_hrv"):
        print(f"\n--- LOO: {var} ---")
        loos = {c: az.loo(idata, var_name=var) for c, idata in fits.items()}
        for c, res in loos.items():
            print(f"{c}:\n{res}")
        print(az.compare(loos))


if __name__ == "__main__":
    main()
