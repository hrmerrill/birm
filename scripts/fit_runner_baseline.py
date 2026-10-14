r"""Fit the unconstrained baseline runner model and cache the posterior.

This is the univariate Bayesian Banister on performance (mmGAP) only -- no exogenous
predictors, no AR(1) -- fit with ``parameterization="direct"``: the gains and timescales
($k_F$, $k_G$, $\tau_F$, $\tau_G$) are sampled from independent weakly-informative
``HalfNormal``/``LogNormal`` priors over wide ranges, with NO gain-ratio or timescale-ordering
constraint. It is the ill-conditioned baseline the ``informed`` reparameterization is designed to
fix; expect ridges, divergences, and poorer mixing near the $\tau_F = \tau_G$ singularity.

Compare against ``cache/mmgap_univariate.nc`` (same data, ``informed`` parameterization).

Run::

    uv run --with netCDF4 python scripts/fit_runner_baseline.py
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import pandas as pd

from birm import IRModel, Load, Metric, Panel

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"


def _panel(long: pd.DataFrame, metrics: list[str]) -> Panel:
    """Build and standardize a panel restricted to ``w`` plus the named metrics."""
    sub = long[long["metric"].isin(["w", *metrics])]
    return Panel.from_long(sub, date="date", metric="metric", value="value", load="w").standardize()


def main() -> None:
    """Fit and cache the unconstrained baseline model."""
    long = pd.read_csv(CACHE / "model_ready_long.csv", parse_dates=["date"])

    baseline = IRModel(
        load=Load("w"),
        fitness="banister",
        fatigue="banister",
        metrics=(Metric("speed"),),
        parameterization="direct",
    )
    idata = baseline.fit(_panel(long, ["speed"]), sampler={"random_seed": 0})
    idata.to_netcdf(str(CACHE / "mmgap_baseline.nc"))
    print("saved -> cache/mmgap_baseline.nc")

    print("\n=== mmgap_baseline (direct, unconstrained) ===")
    print(az.summary(idata, var_names=["k_F", "k_G", "tau_F", "tau_G", "admiss"]))
    diverging = int(idata.sample_stats["diverging"].sum())
    print(f"\ndivergences: {diverging}")


if __name__ == "__main__":
    main()
