"""Fit two reduced variants of the Section 3.2 runner model and cache the posteriors.

Both reuse the cached model-ready table from ``cache/_extract_runner_data.py``:

1. ``runner_joint_noexog.nc`` -- the same joint HRV + performance shared-stock model,
   but with NO exogenous predictors (wet-bulb and previous-day load dropped). HRV is
   still the densest readout (gamma == 1) and keeps its AR(1) residual.
2. ``mmgap_univariate.nc`` -- a univariate Bayesian Banister fit to performance (mmGAP)
   only, no exogenous predictors, no AR(1); the single readout is auto-pinned to
   gamma == 1, recovering the original Banister model.

Run::

    uv run --with netCDF4 python scripts/fit_runner_reduced.py
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
    """Fit and cache both reduced models."""
    long = pd.read_csv(CACHE / "model_ready_long.csv", parse_dates=["date"])

    # 1. Joint model, no exogenous predictors.
    joint = IRModel(
        load=Load("w"),
        fitness="banister",
        fatigue="banister",
        metrics=(Metric("hrv", ar1=True), Metric("speed")),
    )
    idata1 = joint.fit(_panel(long, ["hrv", "speed"]), sampler={"random_seed": 0})
    idata1.to_netcdf(str(CACHE / "runner_joint_noexog.nc"))
    print("saved -> cache/runner_joint_noexog.nc")

    # 2. Univariate Bayesian Banister on mmGAP only.
    uni = IRModel(
        load=Load("w"),
        fitness="banister",
        fatigue="banister",
        metrics=(Metric("speed"),),
    )
    idata2 = uni.fit(_panel(long, ["speed"]), sampler={"random_seed": 0})
    idata2.to_netcdf(str(CACHE / "mmgap_univariate.nc"))
    print("saved -> cache/mmgap_univariate.nc")

    for name, idata in (("joint_noexog", idata1), ("mmgap_univariate", idata2)):
        print(f"\n=== {name} ===")
        print(az.summary(idata, var_names=["k_F", "k_G", "tau_F", "tau_G", "admiss"]))


if __name__ == "__main__":
    main()
