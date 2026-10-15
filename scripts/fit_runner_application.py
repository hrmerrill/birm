"""Fit the paper's Section 3.2 "Application to an amateur runner" model with ``birm``.

Reads the model-ready tidy table produced by ``cache/_extract_runner_data.py`` and fits
the two-response shared-stock impulse-response model from the paper:

    Y^perf_t = alpha^perf + gamma^perf r_t + beta^perf * wetbulb_t + eps^perf_t
    Y^HRV_t  = alpha^HRV  +            r_t + beta^HRV  * w_{t-1}   + u^HRV_t + eps^HRV_t

HRV is the densest outcome, so ``birm`` auto-pins its readout to gamma == 1; the
performance (``speed``) readout scale gamma^perf is estimated. The AR(1) residual
``u`` is carried on HRV only. The posterior is cached as a netCDF ``.nc`` file.

Run from the birm repo with its own virtualenv (``netCDF4`` supplies the netCDF
writer backend)::

    uv run --with netCDF4 python scripts/fit_runner_application.py
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import pandas as pd

from birm import IRModel, Load, Metric, Panel

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
OUT = CACHE / "runner_application_full.nc"


def main() -> None:
    """Build the panel from cached data, fit the model, and cache the posterior."""
    long = pd.read_csv(CACHE / "model_ready_long.csv", parse_dates=["date"])

    panel = Panel.from_long(
        long,
        date="date",
        metric="metric",
        value="value",
        load="w",
        exog={"hrv": ["load_prev"], "speed": ["wetbulb"]},
    ).standardize()
    print(f"panel: T={panel.T} days, densest={panel.densest()!r}")

    model = IRModel(
        load=Load("w"),
        fitness="banister",
        fatigue="banister",
        metrics=(
            Metric("hrv", exog=("load_prev",), ar1=True),  # gamma == 1 (densest), u^HRV
            Metric("speed", exog=("wetbulb",), ar1=False),  # gamma^perf estimated, no u
        ),
    )

    idata = model.fit(panel, sampler={"random_seed": 0, "tune": 5000, "target_accept": 0.995})

    CACHE.mkdir(exist_ok=True)
    idata.to_netcdf(str(OUT))
    print(f"saved posterior -> {OUT}")

    summary = az.summary(
        idata,
        var_names=["k_F", "k_G", "tau_F", "tau_G", "admiss", "theta", "gamma_speed"],
        filter_vars="like",
    )
    print(summary)


if __name__ == "__main__":
    main()
