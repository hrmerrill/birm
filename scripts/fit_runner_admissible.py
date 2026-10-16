r"""Fit the Section 3.2 runner model with the ``admissible`` gain parameterization.

Identical to ``fit_runner_application.py`` except the latent-stock prior floors the gain
ratio at the Ceddia (2026) admissibility metric $a_c$ via $k_G = (a_c + \theta) k_F$, so
$P(k_G/k_F > a_c) = 1$ in the posterior (rather than the weaker $k_G/k_F > 1$ of the
default ``informed`` geometry). The posterior is cached as a separate netCDF ``.nc`` file.

Run from the birm repo with its own virtualenv (``netCDF4`` supplies the netCDF
writer backend)::

    uv run --with netCDF4 python scripts/fit_runner_admissible.py
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import pandas as pd

from birm import IRModel, Load, Metric, Panel

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
OUT = CACHE / "runner_application_admissible.nc"


def main() -> None:
    """Build the panel from cached data, fit the admissible model, and cache the posterior."""
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
        parameterization="admissible",
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
