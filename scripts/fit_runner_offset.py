"""Offset test: refit the runner application letting one channel read a time-shifted stock.

Identical to ``fit_runner_application.py`` except one readout is given an integer time offset ``d``
(support and prior set in ``birm.model``; marginalized via logsumexp), to test whether that channel
is shifted relative to the shared stock. By default the ``speed`` (performance) channel is shifted
and HRV stays the gamma==1 anchor at ``d = 0``. Pass ``--hrv`` to instead shift the HRV channel
(the causal, hypothesis-faithful test of "HRV lags performance": with the causal d>=0 support HRV
reads the *past* stock while speed is anchored at d=0). Writes a separate ``.nc`` so the cached
baseline is preserved.

    uv run --with netCDF4 python scripts/fit_runner_offset.py [out.nc] [--hrv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import arviz as az
import pandas as pd

from birm import IRModel, Load, Metric, Panel

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
SHIFT_HRV = "--hrv" in sys.argv
_positional = [a for a in sys.argv[1:] if not a.startswith("-")]
OUT = Path(_positional[0]) if _positional else CACHE / "runner_application_offset.nc"


def main() -> None:
    """Build the panel, fit the offset model, and cache the posterior."""
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
            # gamma == 1 anchor; shifted (HRV reads r_{t-d}) iff --hrv, else held at d == 0
            Metric("hrv", exog=("load_prev",), ar1=True, offset=SHIFT_HRV),
            # shifted readout unless HRV is the one being shifted
            Metric("speed", exog=("wetbulb",), ar1=False, offset=not SHIFT_HRV),
        ),
    )
    print(f"shifting {'hrv' if SHIFT_HRV else 'speed'}")

    idata = model.fit(panel, sampler={"random_seed": 0, "tune": 5000, "target_accept": 0.995})

    CACHE.mkdir(exist_ok=True)
    idata.to_netcdf(str(OUT))
    print(f"saved posterior -> {OUT}")

    summary = az.summary(
        idata,
        var_names=["k_F", "k_G", "tau_F", "tau_G", "admiss", "theta", "gamma_speed", "offset_d"],
        filter_vars="like",
    )
    print(summary)


if __name__ == "__main__":
    main()
