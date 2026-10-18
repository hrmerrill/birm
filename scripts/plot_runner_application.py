"""Table 1 values and the Section 3.2 figure for the runner application.

Loads the cached posterior (``cache/runner_application_full.nc``) and the model-ready
data, prints the Table 1 parameter estimates (posterior mean and 95% equal-tailed
credible interval), and writes a black-and-white two-panel figure of observed data
with fitted structural impulse-response trends and 95% credible bands.

Run::

    uv run --with netCDF4 --with matplotlib python scripts/plot_runner_application.py
"""

from __future__ import annotations

import argparse
from itertools import zip_longest
from pathlib import Path

import matplotlib  # ty: ignore[unresolved-import]

matplotlib.use("Agg")

import _figstyle  # noqa: E402
import arviz as az  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  # ty: ignore[unresolved-import]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from birm import Panel  # noqa: E402

_figstyle.apply()

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
DEFAULT_NC = CACHE / "runner_application_full.nc"


def _ci(x: np.ndarray) -> tuple[float, float, float]:
    """Return ``(mean, 2.5th, 97.5th)`` percentiles of a flat draw array."""
    return float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def main(nc_path: Path = DEFAULT_NC) -> None:
    """Print Table 1 and render the observed-vs-fitted figure.

    Parameters
    ----------
    nc_path : Path
        Cached posterior netCDF to load. The figure is written alongside it with the same
        stem (``<stem>.jpg``) so alternate parameterizations don't overwrite each other.
    """
    long = pd.read_csv(CACHE / "model_ready_long.csv", parse_dates=["date"])
    panel = Panel.from_long(
        long,
        date="date",
        metric="metric",
        value="value",
        load="w",
        exog={"hrv": ["load_prev"], "speed": ["wetbulb"]},
    ).standardize()
    c_h, s_h = panel.scaling["hrv"]
    c_s, s_s = panel.scaling["speed"]

    post = az.from_netcdf(nc_path).posterior
    fig_path = nc_path.with_suffix(".jpg")

    def flat(name: str) -> np.ndarray:
        """Stack chain/draw into one axis, dropping trailing size-1 dims."""
        return np.asarray(post[name]).reshape(-1, *post[name].shape[2:]).squeeze()

    # --- Table 1 ---------------------------------------------------------------
    k_F, k_G = flat("k_F"), flat("k_G")
    tau_F, tau_G = flat("tau_F"), flat("tau_G")
    sig2_perf, sig2_hrv = flat("sigma_speed") ** 2, flat("sigma_hrv") ** 2
    alpha_perf, alpha_hrv = flat("alpha_speed"), flat("alpha_hrv")
    beta_perf, beta_hrv = flat("beta_speed"), flat("beta_hrv")
    gamma_perf = flat("gamma_speed")
    rho_hrv = flat("phi_u_hrv")

    rows = [
        ("$k_F$", k_F),
        ("$k_G$", k_G),
        ("$\\tau_F$", tau_F),
        ("$\\tau_G$", tau_G),
        ("$\\alpha_0$", alpha_perf),
        ("$\\alpha_1$", alpha_hrv),
        ("$\\gamma_0$", gamma_perf),
        ("$\\beta_0$", beta_perf),
        ("$\\beta_1$", beta_hrv),
        ("$\\sigma^{2}_{0,\\epsilon}$", sig2_perf),
        ("$\\sigma^{2}_{1,\\epsilon}$", sig2_hrv),
        ("$\\sigma^{2}_{1,u}$", flat("sigma_u_hrv") ** 2),
        ("$\\rho_{1,u}$", rho_hrv),
    ]
    print("\n=== Table 1: posterior mean (95% CI) ===")

    def cell(row: tuple[str, np.ndarray] | None) -> str:
        """Format a ``(label, draws)`` pair as ``label & mean (lo, hi)`` or empty."""
        if row is None:
            return " & "
        m, lo, hi = _ci(row[1])
        return f"{row[0]} & {m:.3f} ({lo:.3f}, {hi:.3f})"

    # gamma_0 (index 6) ends the left column; beta_0 (index 7) starts the right.
    left, right = rows[:7], rows[7:]
    for lrow, rrow in zip_longest(left, right):
        print(f"{cell(lrow)} & {cell(rrow)} \\\\")

    # --- structural trends (alpha + gamma * R, back-transformed to raw units) ---
    R = flat("R")  # (draws, T)
    trend_hrv = (alpha_hrv[:, None] + R) * s_h + c_h
    trend_perf = (alpha_perf[:, None] + gamma_perf[:, None] * R) * s_s + c_s
    dates = panel.grid
    if R.shape[1] != len(dates):
        raise SystemExit(
            f"posterior grid T={R.shape[1]} != default panel grid T={len(dates)} "
            f"({nc_path.name}). This script builds the panel from model_ready_long.csv; a "
            "posterior fit on a re-timestamped/alternate grid cannot be overlaid on these dates. "
            "Use a posterior fit on the default panel (e.g. runner_application_full.nc or "
            "runner_application_offset_hrv.nc)."
        )

    def band(ax: plt.Axes, trend: np.ndarray) -> None:
        """Draw posterior mean line and translucent 95% credible band."""
        lo, mid, hi = np.percentile(trend, [2.5, 50, 97.5], axis=0)
        ax.fill_between(dates, lo, hi, color="0.6", alpha=0.35, linewidth=0)
        ax.plot(dates, mid, color="black", linewidth=1.2)

    mm = pd.read_csv(CACHE / "raw_mmgap.csv", parse_dates=["date"])
    mm = mm[(mm["indoor"] == 0.0) & mm[["temp_c", "rh_pct"]].notna().all(axis=1)]
    hv = pd.read_csv(CACHE / "raw_hrv.csv", parse_dates=["date"])

    fig, (ax_p, ax_h) = plt.subplots(2, 1, sharex=True, figsize=_figstyle.size(7.0, 5.0))

    ax_p.scatter(mm["date"], mm["mmgap"], s=14, color="black", zorder=3)
    band(ax_p, trend_perf)
    ax_p.set_ylabel("Speed (m s$^{-1}$)")

    ax_h.scatter(hv["date"], hv["hrv"], s=6, color="0.3", zorder=3)
    band(ax_h, trend_hrv)
    ax_h.set_ylabel("HRV (ms)")
    ax_h.set_xlabel("Date")

    for ax in (ax_p, ax_h):
        ax.spines[["top", "right"]].set_visible(False)
    fig.align_ylabels((ax_p, ax_h))
    fig.tight_layout()
    fig.savefig(fig_path, dpi=600, format="jpg")
    print(f"\nsaved figure -> {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nc",
        type=Path,
        default=DEFAULT_NC,
        help="cached posterior .nc to plot (default: runner_application_full.nc)",
    )
    main(parser.parse_args().nc)
