"""Variance decomposition of overnight HRV into shared-IR, AR1, exogenous, and noise parts.

Addresses item 8 of ``the_paper/REVIEW.md``: with ``rho^HRV_u = 0.932`` the
orthogonalized AR1 residual ``u^HRV_t`` has a large stationary variance, so a
decomposition is needed to show how much of HRV's temporal variability the *shared*
impulse response ``r_t`` actually explains versus the HRV-only structural term ``u_t``.

Loads the cached posterior (``cache/runner_application_full.nc``) and the model-ready
panel, prints the posterior decomposition (per-draw temporal variances of each additive
component on the observed HRV days, as fractions of their sum), and writes a one-panel
black-and-white figure of ``u^HRV_t`` over time with a 95% credible band.

Run::

    uv run --with netCDF4 --with matplotlib python scripts/plot_variance_decomp.py
"""

from __future__ import annotations

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
NC = CACHE / "runner_application_full.nc"
FIG = ROOT / "the_paper" / "runner_u_hrv.jpg"


def _ci(x: np.ndarray) -> tuple[float, float, float]:
    """Return ``(mean, 2.5th, 97.5th)`` percentiles of a flat draw array."""
    return float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def main() -> None:
    """Print the HRV variance decomposition and render the ``u`` time-series figure."""
    long = pd.read_csv(CACHE / "model_ready_long.csv", parse_dates=["date"])
    panel = Panel.from_long(
        long,
        date="date",
        metric="metric",
        value="value",
        load="w",
        exog={"hrv": ["load_prev"], "speed": ["wetbulb"]},
    ).standardize()

    hrv = panel.metrics["hrv"]
    idx = hrv.idx  # grid positions of observed HRV days
    x_hrv = hrv.exog["load_prev"]  # standardized exogenous (previous-day load)
    _, s_hrv = panel.scaling["hrv"]  # HRV (center, scale); u is a deviation, so scale only

    post = az.from_netcdf(NC).posterior

    def flat(name: str) -> np.ndarray:
        """Stack chain/draw into one axis, dropping trailing size-1 dims."""
        return np.asarray(post[name]).reshape(-1, *post[name].shape[2:]).squeeze()

    R = flat("R")  # (D, T) shared impulse response (gamma^HRV == 1)
    u = flat("u_hrv")  # (D, T) orthogonalized AR1 residual on HRV
    beta = flat("beta_hrv")  # (D,)
    sig2_eps = flat("sigma_hrv") ** 2  # (D,) white-noise variance
    sig2_u = flat("sigma_u_hrv") ** 2  # (D,) AR1 innovation variance
    rho = flat("phi_u_hrv")  # (D,)

    # Additive components on observed HRV days (standardized HRV units; identical scale,
    # so variance *fractions* are invariant to the HRV back-transform).
    r_obs = R[:, idx]
    u_obs = u[:, idx]
    x_term = beta[:, None] * x_hrv[None, :]

    v_r = r_obs.var(axis=1)  # shared impulse response
    v_u = u_obs.var(axis=1)  # HRV-only AR1 structure
    v_x = x_term.var(axis=1)  # next-day load suppression
    v_eps = sig2_eps  # white noise
    total = v_r + v_u + v_x + v_eps

    stat_u = sig2_u / (1.0 - rho**2)  # AR1 stationary variance

    print("\n=== HRV variance decomposition (fraction of modeled variance, 95% CI) ===")
    for label, v in [
        ("shared IR r_t", v_r / total),
        ("AR1 residual u_t", v_u / total),
        ("exogenous beta*X", v_x / total),
        ("white noise eps", v_eps / total),
    ]:
        m, lo, hi = _ci(v)
        print(f"{label:22s} {m:.3f} ({lo:.3f}, {hi:.3f})")

    m, lo, hi = _ci(v_r / (v_r + v_u))
    print(f"\nshare of *structural* (r+u) variance from shared IR: {m:.3f} ({lo:.3f}, {hi:.3f})")
    m, lo, hi = _ci(stat_u)
    print(f"stationary variance of u = sig2_u/(1-rho^2):         {m:.3f} ({lo:.3f}, {hi:.3f})")
    m, lo, hi = _ci(sig2_eps)
    print(f"white-noise variance sig2_eps:                       {m:.3f} ({lo:.3f}, {hi:.3f})")

    # --- figure: u^HRV_t over time (back-transformed to raw ms deviations) ----
    dates = panel.grid
    lo_u, mid_u, hi_u = np.percentile(u * s_hrv, [2.5, 50, 97.5], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=_figstyle.size(7.0, 2.8))
    ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=1)
    ax.fill_between(dates, lo_u, hi_u, color="0.6", alpha=0.35, linewidth=0, zorder=2)
    ax.plot(dates, mid_u, color="black", linewidth=1.2, zorder=3)
    ax.set_ylabel("$u^{\\text{HRV}}_t$ (ms)")
    ax.set_xlabel("Date")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG, dpi=600, format="jpg")
    print(f"\nsaved figure -> {FIG}")


if __name__ == "__main__":
    main()
