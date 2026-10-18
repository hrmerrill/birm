"""Two-panel latent-state figure: fitness/fatigue stocks (top) and the AR1 residual u (bottom).

Companion to ``scripts/plot_variance_decomp.py`` (same formatting). The shared impulse response
is ``R_t = f_t - g_t``; only ``R`` is cached, so the fitness stock ``f_t`` and fatigue stock
``g_t`` are reconstructed per posterior draw from ``(k_F, tau_F, k_G, tau_G)`` and the standardized
load drive via the same strictly-causal leaky integral the model uses. Both enter HRV with readout
gain 1, so they share the HRV ms scale (and the AR1 residual ``u^HRV_t``).

Run::

    uv run --with netCDF4 --with matplotlib python scripts/plot_fitness_fatigue.py
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
FIG = ROOT / "the_paper" / "runner_fitness_fatigue.jpg"


def _causal_stock(coef: np.ndarray, drive: np.ndarray) -> np.ndarray:
    """Strictly-causal leaky integral ``sum_{s<t} drive_s coef^(t-s)`` for per-draw ``coef``.

    Mirrors [birm.core.integrator._ar1][] with ``exclude_current=True``.

    Parameters
    ----------
    coef : np.ndarray
        Per-draw decay coefficient ``exp(-1/tau)``, shape ``(D,)``.
    drive : np.ndarray
        Load drive ``w_t`` on the daily grid, shape ``(T,)``.

    Returns
    -------
    np.ndarray
        Accumulated stock, shape ``(D, T)``.
    """
    x = np.zeros((coef.shape[0], drive.shape[0]))
    prev = np.zeros(coef.shape[0])
    for t, w_t in enumerate(drive):
        prev = coef * prev + w_t
        x[:, t] = prev
    return x - drive[None, :]  # exclude same-day term -> sum_{s<t}


def main() -> None:
    """Reconstruct the fitness/fatigue stocks and render the two-panel figure."""
    long = pd.read_csv(CACHE / "model_ready_long.csv", parse_dates=["date"])
    panel = Panel.from_long(
        long,
        date="date",
        metric="metric",
        value="value",
        load="w",
        exog={"hrv": ["load_prev"], "speed": ["wetbulb"]},
    ).standardize()

    drive = np.asarray(panel.load, dtype=float)  # standardized load drive w_t (same as fit)
    _, s_hrv = panel.scaling["hrv"]  # HRV scale; f, g, u are deviations -> scale only

    post = az.from_netcdf(NC).posterior

    def flat(name: str) -> np.ndarray:
        """Stack chain/draw into one axis, dropping trailing size-1 dims."""
        return np.asarray(post[name]).reshape(-1, *post[name].shape[2:]).squeeze()

    k_F, tau_F = flat("k_F"), flat("tau_F")
    k_G, tau_G = flat("k_G"), flat("tau_G")
    R = flat("R")  # (D, T) cached shared impulse response, for validation
    u = flat("u_hrv")  # (D, T) orthogonalized AR1 residual on HRV

    f = k_F[:, None] * _causal_stock(np.exp(-1.0 / tau_F), drive)  # fitness stock
    g = k_G[:, None] * _causal_stock(np.exp(-1.0 / tau_G), drive)  # fatigue stock

    # The reconstruction must reproduce the cached R = f - g exactly (same drive + integrator).
    assert np.allclose(f - g, R, atol=1e-6), "reconstructed f - g does not match cached R"

    dates = panel.grid
    lo_f, mid_f, hi_f = np.percentile(f * s_hrv, [2.5, 50, 97.5], axis=0)
    lo_g, mid_g, hi_g = np.percentile(g * s_hrv, [2.5, 50, 97.5], axis=0)
    lo_u, mid_u, hi_u = np.percentile(u * s_hrv, [2.5, 50, 97.5], axis=0)

    fig, (ax_fg, ax_u) = plt.subplots(2, 1, sharex=True, figsize=_figstyle.size(7.0, 4.6))

    ax_fg.axhline(0.0, color="0.7", linewidth=0.8, zorder=1)
    ax_fg.fill_between(dates, lo_f, hi_f, color="0.6", alpha=0.35, linewidth=0, zorder=2)
    ax_fg.fill_between(dates, lo_g, hi_g, color="0.6", alpha=0.35, linewidth=0, zorder=2)
    ax_fg.plot(dates, mid_f, color="black", linewidth=1.2, zorder=4, label="Fitness")
    ax_fg.plot(
        dates, mid_g, color="black", linewidth=1.2, linestyle="--", zorder=3, label="Fatigue"
    )
    ax_fg.set_ylabel("IR stock (ms)")
    ax_fg.legend(frameon=False, loc="upper left")
    ax_fg.spines[["top", "right"]].set_visible(False)

    ax_u.axhline(0.0, color="0.7", linewidth=0.8, zorder=1)
    ax_u.fill_between(dates, lo_u, hi_u, color="0.6", alpha=0.35, linewidth=0, zorder=2)
    ax_u.plot(dates, mid_u, color="black", linewidth=1.2, zorder=3)
    ax_u.set_ylabel("$u_{0,t}$ (ms)")
    ax_u.set_xlabel("Date")
    ax_u.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG, dpi=600, format="jpg")
    print(f"saved figure -> {FIG}")


if __name__ == "__main__":
    main()
