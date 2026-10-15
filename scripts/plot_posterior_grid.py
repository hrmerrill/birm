r"""Publication figure: joint posteriors of the gain and timescale pairs across four models.

A 2 x 4 grid. Top row is the joint posterior of $(k_F, k_G)$; bottom row is $(\tau_F, \tau_G)$.
Pass ``--times`` to append a third row of the derived recovery and peak times $(\tau_r, \tau_p)$.
Columns left to right: unconstrained baseline, constrained baseline, joint (no exog),
joint + exog. Each panel shows a KDE heat map of the MCMC samples, the posterior mean, and the
2D 95% highest-density credible contour. Axis ranges are shared within each row so the
unconstrained baseline's spread is directly comparable.

Run::

    uv run --with netCDF4 --with matplotlib python scripts/plot_posterior_grid.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib  # ty: ignore[unresolved-import]
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from birm import Panel

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  # ty: ignore[unresolved-import]

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"

MODELS = (
    "mmgap_baseline.nc",  # unconstrained baseline
    "mmgap_univariate.nc",  # constrained baseline
    "runner_joint_noexog.nc",  # joint, no exog
    "runner_application_full.nc",  # joint + exog
)
# Per-model panel reconstruction for the performance R^2: response metrics plus speed exog columns.
MODEL_METRICS: dict[str, list[str]] = {
    "mmgap_baseline": ["speed"],
    "mmgap_univariate": ["speed"],
    "runner_joint_noexog": ["hrv", "speed"],
    "runner_application_full": ["hrv", "speed"],
}
MODEL_EXOG: dict[str, dict[str, list[str]]] = {
    "runner_application_full": {"hrv": ["load_prev"], "speed": ["wetbulb"]},
}
ROWS = (("k_F", "k_G"), ("tau_F", "tau_G"), ("tau_r", "tau_p"))
LABELS = {
    "k_F": r"$k_F$",
    "k_G": r"$k_G$",
    "tau_F": r"$\tau_F$",
    "tau_G": r"$\tau_G$",
    "tau_r": r"$t_r$",
    "tau_p": r"$t_p$",
}
LETTERS = "abcdefghijkl"

# Shared per-row axes span this multiple of the bounding box of the row's smallest posterior
# region (the tightest model), centered on it. Raise to reveal more of the wider models; lower to
# zoom further into the smallest regions.
AXIS_SCALE = 8.0


def _flat(idata: Any, name: str) -> np.ndarray:
    """Return the posterior draws for ``name`` flattened over chains and draws."""
    arr = idata.posterior[name].to_numpy()
    return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])


def _grid(x: np.ndarray, y: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float]):
    """Evaluate a Gaussian KDE on a grid and return ``(X, Y, Z, level_95)``.

    ``level_95`` is the density threshold whose super-level set holds 95% of the posterior mass.
    """
    kde = gaussian_kde(np.vstack([x, y]))
    gx = np.linspace(*xlim, 220)
    gy = np.linspace(*ylim, 220)
    mesh_x, mesh_y = np.meshgrid(gx, gy)
    dens = kde(np.vstack([mesh_x.ravel(), mesh_y.ravel()])).reshape(mesh_x.shape)
    level = float(np.percentile(kde(np.vstack([x, y])), 5))
    return mesh_x, mesh_y, dens, level


def _bbox(s: np.ndarray) -> tuple[float, float, float]:
    """Return ``(lo, hi, width)`` of a marginal 95% posterior interval."""
    lo, hi = (float(v) for v in np.percentile(s, [2.5, 97.5]))
    return lo, hi, hi - lo


def _row_limits(
    draws: dict[str, dict[str, np.ndarray]], keys: list[str], px: str, py: str
) -> dict[str, tuple[float, float]]:
    """Shared x/y limits: ``AXIS_SCALE`` x the tightest region's bbox, centered, kept >= 0."""
    boxes = {k: {p: _bbox(draws[k][p]) for p in (px, py)} for k in keys}
    best = min(keys, key=lambda k: boxes[k][px][2] * boxes[k][py][2])
    limits: dict[str, tuple[float, float]] = {}
    for p in (px, py):
        lo, hi, width = boxes[best][p]
        center = 0.5 * (lo + hi)
        half = AXIS_SCALE * width / 2.0
        a, b = center - half, center + half
        if a < 0.0:
            b -= a
            a = 0.0
        limits[p] = (a, b)
    return limits


def _area95(x: np.ndarray, y: np.ndarray) -> float:
    """Area of the 2D 95% highest-density region, via a KDE on the samples' own support."""
    kde = gaussian_kde(np.vstack([x, y]))
    level = float(np.percentile(kde(np.vstack([x, y])), 5))
    px, py = 0.15 * (x.max() - x.min()), 0.15 * (y.max() - y.min())
    gx = np.linspace(x.min() - px, x.max() + px, 240)
    gy = np.linspace(y.min() - py, y.max() + py, 240)
    mesh_x, mesh_y = np.meshgrid(gx, gy)
    dens = kde(np.vstack([mesh_x.ravel(), mesh_y.ravel()])).reshape(mesh_x.shape)
    cell = (gx[1] - gx[0]) * (gy[1] - gy[0])
    return float((dens >= level).sum() * cell)


def _panel(long: pd.DataFrame, metrics: list[str], exog: dict[str, list[str]]) -> Panel:
    """Rebuild the standardized panel a model was fit on (``w`` plus its response metrics)."""
    sub = long[long["metric"].isin(["w", *metrics])]
    return Panel.from_long(
        sub, date="date", metric="metric", value="value", load="w", exog=exog or None
    ).standardize()


def _r2_performance(idata: Any, panel: Panel, speed_exog: list[str]) -> tuple[float, float, float]:
    """Bayesian performance $R^2$ (mean, 2.5%, 97.5%) for the speed channel across draws."""
    speed = panel.metrics["speed"]
    r_stock = _flat(idata, "R")[:, speed.idx]
    draws_n = r_stock.shape[0]
    alpha = _flat(idata, "alpha_speed")
    gamma = _flat(idata, "gamma_speed") if "gamma_speed" in idata.posterior else np.ones(draws_n)
    mu = alpha[:, None] + gamma[:, None] * r_stock
    if "beta_speed" in idata.posterior:
        beta = _flat(idata, "beta_speed").reshape(draws_n, -1)
        design = np.column_stack([speed.exog[c] for c in speed_exog])
        mu = mu + beta @ design.T
    var_res = _flat(idata, "sigma_speed") ** 2
    r2 = mu.var(axis=1, ddof=1) / (mu.var(axis=1, ddof=1) + var_res)
    return float(r2.mean()), float(np.percentile(r2, 2.5)), float(np.percentile(r2, 97.5))


def _print_table2(
    idatas: dict[str, Any], draws: dict[str, dict[str, np.ndarray]], keys: list[str]
) -> None:
    """Print LaTeX body rows for Table 2 (posterior conditioning and performance fit)."""
    long = pd.read_csv(CACHE / "model_ready_long.csv", parse_dates=["date"])
    r2 = {
        k: _r2_performance(
            idatas[k], _panel(long, MODEL_METRICS[k], MODEL_EXOG.get(k, {})), ["wetbulb"]
        )
        for k in keys
    }
    corr = {
        p: [float(np.corrcoef(draws[k][p[0]], draws[k][p[1]])[0, 1]) for k in keys]
        for p in (("k_F", "k_G"), ("tau_F", "tau_G"))
    }
    area = {
        p: [_area95(draws[k][p[0]], draws[k][p[1]]) for k in keys]
        for p in (("k_F", "k_G"), ("tau_F", "tau_G"))
    }

    def row(label: str, cells: list[str]) -> str:
        """Join one labelled table row into a LaTeX line."""
        return f"{label} & " + " & ".join(cells) + r" \\"

    print("\n% --- Table 2 body (copy into the tabular) ---")
    print(row(r"Corr($k_F$, $k_G$)", [f"{v:.2f}" for v in corr[("k_F", "k_G")]]))
    print(row(r"Corr($\tau_F$, $\tau_G$)", [f"{v:.2f}" for v in corr[("tau_F", "tau_G")]]))
    print(row(r"$\text{Area}_{0.95}$($k_F$, $k_G$)", [f"{v:.3g}" for v in area[("k_F", "k_G")]]))
    print(
        row(
            r"$\text{Area}_{0.95}$($\tau_F$, $\tau_G$)",
            [f"{v:.3g}" for v in area[("tau_F", "tau_G")]],
        )
    )
    r2_cells = [f"{m:.2f} ({lo:.2f}, {hi:.2f})" for m, lo, hi in (r2[k] for k in keys)]
    print(row(r"$R^2$ (95\% CI)", r2_cells))


def _recovery_peak(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    r"""Derive recovery $t_r$ and peak $t_p$ times (paper Eqs. recovery/peak), per draw."""
    factor = d["tau_F"] * d["tau_G"] / (d["tau_F"] - d["tau_G"])
    t_r = factor * np.log(d["k_G"] / d["k_F"])
    t_p = factor * np.log(d["k_G"] * d["tau_F"] / (d["k_F"] * d["tau_G"]))
    return t_r, t_p


def main(times: bool = False) -> None:
    """Build and save the posterior-grid figure, then print the Table 2 body.

    Parameters
    ----------
    times : bool
        When ``True``, append a third row of the derived recovery $t_r$ and peak $t_p$ times.
    """
    idatas: dict[str, Any] = {f.split(".")[0]: az.from_netcdf(str(CACHE / f)) for f in MODELS}
    keys: list[str] = list(idatas)
    draws: dict[str, dict[str, np.ndarray]] = {
        k: {p: _flat(idatas[k], p).ravel() for p in ("k_F", "k_G", "tau_F", "tau_G")} for k in keys
    }
    rows = ROWS if times else ROWS[:2]
    if times:
        for k in keys:
            draws[k]["tau_r"], draws[k]["tau_p"] = _recovery_peak(draws[k])

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 4, figsize=(8.0, 2.15 * nrows), constrained_layout=True)
    for r, (px, py) in enumerate(rows):
        limits = _row_limits(draws, keys, px, py)
        xlim, ylim = limits[px], limits[py]
        for c, k in enumerate(keys):
            ax = axes[r, c]
            x, y = draws[k][px], draws[k][py]
            mesh_x, mesh_y, dens, level = _grid(x, y, xlim, ylim)
            ax.pcolormesh(mesh_x, mesh_y, dens, cmap="Greys", shading="auto", rasterized=True)
            step = max(1, x.size // 2000)
            ax.plot(
                x[::step],
                y[::step],
                ls="",
                marker="o",
                ms=2,
                mfc="black",
                mec="none",
                alpha=0.12,
                rasterized=True,
            )
            ax.contour(mesh_x, mesh_y, dens, levels=[level], colors="black", linewidths=0.9)
            ax.plot(
                x.mean(), y.mean(), marker="o", ms=4.5, mfc="black", mec="white", mew=0.8, ls=""
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.tick_params(labelsize=7)
            ax.text(
                0.95,
                0.03,
                f"({LETTERS[r * 4 + c]})",
                transform=ax.transAxes,
                color="black",
                fontsize=9,
                va="bottom",
                ha="right",
            )
            ax.set_xlabel(LABELS[px], fontsize=10)
            if c == 0:
                ax.set_ylabel(LABELS[py], fontsize=10)
            else:
                ax.tick_params(labelleft=False)

    out = CACHE / "posterior_grid.jpg"
    fig.savefig(out, dpi=600, format="jpg")
    print(f"saved -> {out}")

    _print_table2(idatas, draws, keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--times",
        action="store_true",
        help="append a third row with the derived recovery (t_r) and peak (t_p) times",
    )
    main(times=parser.parse_args().times)
