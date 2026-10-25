"""Offset-experiment tables and figure for the runner application.

Loads the offset posterior (``cache/runner_application_offset.nc``) and, for comparison, the
baseline (``cache/runner_application_full.nc``), then:

* prints a LaTeX table comparing the shared-stock parameters under the baseline and the
  offset model (posterior mean, 95% CI, and split-R-hat), so the reader can see whether the
  readout shift disturbs the fitness/fatigue timescales;
* prints the posterior summary of the performance offset ``d`` (mean, mode, 95% CI, P(d<0))
  recovered from the marginalized per-lag responsibilities; and
* writes a two-panel black-and-white figure: (a) the marginal posterior ``P(d | y)`` over the
  integer lags, and (b) the joint ``(tau_F, d)`` draws that reveal the offset/timescale ridge.

Run::

    uv run --with netCDF4 --with matplotlib python scripts/plot_runner_offset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib  # ty: ignore[unresolved-import]

matplotlib.use("Agg")

import _figstyle  # noqa: E402
import arviz as az  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  # ty: ignore[unresolved-import]
import numpy as np  # noqa: E402

_figstyle.apply()

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
DEFAULT_OFFSET = CACHE / "runner_application_offset.nc"
DEFAULT_BASELINE = CACHE / "runner_application_full.nc"

# Labels for the shared-stock parameters compared across the two fits.
_PARAMS = [
    ("k_F", "$k_F$"),
    ("k_G", "$k_G$"),
    ("tau_F", "$\\tau_F$"),
    ("tau_G", "$\\tau_G$"),
    ("theta", "$\\theta$"),
    ("gamma_speed", "$\\gamma^{\\text{perf}}$"),
    ("admiss", "$a_c$"),
]


def _with_admiss(post: Any) -> Any:
    """Attach the Ceddia (2026) admissibility margin ``admiss`` recomputed from the draws."""
    margin = post["k_G"] / (1.0 - np.exp(-1.0 / post["tau_G"])) - post["k_F"] / np.expm1(
        1.0 / post["tau_F"]
    )
    return post.assign(admiss=margin)


def _summary(post: az.InferenceData, name: str) -> tuple[float, float, float, float]:
    """Return ``(mean, 2.5th, 97.5th, r_hat)`` for one variable's pooled draws."""
    draws = np.asarray(post[name]).reshape(-1)
    r_hat = float(az.rhat(post[name]).values) if post[name].sizes["chain"] > 1 else float("nan")
    return float(draws.mean()), *np.percentile(draws, [2.5, 97.5]), r_hat


def _marginal_d(post: az.InferenceData) -> tuple[np.ndarray, np.ndarray]:
    """Return the integer lags and the marginal ``P(d | y)`` from the responsibilities."""
    w = np.asarray(post["offset_w_speed"])  # (chain, draw, n_lags)
    wbar = w.reshape(-1, w.shape[-1]).mean(0)
    half = w.shape[-1] // 2
    lags = np.arange(-half, half + 1)
    return lags, wbar / wbar.sum()


def main(offset_nc: Path = DEFAULT_OFFSET, baseline_nc: Path = DEFAULT_BASELINE) -> None:
    """Print the comparison tables and render the offset-posterior figure.

    Parameters
    ----------
    offset_nc : Path
        Cached posterior of the offset model (carries ``offset_w_speed``/``offset_d_speed``).
    baseline_nc : Path
        Cached posterior of the baseline (no-offset) model for side-by-side comparison.
    """
    off = _with_admiss(az.from_netcdf(offset_nc).posterior)
    base = _with_admiss(az.from_netcdf(baseline_nc).posterior) if baseline_nc.exists() else None

    # --- Table: baseline vs offset shared-stock parameters ---------------------
    print("\n=== Shared-stock parameters: baseline vs offset ===")
    print("param & baseline mean (95% CI) & offset mean (95% CI) & $\\hat R_{\\text{off}}$ \\\\")
    for name, label in _PARAMS:
        if name not in off:
            continue
        om, olo, ohi, orhat = _summary(off, name)
        if base is not None and name in base:
            bm, blo, bhi, _ = _summary(base, name)
            bcol = f"{bm:.3f} ({blo:.3f}, {bhi:.3f})"
        else:
            bcol = "--"
        print(f"{label} & {bcol} & {om:.3f} ({olo:.3f}, {ohi:.3f}) & {orhat:.2f} \\\\")

    # --- Table: posterior of the performance offset d --------------------------
    lags, pmf = _marginal_d(off)
    mean_d = float((lags * pmf).sum())
    mode_d = int(lags[int(np.argmax(pmf))])
    cum = np.cumsum(pmf)
    lo_d = int(lags[np.searchsorted(cum, 0.025)])
    hi_d = int(lags[np.searchsorted(cum, 0.975)])
    p_neg = float(pmf[lags < 0].sum())
    _, _, _, d_rhat = _summary(off, "offset_d_speed")
    print("\n=== Performance offset d (days) ===")
    print(f"mean     & {mean_d:.1f} \\\\")
    print(f"mode     & {mode_d} \\\\")
    print(f"95% CI   & [{lo_d}, {hi_d}] \\\\")
    print(f"P(d<0)   & {p_neg:.2f} \\\\")
    print(f"r_hat(E[d]) & {d_rhat:.2f} \\\\")
    per_chain = np.asarray(off["offset_d_speed"].mean("draw")).ravel()
    print("per-chain mean E[d]:", np.round(per_chain, 1))

    # --- Figure: P(d|y) and the (tau_F, d) ridge -------------------------------
    fig, (ax_d, ax_r) = plt.subplots(1, 2, figsize=_figstyle.size(7.0, 2.8))

    ax_d.bar(lags, pmf, width=0.9, color="0.6", edgecolor="black", linewidth=0.4)
    ax_d.axvspan(lo_d - 0.5, hi_d + 0.5, color="0.85", zorder=0)
    ax_d.axvline(mean_d, color="black", linestyle="--", linewidth=1.0)
    ax_d.set_xlabel("Performance offset $d$ (days)")
    ax_d.set_ylabel("$P(d \\mid y)$")
    ax_d.annotate(
        f"mean {mean_d:.1f} d\n95% CI [{lo_d}, {hi_d}]",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=_figstyle.ANNOT,
    )

    tau_f = np.asarray(off["tau_F"]).reshape(-1)
    d_draw = np.asarray(off["offset_d_speed"]).reshape(-1)
    ax_r.scatter(tau_f, d_draw, s=3, color="0.4", alpha=0.35, linewidth=0)
    ax_r.set_xlabel("$\\tau_F$ (days)")
    ax_r.set_ylabel("$\\mathbb{E}[d \\mid \\theta]$ (days)")

    for ax in (ax_d, ax_r):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig_path = offset_nc.with_suffix(".jpg")
    fig.savefig(fig_path, dpi=600, format="jpg")
    print(f"\nsaved figure -> {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset-nc", type=Path, default=DEFAULT_OFFSET)
    parser.add_argument("--baseline-nc", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args()
    main(args.offset_nc, args.baseline_nc)
