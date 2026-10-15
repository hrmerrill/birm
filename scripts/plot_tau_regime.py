r"""Illustrate the $(\tau_F,\tau_G)$ correlation regime change across observation cadences.

Fits four **univariate constrained** Banister models to the same representative simulated
dataset (``rep 0``) observed at increasing cadences, then draws a 1x4 panel of the joint
$(\tau_F,\tau_G)$ posterior geometry (MCMC points, KDE heatmap, 95% HDR contour, posterior mean,
true value) in the same black/white style as the geometry figure. Axis limits are shared across
panels so the ridge rotating from a positive (sparse data) to a negative fast--slow trade-off
(dense data) is directly comparable.

Run::

    uv run --with netCDF4 --with matplotlib python scripts/plot_tau_regime.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib  # ty: ignore[unresolved-import]
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  # ty: ignore[unresolved-import]

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"

sys.path.insert(0, str(ROOT / "scripts"))
import _figstyle  # noqa: E402
import fit_study as fs  # noqa: E402
import plot_study as ps  # noqa: E402

_figstyle.apply()

REP = 0  # same dataset in every panel: isolates observation density
PAIR = ("tau_F", "tau_G")


def _cadence_idata(cadence: str) -> Any:
    """Fit (or load the cached) univariate constrained posterior for one cadence.

    Parameters
    ----------
    cadence : str
        Observation-cadence name from :data:`fit_study.CADENCES`.

    Returns
    -------
    arviz.InferenceData
        The posterior for the constrained model on ``REP`` at ``cadence``.
    """
    out = CACHE / f"sim_tau_regime_{cadence}.nc"
    if out.exists():
        return az.from_netcdf(str(out))

    data = dict(np.load(fs.DATA))
    panel = fs._panel(data, REP, cadence, ("perf1",))
    idata = fs._model("constrained").fit(panel, sampler={**fs.SAMPLER, "random_seed": 7})
    idata.to_netcdf(str(out))
    print(f"fit + cached -> {out.name}")
    return idata


def main() -> None:
    """Build the 1x4 tau-correlation regime-change figure across cadences."""
    cadences = list(fs.CADENCES)
    idatas = {c: _cadence_idata(c) for c in cadences}
    draws = {c: (fs._flat(idatas[c], PAIR[0]), fs._flat(idatas[c], PAIR[1])) for c in cadences}

    xs = np.concatenate([draws[c][0] for c in cadences] + [[fs.TRUE[PAIR[0]]]])
    ys = np.concatenate([draws[c][1] for c in cadences] + [[fs.TRUE[PAIR[1]]]])
    pad_x = 0.05 * (xs.max() - xs.min() or 1.0)
    pad_y = 0.05 * (ys.max() - ys.min() or 1.0)
    limits = (xs.min() - pad_x, xs.max() + pad_x, ys.min() - pad_y, ys.max() + pad_y)

    lx, ly = ps.PAIR_LABELS[PAIR]
    true_xy = (fs.TRUE[PAIR[0]], fs.TRUE[PAIR[1]])
    fig, axes = plt.subplots(1, 4, figsize=_figstyle.size(16, 4.0), sharey=True)
    for i, (ax, c) in enumerate(zip(axes, cadences, strict=True)):
        x, y = draws[c]
        ps._contour(ax, x, y, true_xy, "black", limits, log=False)
        ax.set_xlabel(lx)
        if i == 0:
            ax.set_ylabel(ly)
        ps._letter(ax, i, loc="upper left")
    fig.tight_layout(pad=0.4, w_pad=0.4)
    out = CACHE / "sim_tau_regime.jpg"
    fig.savefig(out, dpi=600, format="jpg")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
