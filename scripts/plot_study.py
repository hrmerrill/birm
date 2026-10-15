r"""Simulation-study figures and paper-ready summary lines.

Produces three figures in ``cache/`` and prints the numbers that belong in the text/tables rather
than a figure:

- **Fig 1** ``sim_posterior_geometry.jpg`` -- the identification story on one representative dataset
  (sparsest cadence): the joint posteriors of $(k_F,k_G)$ and $(\tau_F,\tau_G)$ for the three
  models, each with its 95% HDR contour, posterior mean, and the true value. The unconstrained
  baseline's collinear ridge (which misses the truth) versus the constrained/joint blobs centred on
  it make the ill-conditioning visible, so the baseline's artificially small $\tau$ "area" cannot
  be mistaken for precision.
- **Fig 2** ``sim_baseline_vs_constrained.jpg`` -- raw (unpaired) violins of the conditioning
  metrics across cadence for Baseline vs Constrained. The baseline's across-rep variance is its own
  non-convergence, so the comparison is honestly unpaired.
- **Fig 3** ``sim_constrained_vs_joint.jpg`` -- Constrained vs Joint across cadence. The correlation
  panels show the *signed* posterior correlation per model so the sign change with sampling density
  is visible (the $\tau$ ridge is positive under sparse data and becomes a negative fast--slow
  trade-off once the response shape is resolved); the area panels show the within-rep paired
  improvement (Constrained minus Joint), which isolates the auxiliary channel's effect because both
  models share the dataset and primary channel.

The three example fits (Fig 1) are cached as NetCDF and reused on reruns.

Run::

    uv run --with netCDF4 --with matplotlib python scripts/plot_study.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib  # ty: ignore[unresolved-import]
import numpy as np
import pandas as pd
from scipy.stats import binomtest, gaussian_kde

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  # ty: ignore[unresolved-import]

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
RESULTS = CACHE / "sim_study_results.csv"

sys.path.insert(0, str(ROOT / "scripts"))
import _figstyle  # noqa: E402
import fit_study as fs  # noqa: E402

_figstyle.apply()

CADENCES = list(fs.CADENCES)
CAD_LABELS = {
    "every_two_weeks": "2 wk",
    "weekly": "1 wk",
    "twice_weekly": "2/wk",
    "four_weekly": "4/wk",
}
MODEL_COLORS = {"baseline": "#d62728", "constrained": "#1f77b4", "joint": "#2ca02c"}
LETTERS = "abcdefghij"
PAIRS = (("k_F", "k_G"), ("tau_F", "tau_G"))
PAIR_LABELS = {
    ("k_F", "k_G"): (r"$k_F$", r"$k_G$"),
    ("tau_F", "tau_G"): (r"$\tau_F$", r"$\tau_G$"),
}

EXAMPLE_CADENCE = "every_two_weeks"
EXAMPLE_REP = 0


# --------------------------------------------------------------------------- Fig 1: geometry


def _example_idata(model: str) -> Any:
    """Fit (or load the cached) example posterior for one model on the representative dataset.

    Parameters
    ----------
    model : str
        Model name (``baseline``/``constrained``/``joint``).

    Returns
    -------
    arviz.InferenceData
        The posterior for ``EXAMPLE_CADENCE`` / ``EXAMPLE_REP``.
    """
    out = CACHE / f"sim_example_{model}.nc"
    if out.exists():
        return az.from_netcdf(str(out))

    data = dict(np.load(fs.DATA))
    channels = ("perf1", "perf2") if model == "joint" else ("perf1",)
    panel = fs._panel(data, EXAMPLE_REP, EXAMPLE_CADENCE, channels)
    idata = fs._model(model).fit(panel, sampler={**fs.SAMPLER, "random_seed": 7})
    idata.to_netcdf(str(out))
    print(f"fit + cached example -> {out.name}")
    return idata


def _contour(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    true_xy: tuple[float, float],
    color: str,
    limits: tuple[float, float, float, float],
    log: bool = False,
) -> None:
    """Draw a KDE heatmap, its 95% HDR contour, the posterior mean, and the true value.

    Parameters
    ----------
    ax : Any
        Target axes.
    x, y : np.ndarray
        Posterior draws of the parameter pair.
    true_xy : tuple[float, float]
        The generative truth for the pair.
    color : str
        Accent colour for the contour and mean.
    limits : tuple[float, float, float, float]
        Shared ``(lo_x, hi_x, lo_y, hi_y)`` axis limits and KDE grid extent.
    log : bool
        Use log-spaced axes and KDE grid (for strictly positive parameters).
    """
    kde = gaussian_kde(np.vstack([x, y]))
    level = float(np.percentile(kde(np.vstack([x, y])), 5))

    lo_x, hi_x, lo_y, hi_y = limits
    space = (lambda a, b: np.logspace(np.log10(a), np.log10(b), 200)) if log else np.linspace
    gx = space(lo_x, hi_x) if log else np.linspace(lo_x, hi_x, 200)
    gy = space(lo_y, hi_y) if log else np.linspace(lo_y, hi_y, 200)
    mx, my = np.meshgrid(gx, gy)
    dens = kde(np.vstack([mx.ravel(), my.ravel()])).reshape(mx.shape)

    ax.pcolormesh(mx, my, dens, shading="auto", cmap="Greys", rasterized=True)
    step = max(1, x.size // 2000)
    ax.plot(
        x[::step],
        y[::step],
        ls="",
        marker="o",
        ms=2,
        mfc=color,
        mec="none",
        alpha=0.12,
        rasterized=True,
    )
    ax.contour(mx, my, dens, levels=[level], colors=color, linewidths=0.9)
    ax.plot(x.mean(), y.mean(), "o", ms=4.5, mfc=color, mec="white", mew=0.8)
    ax.plot(*true_xy, marker="*", ms=7, mfc="white", mec="black", mew=0.8, ls="")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    ax.tick_params(labelsize=7)


def _letter(ax: Any, i: int, loc: str = "lower right") -> None:
    """Annotate a panel with a parenthesised subfigure letter.

    Parameters
    ----------
    ax : Any
        Target axes.
    i : int
        Zero-based panel index into ``LETTERS``.
    loc : str
        Corner placement: ``"lower right"``, ``"lower left"``, ``"upper right"``,
        or ``"upper left"``.
    """
    xmap = {"left": 0.03, "right": 0.97}
    ymap = {"lower": 0.03, "upper": 0.97}
    vy, hx = loc.split()
    x, y = xmap[hx], ymap[vy]
    ha = hx
    va = "bottom" if vy == "lower" else "top"
    ax.text(x, y, f"({LETTERS[i]})", transform=ax.transAxes, fontsize=_figstyle.ANNOT, ha=ha, va=va)


def fig_geometry() -> None:
    """Build Fig 1: the 2x3 posterior-geometry grid on the representative dataset."""
    idatas = {m: _example_idata(m) for m in fs.MODELS}
    draws = {
        (r, model): (fs._flat(idatas[model], pair[0]), fs._flat(idatas[model], pair[1]))
        for r, pair in enumerate(PAIRS)
        for model in fs.MODELS
    }
    fig, axes = plt.subplots(2, 3, figsize=_figstyle.size(11, 7))
    for r, pair in enumerate(PAIRS):
        log = pair == ("k_F", "k_G")
        xs = np.concatenate([draws[(r, m)][0] for m in fs.MODELS] + [[fs.TRUE[pair[0]]]])
        ys = np.concatenate([draws[(r, m)][1] for m in fs.MODELS] + [[fs.TRUE[pair[1]]]])
        if log:
            limits = (xs.min() / 1.6, xs.max() * 1.6, ys.min() / 1.6, ys.max() * 1.6)
        else:
            pad_x = 0.05 * (xs.max() - xs.min() or 1.0)
            pad_y = 0.05 * (ys.max() - ys.min() or 1.0)
            limits = (xs.min() - pad_x, xs.max() + pad_x, ys.min() - pad_y, ys.max() + pad_y)
        for c, model in enumerate(fs.MODELS):
            ax = axes[r, c]
            x, y = draws[(r, model)]
            true_xy = (fs.TRUE[pair[0]], fs.TRUE[pair[1]])
            _contour(ax, x, y, true_xy, "black", limits, log=log)
            lx, ly = PAIR_LABELS[pair]
            ax.set_xlabel(lx)
            ax.set_ylabel(ly)
            _letter(ax, r * 3 + c, loc="upper left")
    fig.tight_layout(pad=0.4, w_pad=0.4, h_pad=0.6)
    out = CACHE / "sim_posterior_geometry.jpg"
    fig.savefig(out, dpi=600, format="jpg")
    plt.close(fig)
    print(f"saved -> {out}")


# --------------------------------------------------------------------------- violins


def _vpstats(arr: np.ndarray, clip: tuple[float, float] | None = None) -> dict[str, Any]:
    """Build matplotlib violin stats with a KDE extended past the data range.

    Parameters
    ----------
    arr : np.ndarray
        Sample values for one violin.
    clip : tuple[float, float] or None
        Hard ``(low, high)`` bounds to clip the KDE support to (e.g. ``(-1, 1)``
        for a correlation). ``None`` leaves the support unconstrained.

    Returns
    -------
    dict[str, Any]
        A ``vpstats`` entry for ``Axes.violin`` whose density tapers to zero
        beyond the observed extremes instead of being clipped flat.
    """
    arr = np.asarray(arr, dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    span = (hi - lo) or 1.0
    lo_c, hi_c = lo - 0.3 * span, hi + 0.3 * span
    if clip is not None:
        lo_c, hi_c = max(lo_c, clip[0]), min(hi_c, clip[1])
    coords = np.linspace(lo_c, hi_c, 256)
    try:
        vals = gaussian_kde(arr)(coords)
    except np.linalg.LinAlgError:
        vals = np.zeros_like(coords)
    return {
        "coords": coords,
        "vals": vals,
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(coords[0]),
        "max": float(coords[-1]),
    }


def _grouped_violins(
    ax: Any,
    series: dict[str, list[np.ndarray]],
    colors: dict[str, str],
    clip: tuple[float, float] | None = None,
) -> None:
    """Draw per-cadence violins for one or two models side by side.

    Parameters
    ----------
    ax : Any
        Target axes.
    series : dict[str, list[np.ndarray]]
        Model name -> per-cadence arrays (one array of rep values per cadence).
    colors : dict[str, str]
        Model name -> fill colour.
    clip : tuple[float, float] or None
        Hard bounds passed to :func:`_vpstats` (e.g. ``(-1, 1)`` for correlations).
    """
    names = list(series)
    base = np.arange(len(CADENCES))
    span = 0.0 if len(names) == 1 else 0.22
    offsets = [0.0] if len(names) == 1 else [-span, span]
    for name, off in zip(names, offsets, strict=True):
        stats = [_vpstats(a, clip=clip) for a in series[name]]
        parts = ax.violin(
            stats, positions=base + off, widths=0.38, showmedians=True, showextrema=False
        )
        for body in parts["bodies"]:
            body.set_facecolor(colors[name])
            body.set_alpha(0.55)
            body.set_edgecolor(colors[name])
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.0)
    ax.set_xticks(base)
    ax.set_xticklabels([CAD_LABELS[c] for c in CADENCES])


def _by_cadence(df: pd.DataFrame, model: str, col: str) -> list[np.ndarray]:
    """Return one array of rep values per cadence for a model/column.

    Parameters
    ----------
    df : pd.DataFrame
        Results frame.
    model : str
        Model name.
    col : str
        Column to extract.

    Returns
    -------
    list[np.ndarray]
        Per-cadence value arrays in ``CADENCES`` order.
    """
    sub = df[df.model == model]
    return [sub[sub.cadence == c][col].to_numpy() for c in CADENCES]


def fig_baseline_vs_constrained(df: pd.DataFrame) -> None:
    """Build Fig 2: raw violins of conditioning metrics, Baseline vs Constrained.

    Parameters
    ----------
    df : pd.DataFrame
        Results frame with a ``log_area_tau`` and ``log_ess`` helper column.
    """
    panels = [
        ("corr_k", r"corr$(k_F,k_G)$"),
        ("corr_tau", r"corr$(\tau_F,\tau_G)$"),
        ("log_area_tau", r"$\log_{10}$ area$(\tau_F,\tau_G)$"),
        ("log_ess", r"$\log_{10}$ min ESS"),
    ]
    colors = {m: MODEL_COLORS[m] for m in ("baseline", "constrained")}
    fig, axes = plt.subplots(2, 2, figsize=_figstyle.size(6, 4.0))
    for i, (ax, (col, label)) in enumerate(zip(axes.ravel(), panels, strict=True)):
        series = {m: _by_cadence(df, m, col) for m in colors}
        clip = (-1.0, 1.0) if col in ("corr_k", "corr_tau") else None
        _grouped_violins(ax, series, colors, clip=clip)
        ax.set_ylabel(label)
        ax.set_xlabel("observation cadence")
        _letter(ax, i, loc="lower right")
    handles = [plt.Line2D([], [], color=colors[m], lw=6, alpha=0.55) for m in colors]
    fig.legend(handles, list(colors), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.4, h_pad=1.0)
    out = CACHE / "sim_baseline_vs_constrained.jpg"
    fig.savefig(out, dpi=600, format="jpg")
    plt.close(fig)
    print(f"saved -> {out}")


def _paired_diff(df: pd.DataFrame, col: str, use_abs: bool = False) -> list[np.ndarray]:
    """Per-cadence within-rep differences ``Constrained - Joint`` for a column.

    Parameters
    ----------
    df : pd.DataFrame
        Results frame.
    col : str
        Column to difference.
    use_abs : bool
        Difference magnitudes instead of signed values. Use for correlations,
        where conditioning depends on ``|corr|`` and a sign flip (e.g. -0.9 to
        +0.1) is a collinearity *improvement*, not a jump toward +1.

    Returns
    -------
    list[np.ndarray]
        Per-cadence difference arrays in ``CADENCES`` order.
    """
    out = []
    for c in CADENCES:
        piv = df[df.cadence == c].pivot(index="rep", columns="model", values=col)
        if use_abs:
            piv = piv.abs()
        out.append((piv["constrained"] - piv["joint"]).to_numpy())
    return out


def fig_constrained_vs_joint(df: pd.DataFrame) -> None:
    """Build Fig 3: Constrained vs Joint conditioning across cadence.

    Correlation panels show the *signed* posterior correlation per model (two
    violins), so the sign change with sampling density is visible; the area
    panels show the within-rep paired improvement (Constrained minus Joint).

    Parameters
    ----------
    df : pd.DataFrame
        Results frame with ``log_area_tau``/``log_area_k`` helper columns.
    """
    diff_color = "#7f7f7f"
    panels = [
        ("corr_k", r"corr$(k_F,k_G)$", "level"),
        ("corr_tau", r"corr$(\tau_F,\tau_G)$", "level"),
        ("log_area_k", r"$\Delta\,\log_{10}$ area$(k_F,k_G)$", "diff"),
        ("log_area_tau", r"$\Delta\,\log_{10}$ area$(\tau_F,\tau_G)$", "diff"),
    ]
    level_colors = {m: MODEL_COLORS[m] for m in ("constrained", "joint")}
    fig, axes = plt.subplots(2, 2, figsize=_figstyle.size(6, 4.0))
    for i, (ax, (col, label, kind)) in enumerate(zip(axes.ravel(), panels, strict=True)):
        if kind == "level":
            series = {m: _by_cadence(df, m, col) for m in level_colors}
            _grouped_violins(ax, series, level_colors, clip=(-1.0, 1.0))
        else:
            diffs = _paired_diff(df, col)
            _grouped_violins(ax, {"d": diffs}, {"d": diff_color})
        ax.axhline(0.0, color="black", lw=0.8, ls="--")
        ax.set_ylabel(label)
        ax.set_xlabel("observation cadence")
        _letter(ax, i, loc="upper right")
    colors = {**level_colors, "constrained minus joint": diff_color}
    handles = [plt.Line2D([], [], color=c, lw=6, alpha=0.55) for c in colors.values()]
    fig.legend(handles, list(colors), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.4, h_pad=1.0)
    out = CACHE / "sim_constrained_vs_joint.jpg"
    fig.savefig(out, dpi=600, format="jpg")
    plt.close(fig)
    print(f"saved -> {out}")


# --------------------------------------------------------------------------- summary lines


def _wide(df: pd.DataFrame, col: str, agg: str = "median") -> pd.DataFrame:
    """Model x cadence table of an aggregated column.

    Parameters
    ----------
    df : pd.DataFrame
        Results frame.
    col : str
        Column to aggregate.
    agg : str
        Aggregation (``median``/``mean``/``sum``).

    Returns
    -------
    pd.DataFrame
        Rows = models, columns = cadences.
    """
    t = df.groupby(["model", "cadence"], observed=True)[col].agg(agg).unstack("cadence")
    return t[CADENCES].reindex(fs.MODELS)


def print_summary(df: pd.DataFrame) -> None:
    """Print the paper-ready numbers that are better as text than a figure.

    Parameters
    ----------
    df : pd.DataFrame
        Results frame with helper log columns attached.
    """
    fmt = lambda t: t.round(3).to_string()  # noqa: E731

    print("\n" + "=" * 70)
    print("SIMULATION-STUDY SUMMARY  (rows: baseline/constrained/joint; cols: cadence)")
    print("=" * 70)

    print("\n[convergence] median max r_hat:\n" + fmt(_wide(df, "max_rhat")))
    print("\n[convergence] median min ESS:\n" + fmt(_wide(df, "min_ess")))
    print("\n[convergence] total divergences:\n" + fmt(_wide(df, "divergences", "sum")))

    print("\n[conditioning] median corr(k_F,k_G):\n" + fmt(_wide(df, "corr_k")))
    print("\n[conditioning] median corr(tau_F,tau_G):\n" + fmt(_wide(df, "corr_tau")))
    print("\n[conditioning] median area(k_F,k_G)  (note O(1e-4) scale):")
    print(_wide(df, "area_k").map(lambda v: f"{v:.2e}").to_string())
    print("\n[conditioning] median area(tau_F,tau_G):\n" + fmt(_wide(df, "area_tau")))

    print(
        "\n[coverage] capture rate, (k_F,k_G) pair [nominal 0.95]:\n"
        + fmt(_wide(df, "capture_k", "mean"))
    )
    print(
        "\n[coverage] capture rate, (tau_F,tau_G) pair [nominal 0.95]:\n"
        + fmt(_wide(df, "capture_tau", "mean"))
    )

    print("\n[validation] median R^2 perf1 (design R^2~0.5):\n" + fmt(_wide(df, "r2_perf1_mean")))
    joint_r2b = df[df.model == "joint"].groupby("cadence", observed=True).r2_perf2_mean.median()
    print(
        "\n[validation] median R^2 perf2 (joint aux channel):\n"
        + joint_r2b[CADENCES].round(3).to_string()
    )

    print(
        "\n[admissibility] median P(k_G/k_F > a_c)  (soft diagnostic):\n"
        + fmt(_wide(df, "admiss_prob"))
    )

    print("\n[C->J paired] fraction of reps where Joint is better-conditioned (sign test p):")
    for col, use_abs in (
        ("corr_k", True),
        ("corr_tau", True),
        ("log_area_tau", False),
        ("log_area_k", False),
    ):
        line = []
        for c in CADENCES:
            piv = df[df.cadence == c].pivot(index="rep", columns="model", values=col)
            if use_abs:
                piv = piv.abs()
            d = piv["constrained"] - piv["joint"]  # >0 => joint lower/less collinear => better
            wins = int((d > 0).sum())
            n = int(d.notna().sum())
            p = binomtest(wins, n, 0.5).pvalue
            line.append(f"{CAD_LABELS[c]}:{wins}/{n}(p={p:.1e})")
        print(f"  {col:<13} " + "  ".join(line))


def main() -> None:
    """Produce all three figures and print the summary lines."""
    df = pd.read_csv(RESULTS)
    df["cadence"] = pd.Categorical(df.cadence, CADENCES, ordered=True)
    df["log_area_tau"] = np.log10(df.area_tau)
    df["log_area_k"] = np.log10(df.area_k)
    df["log_ess"] = np.log10(df.min_ess)

    fig_geometry()
    fig_baseline_vs_constrained(df)
    fig_constrained_vs_joint(df)
    print_summary(df)


if __name__ == "__main__":
    main()
