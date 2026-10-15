r"""Shared figure styling so every ``width=\textwidth`` JPG renders identical fonts.

Every paper figure is a ``figure*`` included at ``width=\textwidth``, so LaTeX scales
each JPG to the same text width (dgruyter twocolumn "big": ``12*10.5mm + 44mm = 170mm``).
A figure drawn wider than that text width has its text shrunk more: the displayed point
size is ``P * TEXTWIDTH_IN / figwidth_in``. Drawing every figure at exactly
``TEXTWIDTH_IN`` makes that factor 1, so a point size renders identically across figures.

Font sizes reproduce the reference ``posterior_grid.jpg`` (originally drawn at 8.0 in and
down-scaled by LaTeX): ``ref_size * TEXTWIDTH_IN / 8.0``.
"""

from __future__ import annotations

import matplotlib  # ty: ignore[unresolved-import]

TEXTWIDTH_IN = 170.0 / 25.4  # dgruyter twocolumn "big" \textwidth

_REF_WIDTH_IN = 8.0  # posterior_grid.jpg was drawn at this width
_SCALE = TEXTWIDTH_IN / _REF_WIDTH_IN

LABEL = 10.0 * _SCALE  # axis labels (posterior_grid used 10 pt)
TICK = 7.0 * _SCALE  # tick labels (posterior_grid used 7 pt)
ANNOT = 9.0 * _SCALE  # panel letters / inline text (posterior_grid used 9 pt)


def apply() -> None:
    """Install the shared font sizes into ``matplotlib.rcParams``.

    Call once after ``matplotlib.use("Agg")`` and before creating any figure.
    """
    matplotlib.rcParams.update(
        {
            "font.size": LABEL,
            "axes.labelsize": LABEL,
            "axes.titlesize": LABEL,
            "xtick.labelsize": TICK,
            "ytick.labelsize": TICK,
            "legend.fontsize": ANNOT,
            "figure.dpi": 600,
            "savefig.dpi": 600,
        }
    )


def size(ref_w: float, ref_h: float) -> tuple[float, float]:
    r"""Scale a figure's ``(width, height)`` so its width equals ``\textwidth``.

    Preserves the figure's aspect ratio, so its on-page geometry is unchanged while the
    LaTeX scale factor becomes 1 (text no longer shrinks).

    Parameters
    ----------
    ref_w : float
        Original intended figure width in inches.
    ref_h : float
        Original intended figure height in inches.

    Returns
    -------
    tuple[float, float]
        ``(TEXTWIDTH_IN, ref_h * TEXTWIDTH_IN / ref_w)``.
    """
    return (TEXTWIDTH_IN, ref_h * TEXTWIDTH_IN / ref_w)


if __name__ == "__main__":  # pragma: no cover
    # Displayed pt = P * TEXTWIDTH_IN / figwidth. At figwidth == TEXTWIDTH_IN every figure
    # renders LABEL/TICK/ANNOT verbatim; the reference's old 8.0-in render matches too.
    w, h = size(8.0, 2.15)
    assert abs(w - TEXTWIDTH_IN) < 1e-9
    assert abs(h / w - 2.15 / 8.0) < 1e-9  # aspect preserved
    assert abs(LABEL * TEXTWIDTH_IN / w - LABEL) < 1e-9  # scale factor 1 at textwidth
    assert abs(10.0 * TEXTWIDTH_IN / 8.0 - LABEL) < 1e-9  # reproduces old posterior_grid
    print("ok")
