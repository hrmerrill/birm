"""Rest-gap sweep of the in-silico tournament: Placebo vs 2/3/4/5 rest days before the target.

Reuses the forecasting machinery in ``runner_tournament.py`` -- same fitted posterior, warm-start,
scoring (noise-free predicted speed on day ``T+HORIZON``), and volume-matched routines (shared build
block + a fixed 2-session sharpening block, differing only in the number of trailing rest days).
Prints each routine's posterior probability of being the best. Does not refit.

Run from the birm repo::

    uv run --with netCDF4 python scripts/runner_tournament_restsweep.py
"""

from __future__ import annotations

import numpy as np
import runner_tournament as rt

REST_DAYS = (2, 3, 4, 5)


def main() -> None:
    """Forecast all routines over the posterior and print the win-probability table."""
    panel = rt.build_panel().standardize()
    _, load_mean = panel.scaling["load"]
    drive = np.asarray(panel.load, dtype=float)

    flat, _ = rt.flat_posterior()
    i_f, i_g = rt.warmstart(drive, flat)

    labels = ["Placebo", *(f"Rest{d}" for d in REST_DAYS)]
    load_vecs = [np.zeros(rt.HORIZON), *(rt.routine_loads(d) for d in REST_DAYS)]
    vols = [v.sum() for v in load_vecs[1:]]
    assert np.allclose(vols, vols[0]), "training routines are not volume-matched"

    scores = {
        lab: rt.performance(rt.forecast_static(i_f, i_g, w, flat, load_mean), flat)
        for lab, w in zip(labels, load_vecs, strict=True)
    }
    stacked = np.vstack([scores[lab] for lab in labels])
    winners = stacked.argmax(axis=0)
    probs = {lab: float(np.mean(winners == i)) for i, lab in enumerate(labels)}

    c_speed, s_speed = panel.scaling["speed"]
    print(f"draws={flat['k_F'].size}  horizon={rt.HORIZON}d  matched volume={vols[0]:.0f} rTSS")
    print(f"\n--- P(routine = best predicted speed on day T+{rt.HORIZON}) ---")
    for lab in labels:
        raw = scores[lab].mean() * s_speed + c_speed
        print(f"  {lab:8s} {probs[lab]:.3f}   mean speed={raw:+.3f}")


if __name__ == "__main__":
    main()
