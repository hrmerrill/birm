r"""In-silico training-routine tournament for the amateur-runner application (Table 3).

Forecasts the fitted impulse response ``r_t`` an additional HORIZON days under three
competing *static* training routines and reports, for each, the posterior probability that it
maximizes predicted *performance* (running speed) on the target day -- the quantity the paper
scores:

    { P[ w = argmax_w Yhat_speed_{T+HORIZON}(w) ] }_{w in W},  W = {Placebo, Rest2, Rest4},

where the noise-free speed response is ``Yhat_speed = alpha_speed + gamma_speed * r_{T+HORIZON}``
(no exogenous, residual, or white-noise terms enter the score). The three routines are:

* ``Placebo`` -- rest every day (zero load), the detraining floor.
* ``Rest2``   -- a shared build block (the ``[LOW, HIGH, HIGH, LOW, HIGH, HIGH, REST]`` microcycle)
  then a final sharpening block whose last key session is ``SHORT_REST`` days before the target,
  i.e. ``SHORT_REST`` complete rest days precede the target.
* ``Rest4``   -- the *same* build and sharpening block, but the last key session lands
  ``LONG_REST`` days before the target (a longer pre-target recovery window). The two training
  routines carry identical total load, so the comparison isolates the *recovery gap* before the
  target, not volume: the fitness impulse peaks several days after a hard session (long ``tau_F``)
  while fatigue clears in ~1 day (short ``tau_G``), so the posterior is genuinely uncertain about
  how many rest days should precede the target.

Reads the cached posterior ``cache/runner_application_full.nc`` (fit by
``fit_runner_application.py``) -- it does not refit -- and the panel scaling from
``cache/model_ready_long.csv``. For every posterior draw it warm-starts the two leaky integrators
``i_F, i_G`` from the fitted end-of-data state, runs one rollout per routine, maps the final-day
``r_{T+HORIZON}`` to predicted speed, and counts argmax wins. Prints the LaTeX body for
``\label{tab:Table3}``.

Run from the birm repo::

    uv run --with netCDF4 python scripts/runner_tournament.py
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from birm import Panel

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
FIT = CACHE / "runner_application_full.nc"
DATA = CACHE / "model_ready_long.csv"

HORIZON = 42  # forecast length (days)

# Daily training-load levels in raw rTSS units (see paper / user spec).
# Percentile from my data:
# │ Percentile │ rTSS   │
# ├────────────┼────────┤
# │ 80th       │ 71.99  │
# ├────────────┼────────┤
# │ 85th       │ 78.44  │
# ├────────────┼────────┤
# │ 90th       │ 87.81  │
# ├────────────┼────────┤
# │ 95th       │ 108.57 │
# ├────────────┼────────┤
# │ 99th       │ 130.66 │
HIGH = 72  # a HIGH-intensity ("key") session; also the build block's high days
LOW_HIGH_RATIO = 0.592  # from the Kiviniemi paper
# a LOW-intensity / recovery session; the build block's low days (50th pct of training-day loads)
LOW = HIGH * LOW_HIGH_RATIO
REST = 0.0

# All training routines share an identical build block then a fixed final "sharpening" block of
# HIGH sessions; they differ ONLY in how many complete rest days fall between that block and the
# target (the pre-target recovery gap). Because the build and the sharpening block are the same,
# total volume is identical across routines -- the comparison isolates the recovery gap, not volume.
TRA_WEEK = (LOW, HIGH, HIGH, LOW, HIGH, HIGH, REST)  # the periodization microcycle
SHARPEN_SESSIONS = 2  # HIGH sessions in the final sharpening block
SHORT_REST = 2  # last key session 2 days before the target (2 rest days precede it)
LONG_REST = 4  # last key session 4 days before the target (near the fitness-impulse peak)

ROUTINES = ("Placebo", f"Rest{SHORT_REST}", f"Rest{LONG_REST}")
PARAMS = ("k_F", "k_G", "tau_F", "tau_G", "gamma_speed", "alpha_speed")


def build_panel() -> Panel:
    """Rebuild the standardized runner panel so forecast time has the fit's scaling.

    Returns
    -------
    Panel
        The standardized panel (load rescaled to average-day units; hrv and its ``load_prev`` exog
        z-scored), carrying the ``scaling`` needed to map routine loads onto the model's units.
    """
    long = pd.read_csv(DATA, parse_dates=["date"])
    panel = Panel.from_long(
        long,
        date="date",
        metric="metric",
        value="value",
        load="w",
        exog={"hrv": ["load_prev"], "speed": ["wetbulb"]},
    )
    return panel


def flat_posterior() -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load the cached posterior and flatten chains x draws.

    Returns
    -------
    tuple[dict[str, np.ndarray], np.ndarray]
        A dict of flat ``(n_draws,)`` parameter arrays and the fitted stock ``R`` as
        ``(n_draws, T)``.
    """
    post = az.from_netcdf(FIT).posterior
    flat = {}
    for name in PARAMS:
        arr = post[name].values
        flat[name] = arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
    r = post["R"].values
    r = r.reshape(r.shape[0] * r.shape[1], r.shape[2])
    return flat, r


def warmstart(drive: np.ndarray, flat: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """End-of-data leaky-integrator states ``i_F, i_G`` for every posterior draw.

    The leaky recursion ``x_t = a x_{t-1} + drive_t`` (``x_{-1}=0``) gives
    ``x_{T-1} = sum_k drive_rev[k] a^k`` with ``drive_rev`` the reversed drive, evaluated
    vectorized across draws.

    Parameters
    ----------
    drive : np.ndarray
        Observed daily drive ``w_t`` (standardized load), length ``T``.
    flat : dict[str, np.ndarray]
        Flat posterior parameter arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(i_F_end, i_G_end)``, each ``(n_draws,)``.
    """
    a_f = np.exp(-1.0 / flat["tau_F"])[:, None]
    a_g = np.exp(-1.0 / flat["tau_G"])[:, None]
    drive_rev = drive[::-1][None, :]
    k = np.arange(drive.size)[None, :]
    i_f = (drive_rev * a_f**k).sum(axis=1)
    i_g = (drive_rev * a_g**k).sum(axis=1)
    return i_f, i_g


def forecast_static(
    i_f: np.ndarray,
    i_g: np.ndarray,
    loads: np.ndarray,
    flat: dict[str, np.ndarray],
    load_mean: float,
) -> np.ndarray:
    """Final-day ``r_{T+HORIZON}`` for a fixed load schedule, vectorized across draws.

    Parameters
    ----------
    i_f, i_g : np.ndarray
        Warm-start integrator states, each ``(n_draws,)``.
    loads : np.ndarray
        Raw-rTSS daily loads over the horizon, length ``HORIZON``.
    flat : dict[str, np.ndarray]
        Flat posterior parameter arrays.
    load_mean : float
        Load scaling (average-day rTSS) mapping raw load to the model's drive units.

    Returns
    -------
    np.ndarray
        ``r`` on the HORIZONnd forecast day for every draw, ``(n_draws,)``.
    """
    a_f, a_g = np.exp(-1.0 / flat["tau_F"]), np.exp(-1.0 / flat["tau_G"])
    i_f, i_g = i_f.copy(), i_g.copy()
    r = flat["k_F"] * i_f - flat["k_G"] * i_g
    for w in loads:
        d = w / load_mean
        i_f = a_f * i_f + d
        i_g = a_g * i_g + d
        r = flat["k_F"] * i_f - flat["k_G"] * i_g
    return r


def build_block() -> np.ndarray:
    """Build the shared training block: the TRA microcycle tiled over the first half of the horizon.

    Returns
    -------
    np.ndarray
        Raw-rTSS daily loads for days ``0 .. HORIZON//2 - 1``.
    """
    half = HORIZON // 2
    tiled = (list(TRA_WEEK) * (half // len(TRA_WEEK) + 1))[:half]
    return np.array(tiled, dtype=float)


def routine_loads(rest_days: int) -> np.ndarray:
    """Build a training routine: the shared build, then a sharpening block ``rest_days`` out.

    The final ``SHARPEN_SESSIONS`` HIGH sessions end ``rest_days`` days before the target; the
    remaining days are rest. Total volume is independent of ``rest_days`` -- only the timing of
    the final block changes -- so routines differ purely in the pre-target recovery gap.

    Parameters
    ----------
    rest_days : int
        Number of complete rest days between the last HIGH session and the target day.

    Returns
    -------
    np.ndarray
        Raw-rTSS daily loads, length ``HORIZON``.
    """
    loads = np.zeros(HORIZON)
    loads[: HORIZON // 2] = build_block()
    end = HORIZON - 1 - rest_days  # index of the last key session
    loads[end - SHARPEN_SESSIONS + 1 : end + 1] = HIGH
    return loads


def performance(r: np.ndarray, flat: dict[str, np.ndarray]) -> np.ndarray:
    """Map the impulse response to the noise-free predicted (standardized) running speed.

    Parameters
    ----------
    r : np.ndarray
        Final-day impulse response ``r_{T+HORIZON}`` per draw, ``(n_draws,)``.
    flat : dict[str, np.ndarray]
        Flat posterior parameter arrays.

    Returns
    -------
    np.ndarray
        Predicted speed response ``alpha_speed + gamma_speed * r`` per draw, ``(n_draws,)``.
    """
    return flat["alpha_speed"] + flat["gamma_speed"] * r


def _selfcheck(
    drive: np.ndarray, flat: dict[str, np.ndarray], r_fit: np.ndarray, load_mean: float
) -> None:
    """Refutable checks: warm-start reconstructs ``R``; the two routines are volume-matched.

    Parameters
    ----------
    drive : np.ndarray
        Observed daily drive, length ``T``.
    flat : dict[str, np.ndarray]
        Flat posterior parameter arrays.
    r_fit : np.ndarray
        Fitted stock ``R``, ``(n_draws, T)``.
    load_mean : float
        Load scaling.
    """
    i_f, i_g = warmstart(drive, flat)
    r_recon = flat["k_F"] * i_f - flat["k_G"] * i_g
    assert np.allclose(r_recon, r_fit[:, -1], atol=1e-6), "warm-start != fitted R end state"
    short, long = routine_loads(SHORT_REST), routine_loads(LONG_REST)
    assert np.isclose(short.sum(), long.sum()), "the two routines should be volume-matched"
    assert LONG_REST > SHORT_REST and long[-LONG_REST:].sum() == 0.0, "longer recovery gap expected"
    r_high = forecast_static(i_f, i_g, np.full(HORIZON, HIGH), flat, load_mean)
    r_rest = forecast_static(i_f, i_g, np.zeros(HORIZON), flat, load_mean)
    assert performance(r_high, flat).mean() > performance(r_rest, flat).mean(), (
        "HIGH-every-day should beat Placebo in mean predicted speed"
    )
    print("self-check OK (warm-start exact; routines volume-matched; HIGH > Placebo speed signal)")


def win_probabilities(scores: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-routine win probability: the argmax predicted-speed distribution over ``ROUTINES``.

    Parameters
    ----------
    scores : dict[str, np.ndarray]
        Predicted final-day speed per routine, each ``(n_draws,)``.

    Returns
    -------
    dict[str, float]
        Win probability per routine (entries sum to 1).
    """
    stacked = np.vstack([scores[k] for k in ROUTINES])
    winners = stacked.argmax(axis=0)
    return {k: float(np.mean(winners == i)) for i, k in enumerate(ROUTINES)}


def latex_table(probs: dict[str, float]) -> str:
    r"""Render the LaTeX ``tabular`` body for ``\label{tab:Table3}``.

    Parameters
    ----------
    probs : dict[str, float]
        Win probability per routine.

    Returns
    -------
    str
        The table body (routine rows) ready to paste into the paper.
    """
    rows = "\n".join(f"{name} & {probs[name]:.3f} \\\\" for name in ROUTINES)
    return (
        "\\begin{tabular}{lr}\n"
        "Routine  & P(Routine=Best) \\\\ \\midrule\n"
        f"{rows}\n"
        "\\end{tabular}"
    )


def main() -> None:
    """Run the tournament over all posterior draws and print the Table 3 LaTeX body."""
    panel = build_panel().standardize()
    scaling = panel.scaling
    _, load_mean = scaling["load"]
    drive = np.asarray(panel.load, dtype=float)

    flat, r_fit = flat_posterior()
    n_draws = flat["k_F"].size
    print(f"fit={FIT.name}  draws={n_draws}  horizon={HORIZON}d  HIGH={HIGH} LOW={LOW}")

    _selfcheck(drive, flat, r_fit, load_mean)

    i_f, i_g = warmstart(drive, flat)

    r_final = {
        ROUTINES[0]: forecast_static(i_f, i_g, np.zeros(HORIZON), flat, load_mean),
        ROUTINES[1]: forecast_static(i_f, i_g, routine_loads(SHORT_REST), flat, load_mean),
        ROUTINES[2]: forecast_static(i_f, i_g, routine_loads(LONG_REST), flat, load_mean),
    }
    scores = {k: performance(r, flat) for k, r in r_final.items()}

    probs = win_probabilities(scores)

    c_speed, s_speed = scaling["speed"]
    print(f"\n--- P(routine = best predicted speed on day T+{HORIZON}) ---")
    for k in ROUTINES:
        raw = scores[k].mean() * s_speed + c_speed
        print(f"  {k:8s} {probs[k]:.3f}   mean speed={raw:+.3f}")

    print("\n% --- paste into \\label{tab:Table3} ---")
    print(latex_table(probs))


if __name__ == "__main__":
    main()
