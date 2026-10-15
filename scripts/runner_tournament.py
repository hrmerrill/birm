r"""In-silico training-routine tournament for the amateur-runner application (Table 3).

Forecasts the fitted impulse response ``r_t`` an additional 42 days (six weeks) under four
competing training routines and reports, for each routine, the posterior probability that it
produces the highest ``r_{T+42}`` -- the quantity the paper scores:

    { P[ w = argmax_w r_{T+42}(w) ] }_{w in W}.

Scoring is the noise-free impulse response on the 42nd forecast day (no exogenous, residual, or
white-noise terms enter the score); residual/noise terms only drive the *adaptive* HRV routine's
day-by-day decisions. Because the infeasible ``Max`` routine (HIGH load every day) deterministically
maximizes ``r_{T+42}`` in this fitted model, the argmax is taken over the feasible set
``W = {Placebo, TRA, HRV}`` (whose probabilities sum to 1) and ``Max`` is reported separately as the
ceiling: the probability it exceeds every feasible routine. The four routines are:

* ``Placebo`` -- rest every day (zero load), the detraining floor.
* ``Max``     -- the infeasible HIGH-every-day routine.
* ``TRA``     -- predefined periodization: a ``[LOW, HIGH, HIGH, LOW, HIGH, HIGH, REST]`` week.
* ``HRV``     -- the Kiviniemi (2007) HRV-guided closed-loop controller, which simulates the full
  HRV readout ``Y^HRV_t = alpha^HRV + r_t + beta^HRV w_{t-1} + u^HRV_t + eps^HRV_t`` each day
  (de-standardized to raw HRV) and sets the next day's HIGH/LOW/REST from the morning ln-HRV.

Reads the cached posterior ``cache/runner_application_full.nc`` (fit by
``fit_runner_application.py``) -- it does not refit -- and the panel scaling from
``cache/model_ready_long.csv``. For every posterior draw it warm-starts the two leaky integrators
``i_F, i_G`` from the fitted end-of-data state, runs one rollout per routine (one *stochastic*
rollout for HRV), reads ``r_{T+42}``, and counts argmax wins. Prints the LaTeX body for
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

HORIZON = 28  # forecast length (days)
SEED = 0

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
HIGH = 90  # a HIGH-intensity session; also Max's every-day load and TRA's high days
LOW_HIGH_RATIO = 0.592  # from the Kiviniemi paper
LOW = (
    HIGH * LOW_HIGH_RATIO
)  # a LOW-intensity / recovery session; TRA's low days (50th pct of training-day loads)
REST = 0.0

# Kiviniemi (2007) HRV-guided controller settings (Eur J Appl Physiol 101:743-751).
HARD, EASY, RESTLAB = 2, 1, 0
LOAD = {HARD: HIGH, EASY: LOW, RESTLAB: REST}
BASE_WIN = 10  # rolling baseline window (days): reference = mean - SD
TREND = 0.1  # ln-HRV two-day decreasing-trend trigger (ln ms^2)
MAX_TRAIN = 9  # forced REST after this many successive training days
MAX_HIGH = 2  # forced EASY after this many successive HARD days
MAX_REST = 2  # forced EASY after this many successive REST days

ROUTINES = ("Placebo", "Max", "TRA", "HRV")
FEASIBLE = ("Placebo", "TRA", "HRV")  # Max is the infeasible ceiling, excluded from the argmax
PARAMS = (
    "k_F",
    "k_G",
    "tau_F",
    "tau_G",
    "alpha_hrv",
    "beta_hrv",
    "phi_u_hrv",
    "sigma_u_hrv",
    "sigma_hrv",
)


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


def flat_posterior() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load the cached posterior and flatten chains x draws.

    Returns
    -------
    tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]
        A dict of flat ``(n_draws,)`` parameter arrays (``beta_hrv`` squeezed to scalar), the
        fitted stock ``R`` as ``(n_draws, T)``, and the AR(1) residual ``u_hrv`` as
        ``(n_draws, T)``.
    """
    post = az.from_netcdf(FIT).posterior
    flat = {}
    for name in PARAMS:
        arr = post[name].values
        flat[name] = arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
    flat["beta_hrv"] = flat["beta_hrv"][:, 0]  # single exog (load_prev)
    r = post["R"].values
    r = r.reshape(r.shape[0] * r.shape[1], r.shape[2])
    u = post["u_hrv"].values
    u = u.reshape(u.shape[0] * u.shape[1], u.shape[2])
    return flat, r, u


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
    """Final-day ``r_{T+42}`` for a fixed load schedule, vectorized across draws.

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
        ``r`` on the 42nd forecast day for every draw, ``(n_draws,)``.
    """
    a_f, a_g = np.exp(-1.0 / flat["tau_F"]), np.exp(-1.0 / flat["tau_G"])
    i_f, i_g = i_f.copy(), i_g.copy()
    r = None
    for w in loads:
        d = w / load_mean
        i_f = a_f * i_f + d
        i_g = a_g * i_g + d
        r = flat["k_F"] * i_f - flat["k_G"] * i_g
    return r


def _rolling_ref(ln_hrv: list[float], t: int) -> float:
    """Kiviniemi reference = mean - SD over the last ``BASE_WIN`` ln-HRV values through day ``t``.

    Parameters
    ----------
    ln_hrv : list[float]
        Morning ln-HRV series (seeded with real baseline nights).
    t : int
        Absolute index of the current day.

    Returns
    -------
    float
        The daily reference band value.
    """
    lo = max(0, t - BASE_WIN + 1)
    win = ln_hrv[lo : t + 1]
    if len(win) < 2:
        return win[-1]
    return float(np.mean(win) - np.std(win, ddof=1))


def _significant_decrease(ln_hrv: list[float], t: int) -> bool:
    """Whether day ``t`` is below the rolling band or in a >``TREND`` two-day downtrend.

    Parameters
    ----------
    ln_hrv : list[float]
        Morning ln-HRV series.
    t : int
        Absolute index of the current day.

    Returns
    -------
    bool
        True if the HRV decrease is significant per Kiviniemi (2007).
    """
    below = ln_hrv[t] < _rolling_ref(ln_hrv, t)
    trend = (
        t >= 2 and (ln_hrv[t - 1] - ln_hrv[t] > TREND) and (ln_hrv[t - 2] - ln_hrv[t - 1] > TREND)
    )
    return bool(below or trend)


def kiviniemi_decide(ln_hrv: list[float], presc: list[int], t: int) -> int:
    """Kiviniemi (2007) HRV-guided prescription for day ``t`` (HARD/EASY/REST).

    Increased/unchanged HRV -> HARD; a significant decrease -> EASY; a further significant
    decrease after an already-reduced (EASY) day -> REST. Caps: <=9 successive training days
    force REST, <=2 successive HARD days force EASY, <=2 successive REST days force EASY.

    Parameters
    ----------
    ln_hrv : list[float]
        Morning ln-HRV series through day ``t`` (seeded with real baseline nights).
    presc : list[int]
        Forecast-day prescriptions already issued (days ``< t`` within the horizon).
    t : int
        Absolute index of the current day in ``ln_hrv``.

    Returns
    -------
    int
        One of ``HARD``, ``EASY``, ``RESTLAB``.
    """
    prev = presc[-MAX_TRAIN:]
    if len(prev) >= MAX_TRAIN and all(x != RESTLAB for x in prev):
        return RESTLAB
    if _significant_decrease(ln_hrv, t):
        base = RESTLAB if presc and presc[-1] == EASY else EASY
    else:
        base = HARD
    if base == HARD and len(presc) >= MAX_HIGH and all(x == HARD for x in presc[-MAX_HIGH:]):
        base = EASY
    if base == RESTLAB and len(presc) >= MAX_REST and all(x == RESTLAB for x in presc[-MAX_REST:]):
        base = EASY
    return base


def forecast_hrv(
    i_f0: float,
    i_g0: float,
    r_end: float,
    u0: float,
    w_last: float,
    hrv_seed: list[float],
    p: dict[str, float],
    scaling: dict,
    rng: np.random.Generator,
) -> float:
    """One stochastic HRV-guided rollout; returns the 42nd-day ``r`` for this draw.

    Morning (causal) convention: day ``t``'s decision sees the stock through yesterday. The full
    HRV readout (stock + load-suppression + AR(1) residual + white noise) is simulated, mapped
    back to raw HRV, and its ln feeds the controller's next decision; the chosen load then advances
    the shared integrators.

    Parameters
    ----------
    i_f0, i_g0 : float
        Warm-start integrator states for this draw.
    r_end : float
        Fitted ``r`` on the last observed day (stock through yesterday for day 0).
    u0 : float
        AR(1) residual on the last observed day.
    w_last : float
        Last observed raw load (the ``load_prev`` exog for forecast day 0).
    hrv_seed : list[float]
        ln of the last ``BASE_WIN`` observed raw HRV nights, priming the baseline band.
    p : dict[str, float]
        Scalar posterior parameters for this draw.
    scaling : dict
        Panel scaling: ``load`` ``(0, load_mean)``, ``hrv`` ``(c, s)``, and ``hrv.load_prev``
        ``(c, s)``.
    rng : np.random.Generator
        Random source for the residual and noise draws.

    Returns
    -------
    float
        ``r`` on the 42nd forecast day for this draw.
    """
    a_f, a_g = np.exp(-1.0 / p["tau_F"]), np.exp(-1.0 / p["tau_G"])
    _, load_mean = scaling["load"]
    c_hrv, s_hrv = scaling["hrv"]
    c_lp, s_lp = scaling["hrv.load_prev"]

    i_f, i_g, r_prev, u, w_prev = i_f0, i_g0, r_end, u0, w_last
    ln_hrv = list(hrv_seed)
    presc: list[int] = []
    for _ in range(HORIZON):
        u = p["phi_u_hrv"] * u + p["sigma_u_hrv"] * rng.standard_normal()
        lp_std = (w_prev - c_lp) / s_lp
        y_std = (
            p["alpha_hrv"]
            + r_prev
            + p["beta_hrv"] * lp_std
            + u
            + p["sigma_hrv"] * rng.standard_normal()
        )
        y_raw = y_std * s_hrv + c_hrv
        ln_hrv.append(float(np.log(max(y_raw, 1e-6))))
        lab = kiviniemi_decide(ln_hrv, presc, len(ln_hrv) - 1)
        w = LOAD[lab]
        d = w / load_mean
        i_f = a_f * i_f + d
        i_g = a_g * i_g + d
        r_prev = p["k_F"] * i_f - p["k_G"] * i_g
        presc.append(lab)
        w_prev = w
    return r_prev


def tra_loads() -> np.ndarray:
    """Build the TRA schedule: a ``[LOW, HIGH, HIGH, LOW, HIGH, HIGH, REST]`` week over the horizon.

    Returns
    -------
    np.ndarray
        Raw-rTSS daily loads, length ``HORIZON``.
    """
    week = [LOW, HIGH, HIGH, LOW, HIGH, HIGH, REST]
    return np.array((week * ((HORIZON // 7) + 1))[:HORIZON], dtype=float)


def _selfcheck(
    drive: np.ndarray, flat: dict[str, np.ndarray], r_fit: np.ndarray, load_mean: float
) -> None:
    """Refutable checks: warm-start reconstructs the fitted ``R`` end state; Max beats Placebo.

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
    r_max = forecast_static(i_f, i_g, np.full(HORIZON, HIGH), flat, load_mean)
    r_rest = forecast_static(i_f, i_g, np.zeros(HORIZON), flat, load_mean)
    assert r_max.mean() > r_rest.mean(), "Max should beat Placebo in mean final-day r"
    print("self-check OK (warm-start exact; Max > Placebo signal)")


def win_probabilities(scores: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-routine win probability: argmax among the feasible routines, Max as the ceiling.

    The infeasible ``Max`` routine deterministically maximizes ``r_{T+42}`` (HIGH load every day),
    so it is excluded from the argmax to keep the feasible comparison informative; its reported
    value is instead the probability it exceeds every feasible routine (the ceiling, typically
    1.0). The ``FEASIBLE`` routines' probabilities form a proper argmax distribution summing to 1.

    Parameters
    ----------
    scores : dict[str, np.ndarray]
        Final-day ``r`` per routine, each ``(n_draws,)``.

    Returns
    -------
    dict[str, float]
        Win probability per routine (``FEASIBLE`` entries sum to 1; ``Max`` is the ceiling).
    """
    stacked = np.vstack([scores[k] for k in FEASIBLE])
    winners = stacked.argmax(axis=0)
    probs = {k: float(np.mean(winners == i)) for i, k in enumerate(FEASIBLE)}
    best_feasible = stacked.max(axis=0)
    probs["Max"] = float(np.mean(scores["Max"] > best_feasible))
    return probs


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
    rng = np.random.default_rng(SEED)
    panel = build_panel().standardize()
    scaling = panel.scaling
    _, load_mean = scaling["load"]
    drive = np.asarray(panel.load, dtype=float)

    flat, r_fit, u = flat_posterior()
    n_draws = flat["k_F"].size
    print(f"fit={FIT.name}  draws={n_draws}  horizon={HORIZON}d  HIGH={HIGH} LOW={LOW}")

    _selfcheck(drive, flat, r_fit, load_mean)

    i_f, i_g = warmstart(drive, flat)

    scores = {k: np.empty(n_draws) for k in ROUTINES}
    scores["Placebo"] = forecast_static(i_f, i_g, np.zeros(HORIZON), flat, load_mean)
    scores["Max"] = forecast_static(i_f, i_g, np.full(HORIZON, HIGH), flat, load_mean)
    scores["TRA"] = forecast_static(i_f, i_g, tra_loads(), flat, load_mean)

    # Observed baseline band: ln of the last BASE_WIN real HRV nights (raw units).
    hrv = panel.metrics["hrv"]
    c_hrv, s_hrv = scaling["hrv"]
    raw_hrv = hrv.values * s_hrv + c_hrv  # undo standardization, ordered by date
    hrv_seed = list(np.log(np.maximum(raw_hrv[-BASE_WIN:], 1e-6)))
    w_last = float(drive[-1] * load_mean)  # last observed raw load

    for s in range(n_draws):
        p = {k: float(flat[k][s]) for k in PARAMS}
        scores["HRV"][s] = forecast_hrv(
            float(i_f[s]),
            float(i_g[s]),
            float(r_fit[s, -1]),
            float(u[s, -1]),
            w_last,
            hrv_seed,
            p,
            scaling,
            rng,
        )

    probs = win_probabilities(scores)

    print("\n--- P(routine = best r_{T+42}); Max = infeasible ceiling, feasible rows sum to 1 ---")
    for k in ROUTINES:
        print(f"  {k:8s} {probs[k]:.3f}   mean r_42={scores[k].mean():+.3f}")

    print("\n% --- paste into \\label{tab:Table3} ---")
    print(latex_table(probs))


if __name__ == "__main__":
    main()
