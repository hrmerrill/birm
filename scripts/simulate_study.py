r"""Simulate the Results / "Simulation study" datasets and cache them (no model fitting).

Mirrors the design in ``the_paper/journal-article.tex`` (Section: Simulation study):

- 365 days of a progressive periodized training program. A moderate-load day is followed by two
  high-load days; that three-day pattern repeats twice per week, then a rest day. The weekly
  pattern runs four weeks, then a rest week (a 5-week block). Four blocks (20 weeks), two weeks'
  rest, then the same structure again at 10% higher volume for the rest of the year.
- Daily load magnitudes carry small Gaussian noise; 5% of days are randomly zeroed (missed
  training).
- Impulse response with $k_F=0.01,\ k_G=0.02,\ \tau_F=30,\ \tau_G=5$ (fitness minus fatigue,
  using the same leaky-integrator the model fits).
- Two performance metrics are independent Normals with mean equal to the impulse response and
  variance equal to its temporal variance (SNR 1, $R^2\approx 0.5$).
- Four observation cadences (every two weeks, weekly, twice weekly, four times weekly); 30
  replicates each.

Run::

    uv run --with matplotlib python scripts/simulate_study.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib  # ty: ignore[unresolved-import]
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  # ty: ignore[unresolved-import]

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"

DAYS = 365
N_REPS = 30
PARAMS = {"k_F": 0.01, "k_G": 0.02, "tau_F": 30.0, "tau_G": 5.0}
MODERATE, HIGH = 50.0, 100.0
LOAD_NOISE_SD = 5.0  # small variability of training effort
MISS_RATE = 0.05  # fraction of days randomly zeroed (missed training)
VOLUME_BUMP = 1.10  # 10% higher volume in the second phase

# ponytail: cadences as day-of-week offsets; sample size is all that matters for the study.
CADENCES = {
    "every_two_weeks": np.arange(0, DAYS, 14),
    "weekly": np.arange(0, DAYS, 7),
    "twice_weekly": np.sort(np.concatenate([np.arange(0, DAYS, 7), np.arange(3, DAYS, 7)])),
    "four_weekly": np.array([d for d in range(DAYS) if d % 7 in (0, 2, 4, 6)]),
}


def _weekly() -> np.ndarray:
    """Return one week's base load pattern.

    Returns
    -------
    np.ndarray
        Length-7 loads: moderate, high, high, moderate, high, high, rest.
    """
    return np.array([MODERATE, HIGH, HIGH, MODERATE, HIGH, HIGH, 0.0])


def _program() -> np.ndarray:
    """Build the 365-day periodized load program (noise-free, before zeroing).

    Returns
    -------
    np.ndarray
        Dense daily load, length ``DAYS``.
    """
    week = _weekly()
    rest_week = np.zeros(7)
    block = np.concatenate([np.tile(week, 4), rest_week])  # 4 training weeks + rest week

    phase_a = np.tile(block, 4)  # 20 weeks
    gap = np.zeros(14)  # two weeks' rest
    tail = DAYS - len(phase_a) - len(gap)
    reps = tail // len(block) + 1
    phase_b = np.tile(block, reps)[:tail] * VOLUME_BUMP  # 10% higher volume, filled to year end

    return np.concatenate([phase_a, gap, phase_b])[:DAYS]


def _ar1(coef: float, forcing: np.ndarray) -> np.ndarray:
    r"""Leaky integrator $x_t = \text{coef}\,x_{t-1} + \text{forcing}_t$, matching ``birm``.

    Parameters
    ----------
    coef : float
        Decay coefficient $e^{-1/\tau}$.
    forcing : np.ndarray
        Per-day forcing, length ``T``.

    Returns
    -------
    np.ndarray
        Accumulated stock, length ``T``.
    """
    out = np.empty_like(forcing, dtype=float)
    x = 0.0
    for t, f in enumerate(forcing):
        x = coef * x + f
        out[t] = x
    return out


def _impulse(load: np.ndarray) -> np.ndarray:
    """Impulse response $r_t$ = fitness minus fatigue for a load series.

    Parameters
    ----------
    load : np.ndarray
        Dense daily training load.

    Returns
    -------
    np.ndarray
        Impulse response, length ``T``.
    """
    fitness = PARAMS["k_F"] * _ar1(np.exp(-1.0 / PARAMS["tau_F"]), load)
    fatigue = PARAMS["k_G"] * _ar1(np.exp(-1.0 / PARAMS["tau_G"]), load)
    return fitness - fatigue


def _simulate(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Simulate one replicate (load, impulse response, two latent performance series).

    Parameters
    ----------
    rng : np.random.Generator
        Per-replicate random source.

    Returns
    -------
    dict[str, np.ndarray]
        Keys ``load``, ``impulse``, ``perf1``, ``perf2``, each length ``DAYS``.
    """
    load = _program().copy()
    active = load > 0
    load[active] += rng.normal(0.0, LOAD_NOISE_SD, size=active.sum())
    load = np.clip(load, 0.0, None)
    load[rng.random(DAYS) < MISS_RATE] = 0.0  # missed training

    r = _impulse(load)
    noise_sd = np.sqrt(np.var(r))  # SNR 1 -> R^2 ~ 0.5
    perf1 = r + rng.normal(0.0, noise_sd, size=DAYS)
    perf2 = r + rng.normal(0.0, noise_sd, size=DAYS)
    return {"load": load, "impulse": r, "perf1": perf1, "perf2": perf2}


def _figure(sim: dict[str, np.ndarray], obs_idx: np.ndarray, out: Path) -> None:
    """Save the two-panel example figure (load stems below, impulse + observations above).

    Parameters
    ----------
    sim : dict[str, np.ndarray]
        One simulated dataset from [_simulate][].
    obs_idx : np.ndarray
        Observation day indices for the plotted performance channel.
    out : Path
        Output image path.
    """
    day = np.arange(DAYS)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 6), sharex=True, gridspec_kw={"hspace": 0.12}
    )

    ax_top.plot(day, sim["impulse"], color="#1f77b4", lw=1.5, label="impulse response")
    ax_top.plot(
        obs_idx,
        sim["perf1"][obs_idx],
        ls="",
        marker="o",
        ms=4.5,
        mfc="black",
        mec="white",
        mew=0.6,
        label="performance observations",
    )
    ax_top.axhline(0.0, color="grey", lw=0.6, ls="--")
    ax_top.set_ylabel("performance", fontsize=10)
    ax_top.legend(fontsize=8, loc="upper left")
    ax_top.tick_params(labelsize=8)

    ax_bot.vlines(day, 0.0, sim["load"], color="#d62728", lw=1.0)
    ax_bot.set_ylabel("training load", fontsize=10)
    ax_bot.set_xlabel("day of year", fontsize=10)
    ax_bot.set_xlim(0, DAYS)
    ax_bot.tick_params(labelsize=8)

    fig.savefig(out, dpi=300, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def main() -> None:
    """Simulate all replicates, cache arrays, and render the example figure."""
    CACHE.mkdir(exist_ok=True)
    rng = np.random.default_rng(20260101)

    load = np.empty((N_REPS, DAYS))
    impulse = np.empty((N_REPS, DAYS))
    perf1 = np.empty((N_REPS, DAYS))
    perf2 = np.empty((N_REPS, DAYS))
    for i in range(N_REPS):
        sim = _simulate(np.random.default_rng(rng.integers(2**63)))
        load[i], impulse[i] = sim["load"], sim["impulse"]
        perf1[i], perf2[i] = sim["perf1"], sim["perf2"]

    out = CACHE / "simulation_study.npz"
    arrays: dict[str, np.ndarray] = {
        "day": np.arange(DAYS),
        "load": load,
        "impulse": impulse,
        "perf1": perf1,
        "perf2": perf2,
        **{f"cadence_{name}": idx for name, idx in CADENCES.items()},
        **{f"param_{k}": np.array(v) for k, v in PARAMS.items()},
    }
    np.savez_compressed(out, **arrays)  # ty: ignore[invalid-argument-type]
    print(f"saved -> {out}  ({N_REPS} reps x {DAYS} days, {len(CADENCES)} cadences)")

    example = {k: v[0] for k, v in {"load": load, "impulse": impulse, "perf1": perf1}.items()}
    example["perf2"] = perf2[0]
    _figure(example, CADENCES["weekly"], CACHE / "simulation_example.jpg")


if __name__ == "__main__":
    main()
