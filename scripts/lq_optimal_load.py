"""Posterior LQ optimal-control of the fitted runner IR model (paper Section 4.1).

Treats the Banister model as the linear state system ``X_{t+1} = A X_t + B u_t``,
``R_t = C X_t`` with state ``X = [f, g]`` (fitness, fatigue), input ``u = w`` (load) and

    A = diag(exp(-1/tau_F), exp(-1/tau_G)),  B = [k_F, k_G]^T,  C = [1, -1].

For each posterior draw it solves the discrete-time LQ regulator (DARE) that tracks a step
performance target ``R*`` while penalising load, giving the paper's feedback law
``u = -F X + v``. Propagating the *posterior* through the Riccati solve turns the paper's single
deterministic plan into a **distribution over the optimal load schedule** and over the
fitness/fatigue cost of hitting the target.

Caveat (paper, Section 3.2): this runner is not an elite athlete, so the identified IR is
near-monotonic (fitness time constant >> fatigue, fitness steady gain >> fatigue steady gain).
Fatigue barely constrains the optimum, so the "cost" of an aggressive plan is low here -- the LQ
machinery has much more to offer an athlete whose fatigue genuinely limits training.

Run::

    uv run --with netCDF4 --with scipy --with matplotlib python scripts/lq_optimal_load.py
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
from scipy.linalg import solve_discrete_are  # noqa: E402

from birm import Panel  # noqa: E402

_figstyle.apply()

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
NC = CACHE / "runner_application_full.nc"
FIG = ROOT / "the_paper" / "runner_lq_optimal_load.jpg"

TARGET = 1.0  # step target R* in standardized HRV units (+1 SD of overnight HRV)
Q_OUT = 1.0  # tracking weight q on output error  (ponytail: aggressiveness knob)
R_IN = 1.0  # load weight r on control effort    (ponytail: raise r for a gentler ramp)
HORIZON = 42  # days to simulate the closed loop


def _ci(x: np.ndarray) -> tuple[float, float, float]:
    """Return ``(mean, 2.5th, 97.5th)`` percentiles of a flat draw array."""
    return float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def _lq_plan(a_F: float, a_G: float, k_F: float, k_G: float) -> tuple[np.ndarray, ...] | None:
    """Solve the step-tracking LQ regulator for one draw and roll out the closed loop.

    Parameters
    ----------
    a_F, a_G : float
        Fitness/fatigue decay coefficients ``exp(-1/tau)``.
    k_F, k_G : float
        Fitness/fatigue gains.

    Returns
    -------
    tuple of np.ndarray or None
        ``(u, f, g, R)`` horizon-length trajectories, or ``None`` if the draw is not
        stabilisable / the Riccati solve fails.
    """
    A = np.array([[a_F, 0.0], [0.0, a_G]])
    B = np.array([[k_F], [k_G]])
    C = np.array([[1.0, -1.0]])
    Q = C.T @ C * Q_OUT
    Rm = np.array([[R_IN]])
    try:
        P = solve_discrete_are(A, B, Q, Rm)
    except Exception:
        return None
    F = np.linalg.solve(Rm + B.T @ P @ B, B.T @ P @ A)  # (1, 2)
    A_cl = A - B @ F
    if np.max(np.abs(np.linalg.eigvals(A_cl))) >= 1.0:  # unstable closed loop -> drop
        return None
    dc = (C @ np.linalg.solve(np.eye(2) - A_cl, B)).item()  # closed-loop static gain
    if abs(dc) < 1e-9:
        return None
    v = TARGET / dc  # feedforward so steady-state output hits the target

    x = np.zeros((2, 1))
    u = np.empty(HORIZON)
    traj = np.empty((HORIZON, 2))
    for t in range(HORIZON):
        u_t = (-F @ x).item() + v
        u[t] = u_t
        x = A_cl @ x + B * v
        traj[t] = x[:, 0]
    R = traj[:, 0] - traj[:, 1]
    return u, traj[:, 0], traj[:, 1], R


def main() -> None:
    """Run the posterior LQ sweep, print cost CIs, and render the plan figure."""
    post = az.from_netcdf(NC).posterior

    # HRV scale (ms); the IR terms f, g, r read into HRV with unit gain, so multiplying by this
    # back-transforms the standardized stocks to raw HRV milliseconds.
    long = pd.read_csv(CACHE / "model_ready_long.csv", parse_dates=["date"])
    panel = Panel.from_long(
        long,
        date="date",
        metric="metric",
        value="value",
        load="w",
        exog={"hrv": ["load_prev"], "speed": ["wetbulb"]},
    ).standardize()
    _, s_hrv = panel.scaling["hrv"]

    def flat(name: str) -> np.ndarray:
        """Stack chain/draw into one axis, dropping trailing size-1 dims."""
        return np.asarray(post[name]).reshape(-1, *post[name].shape[2:]).squeeze()

    a_F = np.exp(-1.0 / flat("tau_F"))
    a_G = np.exp(-1.0 / flat("tau_G"))
    k_F, k_G = flat("k_F"), flat("k_G")
    n = a_F.shape[0]

    U, Ff, Gg, Rr = [], [], [], []
    for i in range(n):
        out = _lq_plan(float(a_F[i]), float(a_G[i]), float(k_F[i]), float(k_G[i]))
        if out is None:
            continue
        U.append(out[0])
        Ff.append(out[1])
        Gg.append(out[2])
        Rr.append(out[3])
    U = np.array(U, dtype=float)
    Ff = np.array(Ff, dtype=float) * s_hrv  # fitness stock in HRV ms
    Gg = np.array(Gg, dtype=float) * s_hrv  # fatigue stock in HRV ms
    Rr = np.array(Rr, dtype=float)  # kept standardized for the target self-check
    kept = U.shape[0]
    print(f"draws: {kept}/{n} solved ({n - kept} dropped as non-stabilisable/unstable)")

    # The regulator must actually reach the target in steady state: this is the solver's self-check.
    assert np.allclose(np.median(Rr, axis=0)[-1], TARGET, atol=0.05), "LQ did not reach target"

    u_ss = U[:, -1]  # steady-state daily load to hold the target
    cum = U.sum(axis=1)  # total load delivered over the horizon
    peak_g = Gg.max(axis=1)  # peak fatigue incurred
    ss_g = Gg[:, -1]  # steady-state fatigue
    ss_f = Ff[:, -1]  # steady-state fitness
    frac_g = ss_g / ss_f  # fatigue-to-fitness ratio at the held target

    print(
        f"\n=== LQ cost of holding R* = {TARGET:+.1f} SD = {TARGET * s_hrv:.1f} ms "
        f"(q={Q_OUT}, r={R_IN}), 95% CI ==="
    )
    for label, v in [
        ("steady daily load u_ss", u_ss),
        ("total load over horizon", cum),
        ("peak fatigue g (ms)", peak_g),
        ("steady fatigue g_ss (ms)", ss_g),
        ("steady fitness f_ss (ms)", ss_f),
        ("fatigue/fitness ratio", frac_g),
    ]:
        m, lo, hi = _ci(v)
        print(f"{label:24s} {m:8.3f} ({lo:7.3f}, {hi:7.3f})")

    # --- figure: optimal load schedule (top) + fitness/fatigue cost (bottom) -----
    days = np.arange(HORIZON)
    lo_u, mid_u, hi_u = np.percentile(U, [2.5, 50, 97.5], axis=0)
    lo_f, mid_f, hi_f = np.percentile(Ff, [2.5, 50, 97.5], axis=0)
    lo_g, mid_g, hi_g = np.percentile(Gg, [2.5, 50, 97.5], axis=0)

    fig, (ax_u, ax_fg) = plt.subplots(2, 1, sharex=True, figsize=_figstyle.size(7.0, 4.6))

    ax_u.fill_between(days, lo_u, hi_u, color="0.6", alpha=0.35, linewidth=0, zorder=2)
    ax_u.plot(days, mid_u, color="black", linewidth=1.2, zorder=3)
    ax_u.set_ylabel("Optimal load $w_t$")
    ax_u.spines[["top", "right"]].set_visible(False)

    ax_fg.axhline(0.0, color="0.7", linewidth=0.8, zorder=1)
    ax_fg.fill_between(days, lo_f, hi_f, color="0.6", alpha=0.35, linewidth=0, zorder=2)
    ax_fg.fill_between(days, lo_g, hi_g, color="0.6", alpha=0.35, linewidth=0, zorder=2)
    ax_fg.plot(days, mid_f, color="black", linewidth=1.2, zorder=4, label="Fitness")
    ax_fg.plot(days, mid_g, color="black", linewidth=1.2, linestyle="--", zorder=3, label="Fatigue")
    ax_fg.set_ylabel("IR stock (ms)")
    ax_fg.set_xlabel("Day of plan")
    ax_fg.legend(frameon=False, loc="upper left")
    ax_fg.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG, dpi=600, format="jpg")
    print(f"\nsaved figure -> {FIG}")


if __name__ == "__main__":
    main()
