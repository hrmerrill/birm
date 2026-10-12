---
name: diagnostics
description: >-
  Convergence, identifiability, and ill-conditioning diagnostics for a fitted birm
  impulse-response model (Banister / Ceddia / Busso shared-stock). Use when a user fits a
  birm.IRModel, asks whether it converged, why a gain or time constant is uncertain, whether a
  Ceddia/Busso extension is supported, or which fitness x fatigue cell to keep. Knows the
  birm-specific failure modes: the gain-decay ridge, prior-driven tau, single stuck chain in the
  tau tail, and the fragile ceddia+busso cell. (Scaffolded; TODO: revisit this later)
---

# birm Impulse-Response Model Diagnostics

Use this skill after fitting a `birm.IRModel`. `IRModel.fit(panel)` returns an
`arviz.InferenceData` with the posterior, `sample_stats`, and one `log_likelihood` group per
`Metric`. All checks below run on that object with `arviz` (already installed via PyMC).

```python
import arviz as az
from birm import IRModel, Load, Metric

spec = IRModel(
    load=Load("trimp"),
    fitness="banister",          # banister | ceddia_periodize | ceddia_taper | ceddia
    fatigue="banister",          # banister | busso | none
    metrics=(Metric("hrv", ar1=True), Metric("vo2max", link="log")),
)
idata = spec.fit(panel, sampler={"draws": 1000, "tune": 2500, "chains": 4, "seed": 0})
```

The latent stock is `R_t = f_t - g_t`. Its parameters are what you diagnose:

| Parameter | Meaning | Core |
|---|---|---|
| `k_F`, `tau_F` | fitness gain, decay (days) | all |
| `k_G`, `tau_G` | fatigue gain, decay (days) | banister/busso fatigue |
| `r` (`k_G = r*k_F`), `lambda_f` (`tau_F = 1 + tau_G + lambda_f`) | identifiability reparam | reparameterized fit |
| `k_p` | periodization strength (gain fades with lagged fatigue) | `ceddia_periodize`, `ceddia` |
| `k_a` | taper strength (within-session diminishing returns) | `ceddia_taper`, `ceddia` |
| `tau_v` | Busso third (fastest) timescale, `tau_v < tau_G` | `busso` |
| `alpha_*`, `beta_*`, `u_*`, `gamma_*` | per-metric intercept, exog slopes, AR(1), scale | observation layer |

## Step 1 — Convergence (R-hat, ESS), but read it per chain

```python
az.summary(idata, var_names=["k_F", "tau_F", "k_G", "tau_G"], round_to=3)
```

- **R-hat < 1.01** and **ESS (bulk and tail) > 400 per parameter.** ESS < 400 makes tail/interval
  estimates unreliable.
- **Do not act on pooled R-hat alone for `tau_F`/`tau_G`.** The time constants have a long, weakly
  identified prior tail. A pooled `R-hat ~ 1.5` on `tau_F` with **0 divergences** is usually **one
  chain that ran away up the flat-tau tail**, not a multimodal posterior. Confirm before warning:

```python
# Per-chain posterior means expose a single runaway chain.
idata.posterior["tau_F"].mean(dim="draw").to_series()
```

  If 3/4 chains agree and one sits at a much larger `tau_F`, the issue is the stuck chain, not the
  model: tighten/bound the `tau_F` prior (pass a `Prior` override in the spec) or drop the runaway
  chain and re-check. Only call it multimodal if multiple chains disagree **and** divergences or
  distinct ESS clusters back it up.

## Step 2 — Divergences

```python
int(idata.sample_stats["diverging"].sum())
```

Non-zero divergences invalidate the fit. In birm they concentrate on the **gain-decay ridge**
(Step 3) and in the fragile `ceddia` + `busso` cell (Step 5). Raise `target_accept`,
then tighten the ridge priors or reparameterize (`r`, `lambda_f`) rather than just sampling harder.

## Step 3 — The gain-decay ridge (the defining birm pathology)

The impulse `k * exp(-t/tau)` is nearly invariant to trading gain against decay, so `(k_F, tau_F)`
and `(k_G, tau_G)` sit on a curved ridge. A single measurement of a single athlete's record rarely 
separates them.

```python
import numpy as np

az.plot_pair(idata, var_names=["k_F", "tau_F"], kind="kde", marginals=True)
post = idata.posterior
kf = post["k_F"].stack(s=("chain", "draw")).values
tf = post["tau_F"].stack(s=("chain", "draw")).values
print("corr(k_F, tau_F) =", np.corrcoef(kf, tf)[0, 1])
```

Interpretation:

- A tight banana / `|corr| > ~0.7` means `tau_F` is **prior-driven, not measured**. Report it that
  way: quote the gain with its decay held at the prior, and do **not** present `tau_F` as an
  estimate of recovery time. `k_F`'s width here is an **input-contrast** limit — it needs deliberate
  overload/taper blocks in the load series, not a reparameterization.
- Repeat for `(k_G, tau_G)`. Check the reparam actually helped: `r = k_G/k_F` and `lambda_f` should
  be better-mixed (higher ESS) than raw `k_G`, `tau_F`.
- Physiological floor: confirm `P(k_G > k_F) ~ 1` (fatigue gain above fitness gain). If the fit used
  the `k_G = r*k_F` ratio, check `P(r > 1)`.

## Step 4 — Are the nonlinear (Ceddia/Busso) terms identified?

The extra terms are often unsupported on steady self-regulated data and collapse to Banister.
Compare each extension parameter's posterior to its prior:

```python
az.plot_forest(idata, var_names=["k_p", "k_a", "tau_v"], filter_vars="like", combined=True)
```

- `k_p`, `k_a` → **0** (posterior ~ prior, interval includes 0): the periodization/taper feedback is
  not identified. Prefer the plain `banister` fitness core (parsimony); recommend an overload/taper
  block to identify it rather than keeping an unidentified parameter.
- Busso `tau_v` pinned only by its prior / CI touching 0 on `k_G`: the time-varying gain reverts to
  constant-gain Banister. Keep `fatigue="banister"`.

## Step 5 — Fragile cell guard: `ceddia` + `busso`

`fitness="ceddia"` with `fatigue="busso"` drives **two** suppression channels (`k_p` shrinks fitness
gain with accumulated fatigue; Busso grows fatigue gain with accumulated load) from the same load
history, plus three time constants `tau_v < tau_G < tau_F`. Expect ridges between `k_p` and `tau_v`,
divergences, and low ESS. Check `az.plot_pair(idata, var_names=["k_p", "tau_v"])`; if it is a ridge,
demote this cell by LOO/convergence rather than trusting either parameter.

## Step 6 — Per-metric fit and the fast-latent contamination check

With multiple metrics, compare cells by PSIS-LOO **per metric** and watch Pareto-k:

```python
az.loo(idata, var_name="hrv")      # one log_likelihood group per Metric
az.loo(idata, var_name="vo2max")
```

- Pick the cell only when it wins (or ties within ~2 SE) on **every** channel; a term that helps one
  metric but degrades another (classically HRV) is not kept.
- **Bad Pareto-k concentrates on the fast AR(1) metric** (the `ar1=True` autonomic channel). If a
  simplification (e.g. dropping a latent residual) raises that channel's `frac_bad_k`, it has pushed
  slow structure onto the fast latent — revert. The fast AR(1) residual "earns its keep" by keeping
  that latent clean.

## Report

State, in order: converged? (R-hat/ESS, **per-chain** if `tau` is high); divergences; whether `k_F`
/`k_G` are pinned or ride the decay ridge (so which time constants are prior-driven); whether any
Ceddia/Busso term is identified or collapses to Banister; and the per-channel LOO ranking. Warn
loudly before trusting a recovery/peak time read off a `tau` that Step 3 showed is prior-driven.
