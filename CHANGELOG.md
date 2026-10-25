# Changelog

## 0.1.3

Simplifications:

- `IRModel(fitness=..., fatigue=...)` is replaced by a single
  `core` in `{"banister", "busso", "ceddia", "fitness-only"}`
- the `ceddia` core implements Ceddia's recommended form (2026; Fig. 10 and Section 6): a
  power-law diminishing-returns fitness gain (`k_D`) and fitness/fatigue decay time constants
  that grow linearly with session load (`rho_F`, `rho_G`)
- admissibility is removed: the `admissible` parameterization, the `admiss` posterior
  variable and `birm.core.admissibility` are gone. Users can recompute the metric from posterior
  draws if needed

## 0.1.2

- bug fix: Ceddia periodization gate used fatigue lagged twice (ignoring yesterday's load); it now
  uses the causal fatigue stock entering the day

## 0.1.1

- bug fix: remove same-day integration of workout load
- estimate response-specific offsets

## 0.1.0

Initial release:

- unconstrained, constrained, and admissible prior structures on the original Banister model
- Busso fatigue and Ceddia periodization/tapering fitness terms
- Model-ready data preparation tooling
