# Changelog

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
