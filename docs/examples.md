# Examples

## Building a `Panel`

The `Panel` is the model-ready dataset used by `birm`. It holds one contiguous daily grid along
with the observed values and their positions on that grid. Training load is required on every day
and is therefore imputed where missing (a missing day is a rest day, `0.0` load). Each metric and 
its corresponding exogenous variables keeps its own observation cadence; days can be missing per 
metric independently.

### Long form

Use this if you have one long table containing all your measurements. Each row is a `(date, metric, value)` reading.

```python
import numpy as np
import pandas as pd
from birm import Panel

df = pd.DataFrame(
    {
        "date": ["2025-01-03", "2025-01-03", "2025-01-04", "2025-01-06"],
        "metric": ["w", "hrv", "hrv", "speed"],
        "value": [210.0, 64.0, 61.0, 4.71],
        "trimp": [np.nan, 210.0, 185.0, np.nan],  # HRV-day exog
        "temp": [np.nan, np.nan, np.nan, 27.5],  # speed-day exog
        "indoor": [np.nan, np.nan, np.nan, 1.0],
    }
)

panel = Panel.from_long(
    df,
    date="date",
    metric="metric",
    value="value",
    load="w",  # which label is the training load
    exog={"hrv": ["trimp"], "speed": ["temp", "indoor"]},
)

panel.T  # 4 — days from 2025-01-03 to 2025-01-06
panel.metrics["hrv"].idx  # array([0, 1]) — grid positions of the two HRV readings
panel.metrics["speed"].exog  # {"temp": array([27.5]), "indoor": array([1.0])}
panel.densest()  # "hrv" has the most observations
```

### Wide / multi-sheet

Use this if you track loads and metrics in different tables. Each sheet has a `date` column, a
value column named after the metric, and (possibly) exogenous variables. Blank cells mark
unobserved days.

```python
sheets = {
    "w": pd.DataFrame({"date": ["2025-01-03"], "w": [210.0]}),
    "hrv": pd.DataFrame(
        {"date": ["2025-01-03", "2025-01-04"], "hrv": [64.0, 61.0], "trimp": [210.0, 185.0]}
    ),
    "speed": pd.DataFrame(
        {"date": ["2025-01-06"], "speed": [4.71], "temp": [27.5], "indoor": [1.0]}
    ),
}

panel = Panel.from_sheets(
    sheets, date="date", load="w", exog={"hrv": ["trimp"], "speed": ["temp", "indoor"]}
)
```

## Standardizing

`standardize()` z-scores each metric and its exog (storing `(center, scale)` in `panel.scaling` for
back-transform) and rescales load to average-day units so the model's gains stay `O(1)`:

```python
panel = panel.standardize()
panel.scaling["hrv"]  # (center, scale) used to un-standardize HRV forecasts later
```

The stored `scaling` is what forecast and simulation reuse to put future load on the model's scale 
and to back-transform predictions.

## Specifying a model

Once you have a `Panel`, an `IRModel` declares *which* impulse-response model to fit. The IR
model is composed of antagonistic terms `R_t = f_t - g_t`, with `f_t` and `g_t` commonly
called `fitness` and `fatigue`. You choose which form each term can take.

The IR model is shared across metrics, and each metric gets its own exogenous predictors and, 
optionally, an AR1 term orthogonalized against `R_t` and the predictors so that it only explains 
variability not already explained by the drivers of primary importance.

```python
from birm import IRModel, Load, Metric

model = IRModel(
    load=Load("w"),  # the training-load driver w_t
    fitness="banister",
    fatigue="banister",  # the R_t core
    metrics=(
        Metric("hrv", exog=("w",), ar1=True),  # beta^(hrv) suppression + u^(hrv) drift
        Metric("speed", exog=("temp", "indoor")),  # heat + treadmill adjustments; sparse, no u
    ),
)
```

### Choosing the core

`fitness` × `fatigue` select the shared `R_t`. Setting the Ceddia gains `k_p = k_a = 0` recovers
the Banister linear model.

| Argument | Options | Meaning |
|---|---|---|
| `fitness` | `banister` | original linear fitness gain |
| | `ceddia_periodize` | periodization via fatigue feedback (`k_p`) |
| | `ceddia_taper` | tapering via diminishing dose returns (`k_a`) |
| | `ceddia` | both (`k_p` and `k_a`) |
| `fatigue` | `banister` | original linear fatigue |
| | `busso` | variable-dose, time-varying `k_G(s)` (`tau_v`) |
| | `none` | fitness-only model |

### Describing each metric

Each `Metric` is a noisy, optionally transformed readout of the shared stock. Every term but the
impulse response and white noise is optional; drop one by leaving it off.

```python
Metric(
    "speed",
    link="identity",  # l^(j): "identity" | "log" | "logit"
    exog=("temp", "indoor"),  # X^(j) columns -> beta^(j)
    ar1=True,  # include the orthogonalized slow-drift term u^(j)
    intercept=True,  # include the baseline alpha^(j)
    gamma=None,  # fix the scale; None -> densest metric auto-pinned to gamma == 1
)
```

- `link` applies a nonlinear transform between the latent stock and the measurement.
- `exog` names columns that this metric carries in the panel (`panel.metrics[name].exog`); a
  reference to a missing column is an error at fit time.
- `ar1` adds the lag-1 process `u^(j)`, orthogonalized against `{1, impulse, exog}` to capture slow
  unexplained drift — use it for densely observed metrics; omit it for sparse ones.
- `intercept` adds the baseline `alpha^(j)`; drop it only when the baseline is already removed
  (e.g. a differenced metric) or owned by another term.
- `gamma` pins the per-metric scale. Leave it `None` and the densest metric is auto-pinned to
  `gamma == 1`, tying the latent scale to the best-observed channel.

### Fitting

`.fit()` fits the model by drawing posterior samples of the parameters:

```python
idata = model.fit(panel, sampler=dict(draws=1000, tune=2000, seed=0))
```

!!! warning "Under construction"
    `.fit()` currently performs the full spec/panel validation and then raises
    `NotImplementedError` — the latent-stock and observation layers are still being wired up. The
    validation is live, so a misnamed metric or an `exog` column the panel does not carry already
    raises a clear `ValueError`.

### Overriding priors

Pass `Prior` objects keyed by parameter name to override defaults (e.g. the Ceddia `k_p`/`k_a` or
Busso `tau_v`):

```python
from birm import Prior

model = IRModel(
    load=Load("w"),
    fitness="ceddia_periodize",
    fatigue="banister",
    metrics=(Metric("hrv"),),
    priors={"k_p": Prior("HalfNormal", {"sigma": 0.5})},
)
```
