# Examples

## Building a `Panel`

The `Panel` is the model-ready dataset used by `birm`. It holds one contiguous daily grid along
with the observed values and their positions on that grid. Training load is required on every day
and is therefore imputed (a missing day is a rest day, `0.0` load). Each metric and its
corresponding exogenous variables keeps its own observation cadence; days can be missing per metric
independently.

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

panel.T  # 4 — days from 2025-01-03 .. 2025-01-06
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
