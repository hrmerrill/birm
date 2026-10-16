# BIRM

## About

This package implements a **B**ayesian **I**mpulse-**R**esponse **M**odel for athletic
performance. The default prior structure is physiologically informed and resolves
inference issues common in the literature, and the model allows for multivariate inputs and
outputs so that parameters can be informed by multiple metrics of performance.

## Example Usage

Suppose you want to estimate the impacts of daily training load on running speed performance.
You have temperature measurements for each outdoor run and an indicator for each indoor
treadmill run, and you want the model to control for potential affects these exogenous variables
have on running speed. You also have daily-ish heart rate variability data, which you have noticed
correlates with running speed, and which you notice is commonly suppressed the day after intense
training.

Suppose you track loads and metrics in different tables. Each sheet has a `date` column, a
value column named after the metric, and the exogenous variables. Blank cells mark unobserved days.
You prepare your data into a `Panel`:

```python
import pandas as pd

# has columns 'w' (training load) and 'date' (YYYY-MM-DD)
training_load_df = pd.read_csv("/path/to/training_load.csv")

# has columns 'hrv', 'w' (missing on rest days) and 'date'
hrv_df = pd.read_csv("/path/to/hrv.csv")

# has columns 'speed', 'temp', 'indoor' (1 or 0) and 'date'
speed_df = pd.read_csv("/path/to/speed.csv")

sheets = {"w": training_load_df, "hrv": hrv_df, "speed": speed_df}

panel = Panel.from_sheets(
    sheets,
    date="date",
    load="w",
    exog={"hrv": ["w"], "speed": ["temp", "indoor"]},
).standardize()
```

To fit the model in which the classical Banister parameters are jointly estimated from both
HRV and speed controlling for their corresponding exogenous variables:

```python
model = IRModel(
    load=Load("w"),  # the training-load driver
    fitness="banister",
    fatigue="banister",
    metrics=(
        Metric("hrv", exog=("w",)),  # hrv suppression
        Metric("speed", exog=("temp", "indoor")),  # heat + treadmill effects
    ),
)
idata = model.fit(panel, sampler=dict(draws=1000, tune=2000, seed=0))
```

`idata` is an ArviZ `InferenceData`. Summarize the fit and extract the shared stock `R_t`:

```python
import arviz as az

az.summary(idata, var_names=["k_F", "k_G", "tau_F", "tau_G"], ci_prob=0.95)  # mean, 95% CI, r_hat
R_mean = idata.posterior["R"].mean(("chain", "draw"))  # posterior-mean R_t trajectory
idata.to_netcdf("fit.nc")  # cache for later reuse
```

---

## Read More

See [Model](model.md) for a more detailed description of the model, [Examples](examples.md) for
more examples, and [API Reference](api.md) for full documentation.