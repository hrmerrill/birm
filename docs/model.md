# Model Description

The purpose of this package is to unify multiple variants of the impulse-response model under a
single, modular, Bayesian framework, and to resolve some of its statistical flaws by sharing
information across different metrics and by mitigating ill-conditioned parameter spaces with
physiologically meaningful restricted priors. To set notation, let $w_t$ be training load, $f_t$
be  fitness, $g_t$ be fatigue, and $R_t = f_t - g_t$ be the impulse response on day $t$.

## Multivariate Inputs and Outputs

Multiple target variables (e.g., HRV and speed) can be represented as transformed, noisy 
measurements of the underlying IR model. The $j^{th}$ response variable $j=1,...,J$ at time
$t$, $Y^{(j)}_t$, is given by

$$ Y^{(j)}_t = \alpha^{(j)} + \gamma^{(j)} l^{(j)}\left(R^{(\cdot)}_t\right) + X^{(j)}_t\beta^{(j)} + u^{(j)}_t + \epsilon^{(j)}_t$$

where $\alpha^{(j)}$ is the intercept analogous to initial performance $p_0$ in Banister's formulation,
$\gamma^{(j)}$ is a regression parameter for the linear relationship the measurement and the transformed
latent impulse response model, $l^{(j)}(\cdot)$ is a link function to represent nonlinear relationships
across measurement types, and $\beta^{(j)}$ are coefficients corresponding to exogenous predictors
$X^{(j)}$.

There are two types of error: $\epsilon^{(j)}_t \sim N(0, \sigma^2_\epsilon)$ representing IID noise,
and $u^{(j)}_t$, a lag-1 autoregressive process orthogonalized to lie in the null space of the workout
load and exogenous predictors and capturing variation (e.g., slow drift due to unmeasured stress and
fatigue, daily responses to sleep and nutrition, etc.) not explained by them.

Every term except the
impulse-response $R_t$ and the white noise $\epsilon^{(j)}_t$ is optional. Note that there are $J+1$ possible
time grids; they are "ragged" in that training loads $w_t$ and channels $Y^{(j)}_t$ exist on their
own measurement cadence. Each day may or may not have an observation from each channel, independently.

## Impulse-response models

The original [Banister 1975, Calvert 1976](https://ieeexplore.ieee.org/document/5409179) IR model is
defined by

$$R^{(O)}_t = k_F\sum_{s=0}^{t-1} w_s e^{-(t-s)/\tau_F} - k_G\sum_{s=0}^{t-1} w_s e^{-(t-s)/\tau_G}$$


[Busso 2003](https://www.ovid.com/jnls/acsm-msse/abstract/10.1249/01.mss.0000074465.13621.37)
modified the fatigue gain to vary over time by responding to training load:

$$R^{(V)}_t = k_F\sum_{s=0}^{t-1} w_s e^{-(t-s)/\tau_F} - \sum_{s=0}^{t-1}k_G(s) w_s e^{-(t-s)/\tau_G},$$

$$k_G(t)\sum_{s=0}^{t-1} w_s e^{-(t-s)/\tau_v}$$

[Ceddia 2026](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2025-0077/html) studied
the optimal training routines implied by Banister and Busso's models and devised their own
modifications that induce periodization and tapering:

$$R^{(C)}_t = \sum_{s=0}^{t-1}M_F(w_s, g_s)e^{-(t-s)/\tau_F} - k_G\sum_{s=0}^{t-1}w_s\,e^{-(t-s)/\tau_G}$$

All nonlinearity lives in the fitness-gain function $M_F(w_s, g_s)$; Ceddia's Eq. (17) leaves the fatigue
impulse linear in $w_s$ (he explicitly declines to modify the fatigue response, as that "would amount to
redefining the training impulse itself"). Setting $k_p = k_a = 0$ recovers $M_F = k_F w_s$, the Banister
linear model.

Periodization is induced by fatigue feedback: diminishing gain as fatigue accumulates (cf. Ceddia Eq. (28))

$$M_F(w_s, g_s) = \frac{k_F w_s}{1 + k_p g_s}, $$

and tapering is induced by diminishing returns in the impulse (cf. Ceddia Eqs. (19)–(21))

$$M_F(w_s, g_s) = \frac{k_F w_s}{1+k_a w_s}$$

and both by 

$$M_F(w_s, g_s) = \frac{k_F w_s}{(1 + k_p g_s)(1+k_a w_s)}$$

## Concrete example: HRV and speed

$Y^{(1)}_t$ and $Y^{(2)}_t$ are HRV and speed, respectively. $X^{(1)}_t$ contains the training load to
potentially capture a daily HRV suppression effect $\beta^{(1)}_t$, and $X^{(2)}_t$ contains outdoor
temperature to adjust for heat and an indicator of whether each run was performed indoors on a treadmill,
giving speed two exogenous predictors. HRV is observed near daily, so we include a $u^{(1)}_t$ term. speed
is more sparsely observed; no $u$ is warranted. Both channels use the identity link
$l^{(1)}(x) = l^{(2)}(x) = x$, and we set $\gamma^{(1)} \equiv 1$ so the scale of the IR model is pinned
to the more densely observed channel HRV. The shared impulse-response model is the Ceddia
variant whose optimal training sequence is characterized by periodization and tapering. The joint 
model is therefore

$$ Y^{(1)}_t = \alpha^{(1)} + R^{(C)}_t + X^{(1)}_t\beta^{(1)} + u^{(1)}_t + \epsilon^{(1)}_t$$

$$ Y^{(2)}_t = \alpha^{(2)} + \gamma^{(2)} R^{(C)}_t + X^{(2)}_t\beta^{(2)} + \epsilon^{(2)}_t$$

You would define this model in this package with

```python
model = IRModel(
    load=Load("w"),  # the training-load driver w_t
    fitness="ceddia",
    fatigue="banister",
    metrics=(
        Metric("hrv", exog=("w",), ar1=True),  # beta^(hrv) suppression + u^(hrv) drift
        Metric("speed", exog=("temp", "indoor")),  # heat + treadmill effects
    ),
)
```