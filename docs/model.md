# Model Description

The purpose of this package is to unify multiple variants of the impulse-response model under a
single, modular, Bayesian framework, and to resolve some of its statistical flaws by sharing
information across different metrics and by mitigating ill-conditioned parameter spaces with
physiologically meaningful restricted priors.

## Background

The Banister impulse-response model (and its many variants) provides insights into the
effects of training on performance through the inference of athetic performance's responses to 
training dose as well as the prediction of future performance. It suffers from two interacting challenges:
performance data are expensive and physically demanding to collect and therefore sparsely observed, 
and the parameters of interest are ill-conditioned. Without a dense, high-signal dataset from
which to learn, actionable insights from posterior parameter estimates and performance forecasts are
unlikely.

### The original model

The original [Banister 1975, Calvert 1976](https://ieeexplore.ieee.org/document/5409179) model is
defined by

$$p_t = p_0 + k_F\sum_{s=0}^{t-1} w_s e^{-(t-s)/\tau_F} - k_G\sum_{s=0}^{t-1} w_s e^{-(t-s)/\tau_G}$$

where $p_0$ is an initial performance value and $p_t$ and $w_t$ are performance and training load
at time $t$, respectively. The first and second summands above are commonly called "fitness" and
"fatigue" and are described by their gain parameters $k_F, k_G$ and decay rates $\tau_F, \tau_G$.

### Statistical flaws in the original model

[Hellard 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC1974899/) first quantified the
ill-conditioning issue. Using nine elite swimmers with an average of 13 performances across a
single training season, it was demonstrated that, even though goodness-of-fit was strong (average
$R^2 \approx 0.79$), the uncertainty in the parameter estimates was wide (the bootstrapped 
coefficient of variation was over 30% for the decay parameters and 60% for the gain parameters).
This ill-conditioning was primarily caused by correlation of the posterior estimates (0.91 for the
gain terms and 0.99 for the decay rates). In other words, there is a large set of widely varying
parameter combinations that give similar model errors. It isn't difficult to see why: fitness and
fatigue are antagonistic terms competing to explain changes in sparsely observed performance using
similar time-filters on the same training load variable.

[Vermeire 2022](https://pubmed.ncbi.nlm.nih.gov/35320776/) reproduced Hellard's result with eight
performance observations on a cycle ergometer from a single subject and visualized the model error
over a grid of parameter combinations, further demonstrating that (a) the error landscape over
$\tau_F$ and $\tau_G$ is characterized by a sharp ridge near $\tau_F = \tau_G$, and (b) the
physiologically sensible region in which incremental fitness gains are smaller than fatigue
gains, $k_F < k_G$, results in consistently lower error than the unreasonable alternative.

[Peng 2024](https://www.tandfonline.com/doi/abs/10.1080/24748668.2023.2268480) built the first
Bayesian Banister model. Importantly, they use informative prior parameterizations to provide a
prior belief about values the parameters can take based on existing results from the literature. 
Importantly, they impose an informative _prior structure_ by reparameterizing $k_G = \theta k_F$ 
and imposing the constraint $\theta > 1$; in other words, preventing the model from considering the
physiologically unreasonable region $k_F > k_G$ that Vermeire demonstrated resulted in poor model
fit. They demonstrate that the Bayesian approach resulted in better parameter estimates, though
uncertainty was still large, likely due to fitting the model to only 13 performance measurements
(with additional noise likely caused by each performance being races of different lengths, and by
imputing some missing training loads).

[Marchal 2025](https://www.nature.com/articles/s41598-025-88153-7) also take a Bayesian approach
and test weakly- versus moderately-informed Bayesian priors. However, they drop the physiological
constraint $\theta > 1$ and use performance measurements from short-track speed skaters previously
demonstrated to be low-signal (average $R^2 \approx 0.20$) via [alternative machine learning methods](https://www.nature.com/articles/s41598-022-05392-8). Furthermore, their MCMC traceplots
demonstrate a single stuck chain, indicative of a poorly tuned sampler rather than model issues.
They conclude that the impulse response model presents major dysfunctions, though their conclusion
rests on an estimate of a posterior distribution that did not converge, and on cross-validation
applied to a small set of performance measurements (roughly 35 performances from nine athletes,
of which roughly ten performances are allocated to the testing dataset).

## The BIRM Prior

The ill-conditioning issue is mitigated with a reparameterization. BIRM adopts Peng's
constraint $k_F < k_G$ to restrict the model to only search for optimal parameter
combinations in a physiologically reasonable space. To motivate similar reparameterizations for
the decay terms, we represent $k_G/k_F$ in terms of decay rates and the time it takes the impulse
response to recover $t_r$ and peak $t_p$:

$$k_G/k_F = (\tau_F/\tau_G)^{\left(\frac{1}{t_p/t_r} - 1\right)} $$

This satisfies the constraint $k_G/k_F > 1$ either when $\tau_F > \tau_G$ and $t_p > t_r$,
or when $\tau_F < \tau_G$ and $t_p < t_r$. The former is the physiologically reasonable
scenario: fitness is accumulated over longer timescales than fatigue, and the impulse response
recovers before it peaks.

In [Ceddia 2026](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2025-0077/html), the 
authors derive an admissibility condition implied by bounded infinite-horizon optimal training
policies (i.e., finite human performance):

$$ k_G \left(1 + \frac{1}{e^{1/\tau_G} - 1}\right) - \frac{k_F}{e^{1/\tau_F} - 1} > 0 $$

Rearranging to express the condition in terms of our earlier reparameterization, we see that

$$k_G/k_F > \frac{1 - e^{-1/\tau_G}}{ e^{1/\tau_F} - 1} \equiv a_c$$

It can be shown via Taylor expansion that

$$\frac{1}{\log(2 - e^{-1/\tau_G})} = \tau_G + 1 + \mathcal{O}(1/\tau_G^2)$$

and therefore admissibility $a_c$ approximately satisfies the condition $a_c > 1$ when
$\tau_F > \tau_G + 1$. BIRM therefore uses the following parameterization and priors:

$$k_G = (\phi + 1) k_F \quad \text{ and } \quad \tau_F = 1 + \tau_G + \lambda_F,$$

$$\phi \sim \text{LogNormal}(\mu_\phi, \sigma_\phi^2),$$

$$\lambda_F \sim \text{LogNormal}(\mu_{\lambda_F}, \sigma_{\lambda_F}^2)$$

which produce posterior estimates which satisfy the constraints $k_G/k_F > 1$ and
$\tau_F > \tau_G + 1$ with certainty. Note that $k_G/k_F > 1$ and the (approximately) implied
$a_c > 1$ is weaker than the admissibility condition $k_G/k_F > a_c$; we are only motivating
a prior rather than strictly imposing $k_G/k_F > a_c$ because it can bias parameter estimates from
athletes who have not yet reached peak performance.

## Multivariate Inputs and Outputs
Assimilating additional data alongside performance (e.g., heart rate variability from consumer
wearables) and controlling for performance effect modifiers (e.g., temperature and other external
stressors) can further refine parameter estimates and improve forecast performance.
To set notation, let $w_t$ be training load, $f_t$ be fitness, $g_t$ be fatigue, and
$R_t = f_t - g_t$ be the impulse response on day $t$. Multiple target variables (e.g., HRV and
speed) can be represented as transformed, noisy measurements of the underlying IR model. The
$j^{th}$ response variable $j=1,...,J$ at time $t$, $Y^{(j)}_t$, is given by

$$ Y^{(j)}_t = \alpha^{(j)} + \gamma^{(j)} l^{(j)}\left(R_t\right) + X^{(j)}_t\beta^{(j)} + u^{(j)}_t + \epsilon^{(j)}_t$$

where $\alpha^{(j)}$ is the intercept analogous to initial performance $p_0$ in Banister's formulation,
$\gamma^{(j)}$ is a regression parameter for the linear relationship between the measurement and the
transformed latent impulse response model, $l^{(j)}(\cdot)$ is a link function to represent nonlinear
relationships across measurement types, and $\beta^{(j)}$ are coefficients corresponding to exogenous
predictors $X^{(j)}$.

Residuals are composed into two terms: $\epsilon^{(j)}_t \sim N(0, \sigma^2_\epsilon)$ representing IID
noise, and $u^{(j)}_t$, a lag-1 autoregressive process orthogonalized to lie in the null space of the
intercept and the fitness and fatigue integrals and capturing variation (e.g., slow drift due to unmeasured stress
and fatigue, daily responses to sleep and nutrition, etc.) not explained by the main effects.

In this package, every term except the
impulse-response $R_t$ and the white noise $\epsilon^{(j)}_t$ is optional. Note that there are $J+1$ possible
time grids; they are "ragged" in that training loads $w_t$ and channels $Y^{(j)}_t$ exist on their
own measurement cadence. Each day may or may not have an observation from each channel, independently.

## Modifications to the impulse-response model

The original Banister 1975 IR model is defined by

$$R^{(O)}_t = k_F\sum_{s=0}^{t-1} w_s e^{-(t-s)/\tau_F} - k_G\sum_{s=0}^{t-1} w_s e^{-(t-s)/\tau_G}$$

[Busso 2003](https://www.ovid.com/jnls/acsm-msse/abstract/10.1249/01.mss.0000074465.13621.37)
modified the fatigue gain to vary over time by responding to training load:

$$R^{(V)}_t = k_F\sum_{s=0}^{t-1} w_s e^{-(t-s)/\tau_F} - \sum_{s=0}^{t-1}k_G(s) w_s e^{-(t-s)/\tau_G},$$

$$k_G(t) = \sum_{s=0}^{t-1} w_s e^{-(t-s)/\tau_v}.$$

The nonlinear fatigue gain accumulates with more recent training on timescales that are expected to be
smaller than fitness and fatigue level. We therefore impose the physiological constraint
$\tau_F > \tau_G > \tau_v$ with the reparameterization $\tau_G = \tau_v + \lambda_G$ and 
$\tau_F = 1 + \tau_G + \lambda_F$ with log-normal priors on $\lambda_G$ and $\lambda_F$.

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

$$M_F(w_s, g_s) = \frac{k_F w_s}{(1 + k_p g_s)(1+k_a w_s)}.$$

## Example: HRV and speed

$Y^{(1)}_t$ and $Y^{(2)}_t$ are HRV and speed, respectively. $X^{(1)}_t$ is the training load $w_t$ to
represent a possible daily HRV suppression effect $\beta^{(1)}_t$, and $X^{(2)}_t$ contains outdoor
temperature (to adjust for heat) and an indicator of whether each run was performed indoors on a treadmill,
giving speed two exogenous predictors. HRV is observed near daily, so we include a $u^{(1)}_t$ term. speed
is more sparsely observed; no $u$ is warranted. Both channels use the identity link
$l^{(1)}(x) = l^{(2)}(x) = x$, and we set $\gamma^{(1)} \equiv 1$ so the scale of the IR model is pinned
to the more densely observed channel HRV. The shared impulse-response model is the Ceddia
variant whose optimal training sequence is characterized by periodization and tapering. The joint 
model is therefore

$$ Y^{(1)}_t = \alpha^{(1)} + R^{(C)}_t + X^{(1)}_t\beta^{(1)} + u^{(1)}_t + \epsilon^{(1)}_t$$

$$ Y^{(2)}_t = \alpha^{(2)} + \gamma^{(2)} R^{(C)}_t + X^{(2)}_t\beta^{(2)} + \epsilon^{(2)}_t$$

You would define this model in BIRM like so:

```python
model = IRModel(
    load=Load("w"),  # the training-load driver w_t
    fitness="ceddia",
    fatigue="banister",
    metrics=(
        Metric("hrv", exog=("w",), ar1=True),  # hrv suppression + drift (u)
        Metric("speed", exog=("temp", "indoor")),  # heat + treadmill effects
    ),
)
```