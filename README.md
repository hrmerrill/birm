# `birm`: Bayesian Impulse-Response Modeling

<p align="center">
  <img src="https://img.shields.io/pypi/v/birm?style=flat-square&color=1a5&label=pypi" alt="pypi">
  <img src="https://github.com/hrmerrill/birm/actions/workflows/ci.yml/badge.svg?style=flat-square" alt="CI">
  <img src="https://raw.githubusercontent.com/hrmerrill/birm/badges/coverage.svg" alt="coverage">
  <img src="https://raw.githubusercontent.com/hrmerrill/birm/badges/interrogate_badge.svg" alt="docstring coverage">
</p>

## About

This package implements a **B**ayesian **I**mpulse-**R**esponse **M**odel (BIRM) for athletic
performance. The default prior structure is physiologically informed and resolves
inference issues common in the literature, and the model allows for multivariate inputs and
outputs so that parameters can be informed by multiple metrics of performance adjusted for
their external influencers.

## Preparing data

The `Panel` is the model-ready dataset used by `birm` to align loads, metrics and exogenous
predictors. Build a `Panel` from a single long-form table:

```python
import pandas as pd
from birm import Panel

df = pd.DataFrame(
    {
        "date": ["2025-01-03", "2025-01-03", "2025-01-04", "2025-01-06"],
        "metric": ["w", "hrv", "hrv", "speed"],
        "value": [210.0, 64.0, 61.0, 4.71],
    }
)

panel = Panel.from_long(df, date="date", metric="metric", value="value", load="w").standardize()
panel.T  # 4 days on the shared daily grid
panel.densest()  # 'hrv' has the most observations
```

See the [docs](https://hrmerrill.github.io/birm/examples/) for more ways to build the `Panel` from
your data.

## Specifying a model

An `IRModel` is the declarative spec of the shared-stock impulse-response model. You define it by
picking a `fitness` × `fatigue` core and attach one `Metric` per response series, then fit it with
`.fit(panel)`.

```python
from birm import IRModel, Load, Metric

model = IRModel(
    load=Load("w"),  # training-load driver w_t
    fitness="banister",
    fatigue="banister",  # selects the impulse-response core R_t
    metrics=(
        Metric("hrv", exog=("w",), ar1=True),  # HRV-suppression beta + slow drift u
        Metric("speed", exog=("temp", "indoor")),  # heat + treadmill adjustments
    ),
)
idata = model.fit(panel, sampler=dict(draws=1000, tune=2000, seed=0))
```

(TODO: Examples of what to do with `idata`.)

See the [docs](https://hrmerrill.github.io/birm/examples/) for the full model reference.

## For Developers

### Setup

```bash
uv sync  # install package + dev dependencies into .venv
```

### Development workflow

All changes should be made on a branch and merged to `main` via pull request. All changes require
accompanying unit tests in `tests/` and documentation in `docs/`.

```bash
git checkout -b feat/my-feature   # or fix/, docs/, refactor/, etc.

# ...you make your changes, and then

make format      # auto-fix formatting and lint violations
make check       # lint, type-check, and verify docstring coverage
make test        # run tests with coverage (90% minimum)
make docs-serve  # preview docs locally at http://127.0.0.1:8000

git push -u origin feat/my-feature
# open a pull request on GitHub
```

CI runs `make check` and `make test` automatically on every push and pull request. A PR cannot be merged if CI fails.

### What triggers what

| Action | CI checks | Docs deployed | Package published |
|---|---|---|---|
| Push to any branch / open PR | ✓ | | |
| Merge to `main` | ✓ | ✓ | |
| Push a `v*` tag | | | ✓ |

**Docs-only change** (e.g. fix a typo in `docs/` or a docstring): open a PR and merge to `main`. Docs redeploy automatically, no tag needed.

**Code-only change** (e.g. bug fix): merge to `main`, then tag when ready to publish (see below). Docs will also redeploy on merge, reflecting any updated docstrings.

### Publishing a new package version

1. Bump the version in `pyproject.toml`:
   ```bash
   # choose one:
   make bump-patch   # 0.1.0 → 0.1.1  (bug fixes)
   make bump-minor   # 0.1.0 → 0.2.0  (new features)
   make bump-major   # 0.1.0 → 1.0.0  (breaking changes)
   ```
2. Commit, tag, and push:
   ```bash
   git add pyproject.toml
   git commit -m "chore: bump version to v0.x.x"
   git tag v0.x.x
   git push && git push --tags
   ```

Pushing the tag triggers the publish workflow, which runs the test suite and publishes the package to PyPI. Check that the release appeared:

- Package: https://pypi.org/project/birm/
- Docs: https://hrmerrill.github.io/birm/