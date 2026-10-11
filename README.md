# `birm`: Bayesian Impulse-Response Modeling

# UNDER CONSTRUCTION

None of the below is true yet.

<p align="center">
  <img src="https://img.shields.io/pypi/v/birm?style=flat-square&color=111111&label=pypi" alt="pypi">
  <img src="https://github.com/hrmerrill/birm/actions/workflows/ci.yml/badge.svg?style=flat-square" alt="CI">
  <img src="https://raw.githubusercontent.com/hrmerrill/birm/main/.github/badges/coverage.svg" alt="coverage">
  <img src="https://raw.githubusercontent.com/hrmerrill/birm/main/.github/badges/interrogate_badge.svg" alt="docstring coverage">
</p>

## About

TODO

## Example

TODO

## For Developers

### Setup

```bash
uv sync  # install package + dev dependencies into .venv
```

### Development workflow

All changes should be made on a branch and merged to `main` via pull request.

```bash
git checkout -b feat/my-feature   # or fix/, docs/, refactor/, etc.

# ... make changes ...

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
   make patch   # 0.1.0 → 0.1.1  (bug fixes)
   make minor   # 0.1.0 → 0.2.0  (new features)
   make major   # 0.1.0 → 1.0.0  (breaking changes)
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