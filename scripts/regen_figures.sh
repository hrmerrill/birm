#!/usr/bin/env bash
# Regenerate every paper figure into cache/ and copy it into the_paper/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN=(uv run --with netCDF4 --with matplotlib python)

"${RUN[@]}" scripts/plot_posterior_grid.py
"${RUN[@]}" scripts/plot_runner_application.py
"${RUN[@]}" scripts/plot_study.py
"${RUN[@]}" scripts/plot_tau_regime.py

FIGURES=(
  posterior_grid
  runner_application
  sim_posterior_geometry
  sim_baseline_vs_constrained
  sim_constrained_vs_joint
  sim_tau_regime
)
for f in "${FIGURES[@]}"; do
  cp "cache/$f.jpg" "the_paper/$f.jpg"
  echo "copied -> the_paper/$f.jpg"
done
