#!/usr/bin/env bash
set -euo pipefail

shopt -s nullglob
for script in plot_reg_*.py; do
  python "$script"
done
