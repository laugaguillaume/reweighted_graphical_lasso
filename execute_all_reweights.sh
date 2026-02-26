#!/usr/bin/env bash
set -euo pipefail

for k in 1 2 3 4 5 10; do
  for script in reg_path_gista.py reg_path_pglasso.py reg_path_quic.py; do
    python "$script" --max_iter_reweights "$k"
  done
done
