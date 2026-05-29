#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

git clone https://github.com/sschrod/CODEX.git ~/CODEX/CODEX

python "${SCRIPT_DIR}/run_codex_repo_launcher.py" \
  --codex-dir "${SCRIPT_DIR}/CODEX" \
  --data-dir "${SCRIPT_DIR}/data" \
  --results-dir "${SCRIPT_DIR}/results" \
  --dataset norman \
  --download-data \
  --layers 256 64 32 \
  --seed 42 \
  --epochs 20 \
  --patience 1 \
  --batch-size 16
