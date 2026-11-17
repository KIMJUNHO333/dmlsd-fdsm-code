#!/usr/bin/env bash
set -euo pipefail

# Physical GPU IDs; cuda:0 uses the first selected GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$(dirname "$0")/.."
python scripts/simulate.py "$@"
