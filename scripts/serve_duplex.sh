#!/usr/bin/env bash
set -euo pipefail

# Physical GPU IDs; cuda:0 uses the first selected GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Transformers names its remote-code copy after the model directory, so separate runs
# would share and overwrite one copy. Each run keeps its own and removes it on exit.
if [ -z "${HF_MODULES_CACHE:-}" ]; then
  HF_MODULES_CACHE="$(mktemp -d)"
  export HF_MODULES_CACHE
  trap 'rm -rf "$HF_MODULES_CACHE"' EXIT
fi

cd "$(dirname "$0")/.."
python scripts/serve_duplex.py "$@"
