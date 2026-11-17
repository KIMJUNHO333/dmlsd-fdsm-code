#!/usr/bin/env bash
set -euo pipefail

# Select GPU IDs; tensor parallelism uses the number of selected GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$HERE/serve_llm.py" "$@"
