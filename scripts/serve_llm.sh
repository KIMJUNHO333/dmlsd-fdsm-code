#!/usr/bin/env bash
set -euo pipefail

# Select GPU IDs; tensor parallelism uses the number of selected GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MODEL_ROOT="${MODEL_ROOT:-$HOME/model}"
export HF_HOME="${HF_HOME:-$MODEL_ROOT/huggingface}"

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$HERE/serve_llm.py" "$@"
