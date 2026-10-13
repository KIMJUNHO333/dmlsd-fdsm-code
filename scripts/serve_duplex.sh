#!/usr/bin/env bash
set -euo pipefail

# Physical GPU IDs; cuda:0 uses the first selected GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL_ROOT="${MODEL_ROOT:-$HOME/model}"
export HF_HOME="${HF_HOME:-$MODEL_ROOT/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$MODEL_ROOT/torch}"
export SPEECHBRAIN_ECAPA_SAVEDIR="${SPEECHBRAIN_ECAPA_SAVEDIR:-$MODEL_ROOT/duplex/raon-speechchat-9b/speechbrain}"

cd "$(dirname "$0")/.."
exec python scripts/serve_duplex.py "$@"
