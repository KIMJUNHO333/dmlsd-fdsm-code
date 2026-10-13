#!/usr/bin/env bash
set -euo pipefail

# For speaker evaluation on GPU, select GPU IDs and set device in configs/evaluation/speaker.json.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL_ROOT="${MODEL_ROOT:-$HOME/model}"
export HF_HOME="${HF_HOME:-$MODEL_ROOT/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$MODEL_ROOT/torch}"

cd "$(dirname "$0")/.."
python scripts/evaluate.py "$@"
