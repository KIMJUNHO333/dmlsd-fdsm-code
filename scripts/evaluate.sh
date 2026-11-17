#!/usr/bin/env bash
set -euo pipefail

# For speaker evaluation on GPU, select GPU IDs and set device in configs/evaluation/speaker.json.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$(dirname "$0")/.."
python scripts/evaluate.py "$@"
