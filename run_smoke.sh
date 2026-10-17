#!/usr/bin/env bash
# Smoke-test the patched Step5 DSA kernels on ONE GPU (no model weights needed).
# Usage:  ./run_smoke.sh [gpu_index]      (default: 0)
set -euo pipefail
cd "$(dirname "$0")"
GPU="${1:-0}"
IMAGE="${IMAGE:-vllm/vllm-openai:step-5-preview}"
V=/usr/local/lib/python3.12/dist-packages/vllm/models/step5/nvidia
P="$PWD/patches/vllm/models/step5/nvidia"

docker run --rm --gpus "\"device=${GPU}\"" --ipc=host \
  --entrypoint python3 \
  -v "$P/sparse_attention.py:$V/sparse_attention.py:ro" \
  -v "$P/flashinfer_sparse.py:$V/flashinfer_sparse.py:ro" \
  -v "$P/layernorm.py:$V/layernorm.py:ro" \
  -v "$P/ops/sparse_indexer_scores.py:$V/ops/sparse_indexer_scores.py:ro" \
  -v "$P/ops/sparse_decode_attention.py:$V/ops/sparse_decode_attention.py:ro" \
  -v "$PWD/smoke_sm120.py:/smoke_sm120.py:ro" \
  "$IMAGE" /smoke_sm120.py 2>&1 | tee smoke_result.txt
