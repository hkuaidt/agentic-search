#!/usr/bin/env bash
# Frozen OPD teacher (Qwen3-235B-A22B) on its own node. Only prefill / prompt_logprobs is used.
# BF16 weights are ~470 GB: 8x80GB fits with limited KV cache; the FP8 checkpoint leaves more headroom
# (logprob numerics differ slightly from BF16 -- keep one choice fixed for a whole run).
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3-235B-A22B}
vllm serve "$MODEL" \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 20480 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --port "${PORT:-8000}"
