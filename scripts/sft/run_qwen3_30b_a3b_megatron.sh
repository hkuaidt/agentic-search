#!/usr/bin/env bash
# Full-parameter SFT of Qwen3-30B-A3B on 37k horizontal/vertical ReAct trajectories.
# Reference setup (PROPOSAL §5.1): 2 nodes x 8 A800 80GB, BF16, gradient checkpointing, expert parallel EP=8, ~24h.
# Based on ms-swift examples/megatron/moe/qwen3_moe.sh -- check flag names against your ms-swift version.
#
# Data: JSONL written by agentic_search.data.export_sft_jsonl (system/user/assistant/tool-result messages). ms-swift
# trains every assistant turn with --loss_scale default, so [Tool Result], system and user tokens get
# no loss -- exactly the mask of PROPOSAL §4.2(3).
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-30B-A3B}
DATA=${DATA:-data/sft/trajectories.jsonl}
OUT=${OUT:-checkpoints/sft/qwen3-30b-a3b-agentic-search}
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
NPROC_PER_NODE=8 \
megatron sft \
  --model "$MODEL" \
  --load_safetensors true \
  --save_safetensors true \
  --dataset "$DATA" \
  --split_dataset_ratio 0.01 \
  --loss_scale default \
  --tensor_model_parallel_size 2 \
  --expert_model_parallel_size 8 \
  --moe_grouped_gemm true \
  --moe_shared_expert_overlap true \
  --moe_permute_fusion true \
  --moe_aux_loss_coeff 1e-3 \
  --sequence_parallel true \
  --micro_batch_size 1 \
  --global_batch_size 64 \
  --packing true \
  --max_length 16384 \
  --recompute_granularity full \
  --recompute_method uniform \
  --recompute_num_layers 1 \
  --bf16 true \
  --finetune true \
  --cross_entropy_loss_fusion true \
  --lr 1e-5 \
  --min_lr 1e-6 \
  --lr_warmup_fraction 0.05 \
  --max_epochs 2 \
  --save "$OUT" \
  --save_interval 200 \
  --eval_interval 200 \
  --no_save_optim true \
  --no_save_rng true \
  --num_workers 8 \
  --dataset_num_proc 8 \
  --attention_backend flash
