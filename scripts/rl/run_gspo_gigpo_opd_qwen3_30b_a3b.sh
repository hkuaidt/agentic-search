#!/usr/bin/env bash
# Online agentic RL: GSPO (trajectory) + GiGPO (step) + OPD (token) on Qwen3-30B-A3B.
# Reference setup (PROPOSAL §5.1): 10k deduplicated queries x 8 rollouts, 4 nodes x 8 A800 80GB, ~72h, VERL + Megatron-Core
# (training) + vLLM (rollout); teacher Qwen3-235B-A22B on a separate vLLM cluster (scripts/serve).
# Template modelled on VERL's examples/gspo_trainer/run_qwen3_30b_a3b_megatron.sh -- verify keys against
# the installed VERL version before a long run, and follow docs/rl_debug_protocol.md first.
set -euo pipefail

export AGENTIC_SEARCH_CONFIG=${AGENTIC_SEARCH_CONFIG:-$(pwd)/configs/agentic_search.yaml}  # needed on every node
SFT_CKPT=${SFT_CKPT:-checkpoints/sft/qwen3-30b-a3b-agentic-search}
TRAIN=${TRAIN:-data/rl/train.parquet}
VAL=${VAL:-data/rl/val.parquet}
REWARD_ENTRY=$(python -c "import agentic_search.integrations.verl.reward_entry as m; print(m.__file__)")

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gspo_gigpo_opd \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN" \
  data.val_files="$VAL" \
  data.return_raw_chat=True \
  data.train_batch_size=128 \
  data.max_prompt_length=6144 \
  data.max_response_length=10240 \
  actor_rollout_ref.model.path="$SFT_CKPT" \
  actor_rollout_ref.actor.strategy=megatron \
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.clip_ratio_low=3e-4 \
  actor_rollout_ref.actor.clip_ratio_high=4e-4 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=8 \
  actor_rollout_ref.actor.megatron.param_offload=True \
  actor_rollout_ref.actor.megatron.optimizer_offload=True \
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
  actor_rollout_ref.ref.megatron.expert_model_parallel_size=8 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/verl/agent_loop.yaml \
  custom_reward_function.path="$REWARD_ENTRY" \
  custom_reward_function.name=compute_score \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  trainer.total_epochs=1 \
  trainer.save_freq=20 \
  trainer.test_freq=20 \
  trainer.project_name=agentic-search \
  trainer.experiment_name=qwen3-30b-a3b-gspo-gigpo-opd \
  trainer.logger='["console","wandb"]' \
  "$@"
