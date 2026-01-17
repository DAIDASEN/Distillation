#!/usr/bin/env bash
set -euo pipefail

NOW=$(date +%Y%m%d)

export WANDB_DIR=${WANDB_DIR:-"simplerl-grpo-qwen3-1.7b-${NOW}"}
export WANDB_PROJECT=${WANDB_PROJECT:-${WANDB_DIR}}
export WANDB_EXP=${WANDB_EXP:-"grpo-${NOW}"}
export WANDB_MODE=${WANDB_MODE:-"online"}

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-1.7B"}
TRAIN_FILE=${TRAIN_FILE:-"data/simpleRL/simplelr_qwen_level3to5/train.parquet"}
TEST_FILE=${TEST_FILE:-"data/simpleRL/simplelr_qwen_level3to5/test.parquet"}

# Effective global batch size (defaults to SimpleRL-aligned 1024 = 128 * 8).
NGPU_PER_NODE=${NGPU_PER_NODE:-8}
NPROC_PER_GPU=${NPROC_PER_GPU:-128}
TOTAL_PROCS=$(( NPROC_PER_GPU * NGPU_PER_NODE ))

# PPO micro-batch for grad accumulation (global, legacy actor path).
PPO_MICRO=${PPO_MICRO:-128}

SMOKE=${SMOKE:-0}
FILTER_TRIVIAL_GROUPS=${FILTER_TRIVIAL_GROUPS:-1}  # 1 = drop all-correct/all-wrong GRPO groups

extra_args=()
if [[ "${SMOKE}" == "1" ]]; then
  echo "[SMOKE=1] applying small overrides"
  TOTAL_PROCS=8
  PPO_MICRO=4
  extra_args+=(
    data.max_prompt_length=256
    data.max_response_length=256
    actor_rollout_ref.rollout.n=2
    actor_rollout_ref.rollout.max_model_len=512
    actor_rollout_ref.rollout.max_num_batched_tokens=512
    actor_rollout_ref.rollout.max_num_seqs=32
  )
fi
if [[ "${FILTER_TRIVIAL_GROUPS}" == "1" ]]; then
  extra_args+=(
    algorithm.filter_groups.enable=True
    algorithm.filter_groups.metric=acc
  )
fi

set -x
python3 -m verl.trainer.main_ppo \
  trainer.use_legacy_worker_impl=enable \
  algorithm.adv_estimator=grpo \
  data.custom_cls.path=verl/experimental/dataset/simple_rl_dataset.py \
  data.custom_cls.name=SimpleRLRLDataset \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.train_batch_size="${TOTAL_PROCS}" \
  data.val_batch_size="${TOTAL_PROCS}" \
  data.max_prompt_length=1024 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.shuffle=False \
  reward_model.launch_reward_fn_async=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=simple_rl_raw_prompt_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len=9216 \
  actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
  actor_rollout_ref.rollout.max_num_seqs=256 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TOTAL_PROCS}" \
  actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="${PPO_MICRO}" \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=1e-4 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  algorithm.kl_ctrl.kl_coef=1e-4 \
  algorithm.use_kl_in_reward=False \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${WANDB_EXP}" \
  trainer.n_gpus_per_node="${NGPU_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=5 \
  trainer.total_epochs=1 \
  "${extra_args[@]}" \
  "$@" 2>&1 | tee "${WANDB_PROJECT}.log"
