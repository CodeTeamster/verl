#!/usr/bin/env bash
# GiGPO differs from the shared GRPO launcher only in its advantage estimator
# and action-local metadata collection.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PROJECT_NAME=${PROJECT_NAME:-alfworld_gigpo}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_lr1e-6_kl0.001}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-140}
export ROLLOUT_N=${ROLLOUT_N:-8}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TEST_FREQ=${TEST_FREQ:-10}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

GIGPO_GAMMA=${GIGPO_GAMMA:-0.95}
GIGPO_STEP_ADVANTAGE_W=${GIGPO_STEP_ADVANTAGE_W:-1.0}
GIGPO_MODE=${GIGPO_MODE:-mean_std_norm}
GIGPO_ENABLE_SIMILARITY=${GIGPO_ENABLE_SIMILARITY:-False}
GIGPO_SIMILARITY_THRESH=${GIGPO_SIMILARITY_THRESH:-0.95}

exec bash "${SCRIPT_DIR}/qwen2_5_alfworld_grpo.sh" \
    algorithm.adv_estimator=gigpo \
    algorithm.gamma=${GIGPO_GAMMA} \
    +algorithm.gigpo.step_advantage_w=${GIGPO_STEP_ADVANTAGE_W} \
    +algorithm.gigpo.mode=${GIGPO_MODE} \
    +algorithm.gigpo.enable_similarity=${GIGPO_ENABLE_SIMILARITY} \
    +algorithm.gigpo.similarity_thresh=${GIGPO_SIMILARITY_THRESH} \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.gigpo.agent_loop.GiGPOAgentLoopManager \
    "$@"
