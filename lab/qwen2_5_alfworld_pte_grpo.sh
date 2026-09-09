#!/usr/bin/env bash
# From-scratch PTE-GRPO. The ordinary ALFWorld policy loop is unchanged; only
# the trajectory advantage additionally prefers lower-PTE successful rollouts.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PROJECT_NAME=${PROJECT_NAME:-alfworld_pte_grpo_from_scratch}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_cost0.10_full}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-140}
export ROLLOUT_N=${ROLLOUT_N:-8}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TEST_FREQ=${TEST_FREQ:-10}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export PTE_COST_COEF=${PTE_COST_COEF:-0.10}
export PTE_GAMMA=${PTE_GAMMA:-0.00704}

exec bash "${SCRIPT_DIR}/qwen2_5_alfworld_grpo.sh" \
    algorithm.adv_estimator=pte_grpo \
    +algorithm.pte_grpo.cost_coef=${PTE_COST_COEF} \
    +algorithm.pte_grpo.gamma=${PTE_GAMMA} \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.pte_grpo.agent_loop.PTEGRPOAgentLoopManager \
    "$@"
