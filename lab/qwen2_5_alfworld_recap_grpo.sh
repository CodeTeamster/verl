#!/usr/bin/env bash
# Canonical launcher for the best RECAP-GRPO V2 configuration:
# fixed K=2, Mixed candidates, and signed abs-max turn-credit normalization.
# Use qwen2_5_alfworld_recap_ablations.sh for all ablation configurations.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

export PROJECT_NAME=${PROJECT_NAME:-alfworld_recap_cfutility_from_scratch}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_replay2_signed_delta_beta0.10}
LOCAL_CREDIT_MODE=${LOCAL_CREDIT_MODE:-counterfactual_utility}
LOCAL_NORMALIZATION=${LOCAL_NORMALIZATION:-signed_absmax}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-140}
export ROLLOUT_N=${ROLLOUT_N:-8}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TEST_FREQ=${TEST_FREQ:-10}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export RECAP_COST_COEF=${RECAP_COST_COEF:-0.10}
export RECAP_LOCAL_WEIGHT=${RECAP_LOCAL_WEIGHT:-0.10}
export RECAP_REPLAY_BUDGET=${RECAP_REPLAY_BUDGET:-2}
export PTE_GAMMA=${PTE_GAMMA:-0.00704}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-16}

exec bash "${SCRIPT_DIR}/qwen2_5_alfworld_grpo.sh" \
    ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS} \
    algorithm.adv_estimator=recap_grpo \
    +algorithm.recap_grpo.cost_coef=${RECAP_COST_COEF} \
    +algorithm.recap_grpo.local_weight=${RECAP_LOCAL_WEIGHT} \
    +algorithm.recap_grpo.replay_budget=${RECAP_REPLAY_BUDGET} \
    +algorithm.recap_grpo.gamma=${PTE_GAMMA} \
    +algorithm.recap_grpo.local_credit_mode=${LOCAL_CREDIT_MODE} \
    +algorithm.recap_grpo.local_normalization=${LOCAL_NORMALIZATION} \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.pte_grpo.agent_loop.PTEGRPOAgentLoopManager \
    "$@"
