#!/usr/bin/env bash
# Canonical RECAP-GRPO launcher. Use RECAP_VERSION=v1 for the binary ablation;
# V2 is the complete signed counterfactual-utility method and the default.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RECAP_VERSION=${RECAP_VERSION:-v2}

case "${RECAP_VERSION}" in
  v1)
    export PROJECT_NAME=${PROJECT_NAME:-alfworld_recap_grpo_from_scratch}
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_replay2_weight0.10}
    LOCAL_CREDIT_MODE=binary_removable
    LOCAL_NORMALIZATION=none
    ;;
  v2)
    export PROJECT_NAME=${PROJECT_NAME:-alfworld_recap_cfutility_from_scratch}
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_replay2_signed_delta_beta0.10}
    LOCAL_CREDIT_MODE=counterfactual_utility
    LOCAL_NORMALIZATION=signed_absmax
    ;;
  *)
    echo "RECAP_VERSION must be v1 or v2, got: ${RECAP_VERSION}" >&2
    exit 2
    ;;
esac

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

exec bash "${SCRIPT_DIR}/qwen2_5_alfworld_grpo.sh" \
    algorithm.adv_estimator=recap_grpo \
    +algorithm.recap_grpo.cost_coef=${RECAP_COST_COEF} \
    +algorithm.recap_grpo.local_weight=${RECAP_LOCAL_WEIGHT} \
    +algorithm.recap_grpo.replay_budget=${RECAP_REPLAY_BUDGET} \
    +algorithm.recap_grpo.gamma=${PTE_GAMMA} \
    +algorithm.recap_grpo.local_credit_mode=${LOCAL_CREDIT_MODE} \
    +algorithm.recap_grpo.local_normalization=${LOCAL_NORMALIZATION} \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.pte_grpo.agent_loop.PTEGRPOAgentLoopManager \
    "$@"
