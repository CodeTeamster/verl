#!/usr/bin/env bash
# From-scratch ELICA-GRPO with training-only latent turn credit.
# The rollout/environment flow and trajectory-level PTE reward are unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PROJECT_NAME=${PROJECT_NAME:-alfworld_elica_grpo_from_scratch}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_beta0.10}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-140}
export ROLLOUT_N=${ROLLOUT_N:-8}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TEST_FREQ=${TEST_FREQ:-10}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

exec bash "${SCRIPT_DIR}/qwen2_5_alfworld_pte_grpo.sh" \
    algorithm.adv_estimator=elica_grpo \
    +actor_rollout_ref.actor.elica.enable=True \
    +actor_rollout_ref.actor.elica.beta=${ELICA_BETA:-0.10} \
    +actor_rollout_ref.actor.elica.loss_coef=${ELICA_LOSS_COEF:-0.10} \
    +actor_rollout_ref.actor.elica.lr=${ELICA_LR:-1e-3} \
    +actor_rollout_ref.actor.elica.width=${ELICA_WIDTH:-128} \
    "$@"
