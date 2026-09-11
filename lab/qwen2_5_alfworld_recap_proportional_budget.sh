#!/usr/bin/env bash
# Validate proportional replay allocation and suffix-heavy candidate selection.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

export RECAP_VERSION=v2
export PROJECT_NAME=${PROJECT_NAME:-alfworld_recap_proportional_budget_from_scratch}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_ratio0.40_max6_suffix0.75}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TEST_FREQ=${TEST_FREQ:-10}

exec bash "${SCRIPT_DIR}/qwen2_5_alfworld_recap_grpo.sh" \
    +algorithm.recap_grpo.replay_ratio=0.40 \
    +algorithm.recap_grpo.max_replay_budget=6 \
    +algorithm.recap_grpo.replay_budget_rounding=ceil \
    +algorithm.recap_grpo.suffix_fraction=0.75 \
    "$@"
