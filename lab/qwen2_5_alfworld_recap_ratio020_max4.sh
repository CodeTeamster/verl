#!/usr/bin/env bash
# Test length-proportional replay near the fixed-two credit and compute budget.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

export RECAP_VERSION=v2
export PROJECT_NAME=${PROJECT_NAME:-alfworld_recap_ratio020_max4_from_scratch}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_ratio0.20_max4_nearest_suffix0.75}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TEST_FREQ=${TEST_FREQ:-10}

exec bash "${SCRIPT_DIR}/qwen2_5_alfworld_recap_grpo.sh" \
    +algorithm.recap_grpo.replay_ratio=0.20 \
    +algorithm.recap_grpo.max_replay_budget=4 \
    +algorithm.recap_grpo.replay_budget_rounding=nearest \
    +algorithm.recap_grpo.suffix_fraction=0.75 \
    "$@"
