#!/usr/bin/env bash
# Unified launcher for completed RECAP-GRPO ablations.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

usage() {
    cat <<'EOF'
Usage: bash lab/qwen2_5_alfworld_recap_ablations.sh <ablation> [hydra overrides...]

Ablations:
  v1                 Binary removable-action credit, fixed K=2
  ratio040-max6      40% replay budget, max 6, 75% suffix candidates
  ratio020-max4      20% replay budget, max 4, 75% suffix candidates
  suffix-only        Fixed K=2, both candidates ranked by suffix impact
  signed-sum         Fixed K=2 Mixed, per-sign sum normalization
  anchorcap          25% budget, min 2/max 4, Mixed anchors, credit cap 2
EOF
}

if (( $# == 0 )); then
    usage >&2
    exit 2
fi

ABLATION=$1
shift
CONFIG_OVERRIDES=()

# All variants keep the same validation, checkpoint, and parallel environment
# defaults. PROJECT_NAME and EXPERIMENT_NAME remain externally overridable.
export SAVE_FREQ=${SAVE_FREQ:-50}
export TEST_FREQ=${TEST_FREQ:-10}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-16}

case "${ABLATION}" in
  v1)
    export PROJECT_NAME=${PROJECT_NAME:-alfworld_recap_grpo_from_scratch}
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_replay2_weight0.10}
    export RECAP_REPLAY_BUDGET=2
    export LOCAL_CREDIT_MODE=binary_removable
    export LOCAL_NORMALIZATION=none
    ;;
  ratio040-max6)
    export PROJECT_NAME=${PROJECT_NAME:-alfworld_recap_proportional_budget_from_scratch}
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_ratio0.40_max6_suffix0.75}
    export RECAP_REPLAY_BUDGET=2
    export LOCAL_CREDIT_MODE=counterfactual_utility
    export LOCAL_NORMALIZATION=signed_absmax
    CONFIG_OVERRIDES+=(
        +algorithm.recap_grpo.replay_ratio=0.40
        +algorithm.recap_grpo.max_replay_budget=6
        +algorithm.recap_grpo.replay_budget_rounding=ceil
        +algorithm.recap_grpo.suffix_fraction=0.75
    )
    ;;
  ratio020-max4)
    export PROJECT_NAME=${PROJECT_NAME:-alfworld_recap_ratio020_max4_from_scratch}
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_ratio0.20_max4_nearest_suffix0.75}
    export RECAP_REPLAY_BUDGET=2
    export LOCAL_CREDIT_MODE=counterfactual_utility
    export LOCAL_NORMALIZATION=signed_absmax
    CONFIG_OVERRIDES+=(
        +algorithm.recap_grpo.replay_ratio=0.20
        +algorithm.recap_grpo.max_replay_budget=4
        +algorithm.recap_grpo.replay_budget_rounding=nearest
        +algorithm.recap_grpo.suffix_fraction=0.75
    )
    ;;
  suffix-only)
    export PROJECT_NAME=${PROJECT_NAME:-alfworld_recap_fixed2_suffix_only_from_scratch}
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_replay2_suffix_only_beta0.10}
    export RECAP_REPLAY_BUDGET=2
    export LOCAL_CREDIT_MODE=counterfactual_utility
    export LOCAL_NORMALIZATION=signed_absmax
    CONFIG_OVERRIDES+=(+algorithm.recap_grpo.suffix_fraction=1.0)
    ;;
  signed-sum)
    export PROJECT_NAME=${PROJECT_NAME:-alfworld_recap_fixed2_mixed_signed_sum_from_scratch}
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_replay2_mixed_signed_sum_beta0.10}
    export RECAP_REPLAY_BUDGET=2
    export LOCAL_CREDIT_MODE=counterfactual_utility
    export LOCAL_NORMALIZATION=signed_sum
    ;;
  anchorcap)
    export PROJECT_NAME=${PROJECT_NAME:-alfworld_recap_anchorcap_from_scratch}
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_ratio025_min2_max4_anchor_cap2_beta0.10}
    export RECAP_REPLAY_BUDGET=4
    export LOCAL_CREDIT_MODE=counterfactual_utility
    export LOCAL_NORMALIZATION=signed_absmax_cap
    CONFIG_OVERRIDES+=(
        +algorithm.recap_grpo.replay_ratio=0.25
        +algorithm.recap_grpo.min_replay_budget=2
        +algorithm.recap_grpo.max_replay_budget=4
        +algorithm.recap_grpo.replay_budget_rounding=nearest
        +algorithm.recap_grpo.candidate_selection=anchored_alternating
        +algorithm.recap_grpo.sign_mass_cap=2.0
    )
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown RECAP ablation: ${ABLATION}" >&2
    usage >&2
    exit 2
    ;;
esac

exec bash "${SCRIPT_DIR}/qwen2_5_alfworld_recap_grpo.sh" \
    "${CONFIG_OVERRIDES[@]}" \
    "$@"
