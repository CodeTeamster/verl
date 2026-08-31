#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

export ALFWORLD_DATA=/nfs-medical1-NB/yrc/datasets/alfworld
DATASETS_ROOT=${PROJECT_ROOT}/assets/datasets/alfworld
VALID_SEEN_FILE=${DATASETS_ROOT}/valid_seen.parquet
VALID_UNSEEN_FILE=${DATASETS_ROOT}/valid_unseen.parquet
if [[ -f "${VALID_SEEN_FILE}" &&
      -f "${VALID_UNSEEN_FILE}" ]]; then
    echo "Found ALFWorld dataset at ${DATASETS_ROOT}."
else
    echo "ALFWorld dataset not found. Preprocessing..."
    mkdir -p "${DATASETS_ROOT}"
    python ${PROJECT_ROOT}/lab/alfworld_preprocess.py \
        --splits valid_seen valid_unseen \
        --output_dir ${DATASETS_ROOT}
fi

# ---- user-adjustable ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NUM_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
INFER_BACKEND=${INFER_BACKEND:-vllm}
# outputs/ckpts/lingjing/alfworld_qwen2_5_1_5b_grpo_222
# /nfs-medical1-NB/yrc/models/Qwen/Qwen2.5-1.5B-Instruct
MODEL_PATH=${MODEL_PATH:-outputs/ckpts/lingjing/alfworld_qwen2_5_1_5b_grpo_222}
NNODES=1
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${NUM_DEVICES}}

# Validating parameters
VAL_BATCH_SIZE=null
VAL_MAX_SAMPLES=140
# VAL_SAMPLE_START=0
# VAL_SAMPLE_END=140
VAL_ROLLOUT_N=1
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
TOTAL_SEQUENCE_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_TURNS=${MAX_TURNS:-35}

if (( NUM_DEVICES == 1 )); then
    ROLLOUT_TP=${ROLLOUT_TP:-1}
else
    ROLLOUT_TP=${ROLLOUT_TP:-2}
fi
ALFWORLD_ENV_SLOTS=${ALFWORLD_ENV_SLOTS:-16}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-8}

PROJECT_NAME=${PROJECT_NAME:-alfworld_grpo_val}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_lr1e-6_kl0.001}
RUN_TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
HYDRA_DIR=${PROJECT_ROOT}/outputs/hydra/${PROJECT_NAME}/${EXPERIMENT_NAME}/${RUN_TIMESTAMP}
EVALUATION_DIR=${PROJECT_ROOT}/outputs/validation_log/${PROJECT_NAME}/${EXPERIMENT_NAME}/${RUN_TIMESTAMP}
VAL_LOG_FILE=${HYDRA_DIR}/validating.log
mkdir -p "${HYDRA_DIR}"

########################### parameter arrays ###########################
DATA=(
    "data.val_files=[${VALID_SEEN_FILE},${VALID_UNSEEN_FILE}]"
    "data.val_max_samples=${VAL_MAX_SAMPLES}"
    "data.val_batch_size=${VAL_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=False"
    "data.truncation=error"
    "data.seed=${RANDOM_SEED:-42}"
    "data.val_sample_start=${VAL_SAMPLE_START:-null}"
    "data.val_sample_end=${VAL_SAMPLE_END:-null}"
)

MODEL=(
    "actor_rollout_ref.model.path=${MODEL_PATH}"
)

ROLLOUT=(
    "actor_rollout_ref.rollout.nnodes=${NNODES}"
    "actor_rollout_ref.rollout.n_gpus_per_node=${NGPUS_PER_NODE}"
    "actor_rollout_ref.rollout.name=${INFER_BACKEND}"
    "actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
    "actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}"
    "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}"
    "actor_rollout_ref.rollout.max_model_len=${TOTAL_SEQUENCE_LENGTH}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${TOTAL_SEQUENCE_LENGTH}"
    "actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS}"
    "actor_rollout_ref.rollout.agent.alfworld_env_slots=${ALFWORLD_ENV_SLOTS}"
    "actor_rollout_ref.rollout.multi_turn.enable=True"
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS}"
)

EVALUATOR=(
    "evaluator.output_dir=${EVALUATION_DIR}"
)

EXTRA=(
    "hydra.run.dir=${HYDRA_DIR}"
)

########################### launch ###########################
python3 -m verl.evaluator.main \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${EVALUATOR[@]}" \
    "${EXTRA[@]}" \
    "$@" \
    2>&1 | tee "${VAL_LOG_FILE}"
