#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

export ALFWORLD_DATA=/nfs-medical1-NB/yrc/datasets/alfworld
DATASETS_ROOT=${PROJECT_ROOT}/assets/datasets/alfworld
TRAIN_FILE=${DATASETS_ROOT}/train.parquet
VALID_SEEN_FILE=${DATASETS_ROOT}/valid_seen.parquet
VALID_UNSEEN_FILE=${DATASETS_ROOT}/valid_unseen.parquet
if [[ -f "${TRAIN_FILE}" &&
      -f "${VALID_SEEN_FILE}" &&
      -f "${VALID_UNSEEN_FILE}" ]]; then
    echo "Found ALFWorld dataset at ${DATASETS_ROOT}."
else
    echo "ALFWorld dataset not found. Preprocessing..."
    mkdir -p "${DATASETS_ROOT}"
    python ${PROJECT_ROOT}/lab/alfworld_preprocess.py \
        --splits train valid_seen valid_unseen \
        --output_dir ${DATASETS_ROOT}
fi

# ---- user-adjustable ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NUM_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
DEVICE=${DEVICE:-gpu}
INFER_BACKEND=${INFER_BACKEND:-vllm}
MODEL_PATH=/nfs-medical1-NB/yrc/models/Qwen/Qwen2.5-1.5B-Instruct
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${NUM_DEVICES}}

# Training parameters
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-140}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
TOTAL_SEQUENCE_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_TURNS=${MAX_TURNS:-35}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

if (( NUM_DEVICES == 1 )); then
    ROLLOUT_TP=${ROLLOUT_TP:-1}
else
    ROLLOUT_TP=${ROLLOUT_TP:-2}
fi
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
ROLLOUT_N=${ROLLOUT_N:-8}
# TODO: choose the best
ALFWORLD_ENV_SLOTS=${ALFWORLD_ENV_SLOTS:-16}

PROJECT_NAME=${PROJECT_NAME:-alfworld_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_lr1e-6_kl0.001}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

RUN_TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
TENSORBOARD_ROOT=${PROJECT_ROOT}/outputs/tensorboard
export TENSORBOARD_DIR=${TENSORBOARD_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}/${RUN_TIMESTAMP}
HYDRA_DIR=${PROJECT_ROOT}/outputs/hydra/${PROJECT_NAME}/${EXPERIMENT_NAME}/${RUN_TIMESTAMP}
CKPTS_DIR=${PROJECT_ROOT}/outputs/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}/${RUN_TIMESTAMP}
VALIDATION_DATA_DIR=${PROJECT_ROOT}/outputs/validation_log/${PROJECT_NAME}/${EXPERIMENT_NAME}/${RUN_TIMESTAMP}
TRAIN_LOG_FILE=${HYDRA_DIR}/training.log
mkdir -p "${TENSORBOARD_DIR}" "${HYDRA_DIR}"

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    data.train_files=${TRAIN_FILE}
    data.val_files=${VALID_SEEN_FILE}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=False
    data.truncation='error'
    algorithm.use_kl_in_reward=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    # False or True
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    # TODO: choose the best
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.agent.alfworld_env_slots=${ALFWORLD_ENV_SLOTS}
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS}
    actor_rollout_ref.rollout.max_model_len=${TOTAL_SEQUENCE_LENGTH}
    actor_rollout_ref.rollout.max_num_batched_tokens=${TOTAL_SEQUENCE_LENGTH}
)
if [[ "${INFER_BACKEND}" == "sglang" ]]; then
    ROLLOUT+=(
        +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer
    )
fi

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.device=cuda
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.default_local_dir=${CKPTS_DIR}
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}"
    trainer.val_before_train=False
)

EXTRA=(
    hydra.run.dir="${HYDRA_DIR}"
)

########################### launch ###########################
tensorboard \
  --logdir "${TENSORBOARD_ROOT}" \
  --host 0.0.0.0 \
  --port 6006 \
  > "${TENSORBOARD_ROOT}/tensorboard.log" 2>&1 &

TENSORBOARD_PID=$!
cleanup() {
    kill "${TENSORBOARD_PID}" 2>/dev/null || true
    wait "${TENSORBOARD_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@" \
    2>&1 | tee "${TRAIN_LOG_FILE}"
