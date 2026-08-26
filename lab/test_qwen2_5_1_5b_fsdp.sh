#!/usr/bin/env bash
# GRPO | Qwen2.5-32B | FSDP training | NVIDIA GPUs or Ascend NPUs
#
# INFER_BACKEND controls rollout backend: vllm

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

DATASETS_ROOT=${PROJECT_ROOT}/assets/datasets/gsm8k
TRAIN_FILE=${DATASETS_ROOT}/train.parquet
TEST_FILE=${DATASETS_ROOT}/test.parquet

if [[ -f ${TRAIN_FILE} && -f ${TEST_FILE} ]]; then
    echo "Found GSM8K dataset at ${DATASETS_ROOT}."
else
    echo "GSM8K dataset not found. Preprocessing..."
    mkdir -p "${DATASETS_ROOT}"
    python ${PROJECT_ROOT}/examples/data_preprocess/gsm8k.py \
        --local_dataset_path /nfs-medical1-NB/yrc/datasets/gsm8k \
        --local_save_dir ${DATASETS_ROOT}
fi

# ---- user-adjustable ----
export CUDA_VISIBLE_DEVICES=0,1
DEVICE=${DEVICE:-gpu}
INFER_BACKEND=${INFER_BACKEND:-vllm}
MODEL_PATH=/nfs-medical1-NB/yrc/models/Qwen/Qwen2.5-1.5B-Instruct
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}

# Training parameters
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
ROLLOUT_N=${ROLLOUT_N:-2}

PROJECT_NAME=${PROJECT_NAME:-gsm8k_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5_1.5b_fsdp_lr1e-6}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}

RUN_TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
TENSORBOARD_ROOT=${PROJECT_ROOT}/outputs/tensorboard
export TENSORBOARD_DIR=${TENSORBOARD_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}/${RUN_TIMESTAMP}
HYDRA_DIR=${PROJECT_ROOT}/outputs/hydra/${PROJECT_NAME}/${EXPERIMENT_NAME}/${RUN_TIMESTAMP}
TRAIN_LOG_FILE=${HYDRA_DIR}/training.log
mkdir -p "${TENSORBOARD_DIR}" "${HYDRA_DIR}"

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    algorithm.use_kl_in_reward=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
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
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
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
