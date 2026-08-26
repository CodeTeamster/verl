#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${MODEL_PATH:?Set MODEL_PATH to the local model directory.}"
: "${ALFWORLD_DATA:?Set ALFWORLD_DATA to the uploaded ALFWorld assets.}"
: "${DATA_ROOT:?Set DATA_ROOT to the persistent parquet directory.}"
: "${RUN_DIR:?Set RUN_DIR to the persistent output directory.}"
: "${GPU_PER_POD:?Set GPU_PER_POD.}"

RUNNER="${RUNNER:-vllm}"
case "${RUNNER}" in
    vllm|sglang) ;;
    *)
        echo "RUNNER must be vllm or sglang, got: ${RUNNER}" >&2
        exit 2
        ;;
esac
python3 -c "import ${RUNNER}" || {
    echo "RUNNER=${RUNNER} is not available in the selected training image." >&2
    exit 1
}

[[ -d "${MODEL_PATH}" ]] || { echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2; exit 2; }
[[ -d "${ALFWORLD_DATA}" ]] || { echo "ALFWORLD_DATA does not exist: ${ALFWORLD_DATA}" >&2; exit 2; }
export ALFWORLD_DATA

TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/train.parquet}"
VALID_SEEN_FILE="${VALID_SEEN_FILE:-${DATA_ROOT}/valid_seen.parquet}"
VALID_UNSEEN_FILE="${VALID_UNSEEN_FILE:-${DATA_ROOT}/valid_unseen.parquet}"
if [[ ! -f "${TRAIN_FILE}" || ! -f "${VALID_SEEN_FILE}" || ! -f "${VALID_UNSEEN_FILE}" ]]; then
    mkdir -p "${DATA_ROOT}"
    python3 lab/alfworld_preprocess.py \
        --splits train valid_seen valid_unseen \
        --output_dir "${DATA_ROOT}"
fi

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-140}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-${PPO_MICRO_BATCH_SIZE_PER_GPU}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
TOTAL_SEQUENCE_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_TURNS="${MAX_TURNS:-35}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.6}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ALFWORLD_ENV_SLOTS="${ALFWORLD_ENV_SLOTS:-16}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-8}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
PROJECT_NAME="${PROJECT_NAME:-alfworld_grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2.5}"
TRAINER_NNODES="${TRAINER_NNODES:-1}"
DEVICE="${DEVICE:-cuda}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-True}"
ENFORCE_EAGER="${ENFORCE_EAGER:-False}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
UPDATE_WEIGHTS_BUCKET_MB="${UPDATE_WEIGHTS_BUCKET_MB:-4096}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

# Preserve the lab/default FSDP precision while allowing an image-specific
# choice to be supplied by a submission script without changing this repo.
ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-fp32}"
ACTOR_FSDP_DTYPE="${ACTOR_FSDP_DTYPE:-bfloat16}"
REF_MODEL_DTYPE="${REF_MODEL_DTYPE:-${ACTOR_MODEL_DTYPE}}"
REF_FSDP_DTYPE="${REF_FSDP_DTYPE:-${ACTOR_FSDP_DTYPE}}"

HYDRA_DIR="${RUN_DIR}/hydra"
CKPTS_DIR="${RUN_DIR}/ckpts"
VALIDATION_DATA_DIR="${RUN_DIR}/validation"
mkdir -p "${HYDRA_DIR}" "${CKPTS_DIR}" "${VALIDATION_DATA_DIR}"

echo "================== ALFWorld GRPO preflight =================="
echo "RUNNER=${RUNNER} TRAINER_NNODES=${TRAINER_NNODES} GPU_PER_POD=${GPU_PER_POD} ROLLOUT_TP=${ROLLOUT_TP}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "ALFWORLD_DATA=${ALFWORLD_DATA}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "VALID_SEEN_FILE=${VALID_SEEN_FILE}"
echo "VALID_UNSEEN_FILE=${VALID_UNSEEN_FILE}"
echo "RUN_DIR=${RUN_DIR}"
echo "QUAKE_TENSORBOARD_DIR=${QUAKE_TENSORBOARD_DIR:-<unset>}"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} VAL_BATCH_SIZE=${VAL_BATCH_SIZE} ROLLOUT_N=${ROLLOUT_N} VAL_ROLLOUT_N=${VAL_ROLLOUT_N}"
echo "AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS} ALFWORLD_ENV_SLOTS=${ALFWORLD_ENV_SLOTS}"
echo "PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
echo "MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH} MAX_TURNS=${MAX_TURNS}"
echo "ACTOR_MODEL_DTYPE=${ACTOR_MODEL_DTYPE} ACTOR_FSDP_DTYPE=${ACTOR_FSDP_DTYPE} ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "RESUME_MODE=${RESUME_MODE} RESUME_FROM_PATH=${RESUME_FROM_PATH:-<auto>}"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; elif command -v ppu-smi >/dev/null 2>&1; then ppu-smi; fi

monitor_tmp_disk() {
    local interval="${DISK_MONITOR_INTERVAL:-60}"
    while true; do
        echo "--- /tmp disk sample $(date '+%F %T') ---"
        df -hP /tmp
        sleep "${interval}"
    done
}

monitor_tmp_disk &
tmp_disk_monitor_pid=$!
trap 'kill "${tmp_disk_monitor_pid}" 2>/dev/null || true; wait "${tmp_disk_monitor_pid}" 2>/dev/null || true' EXIT

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_PATH}" ]]; then
    RESUME_ARGS+=("trainer.resume_from_path=${RESUME_FROM_PATH}")
fi

ROLLOUT_ARGS=(
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
    "actor_rollout_ref.rollout.name=${RUNNER}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}"
    "actor_rollout_ref.rollout.enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL}"
    "actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER}"
    "actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE}"
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MB}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N}"
    "actor_rollout_ref.rollout.agent.alfworld_env_slots=${ALFWORLD_ENV_SLOTS}"
    "actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS}"
    "actor_rollout_ref.rollout.multi_turn.enable=True"
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS}"
    "actor_rollout_ref.rollout.max_model_len=${TOTAL_SEQUENCE_LENGTH}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${TOTAL_SEQUENCE_LENGTH}"
)
if [[ "${RUNNER}" == "sglang" ]]; then
    ROLLOUT_ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer")
fi

python3 -m verl.trainer.main_ppo \
    "+ray_kwargs.ray_init.address=auto" \
    "algorithm.adv_estimator=grpo" \
    "algorithm.use_kl_in_reward=False" \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=${VALID_SEEN_FILE}" \
    "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
    "data.val_batch_size=${VAL_BATCH_SIZE}" \
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}" \
    "data.max_response_length=${MAX_RESPONSE_LENGTH}" \
    "data.filter_overlong_prompts=False" \
    "data.truncation=error" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}" \
    "actor_rollout_ref.model.enable_gradient_checkpointing=True" \
    "+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}" \
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    "actor_rollout_ref.actor.use_kl_loss=True" \
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}" \
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl" \
    "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}" \
    "actor_rollout_ref.actor.fsdp_config.model_dtype=${ACTOR_MODEL_DTYPE}" \
    "actor_rollout_ref.actor.fsdp_config.dtype=${ACTOR_FSDP_DTYPE}" \
    "actor_rollout_ref.actor.fsdp_config.param_offload=False" \
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    "actor_rollout_ref.ref.fsdp_config.model_dtype=${REF_MODEL_DTYPE}" \
    "actor_rollout_ref.ref.fsdp_config.dtype=${REF_FSDP_DTYPE}" \
    "actor_rollout_ref.ref.fsdp_config.param_offload=True" \
    "${ROLLOUT_ARGS[@]}" \
    "trainer.device=${DEVICE}" \
    "trainer.critic_warmup=0" \
    "trainer.logger=['console','tensorboard']" \
    "trainer.project_name=${PROJECT_NAME}" \
    "trainer.experiment_name=${EXPERIMENT_NAME}" \
    "trainer.n_gpus_per_node=${GPU_PER_POD}" \
    "trainer.nnodes=${TRAINER_NNODES}" \
    "trainer.save_freq=${SAVE_FREQ}" \
    "trainer.test_freq=${TEST_FREQ}" \
    "trainer.total_epochs=${TOTAL_EPOCHS}" \
    "trainer.default_local_dir=${CKPTS_DIR}" \
    "trainer.validation_data_dir=${VALIDATION_DATA_DIR}" \
    "trainer.val_before_train=${VAL_BEFORE_TRAIN}" \
    "trainer.resume_mode=${RESUME_MODE}" \
    "hydra.run.dir=${HYDRA_DIR}" \
    "${RESUME_ARGS[@]}" \
    "$@"
