#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65535

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}}"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 1
  fi
}

require_env MODEL_PATH
require_env TRAIN_FILE_PATH
require_env TEST_FILE_PATH
require_env RETRIEVER_MODEL_PATH
require_env EXECUTOR_API_URL

if [[ ! -f "${TRAIN_FILE_PATH}" ]]; then
  echo "Training parquet not found: ${TRAIN_FILE_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TEST_FILE_PATH}" ]]; then
  echo "Validation parquet not found: ${TEST_FILE_PATH}" >&2
  exit 1
fi

cd "${WORK_DIR}"

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-umem_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${WORK_DIR}/outputs}"
SAVE_PATH="${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}"
mkdir -p "${SAVE_PATH}"

N_GPUS="${N_GPUS:-8}"
WORLD_SIZE="${WORLD_SIZE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
ROLLOUT_NUM="${ROLLOUT_NUM:-8}"
EPOCH="${EPOCH:-3}"
LR="${LR:-1e-6}"
TAU="${TAU:-1.0}"
KL_W="${KL_W:-0.001}"
USE_KL="${USE_KL:-true}"

RETRIEVER_ENABLE="${RETRIEVER_ENABLE:-true}"
RETRIEVER_USE_FP16="${RETRIEVER_USE_FP16:-true}"
RETRIEVER_MAX_LENGTH="${RETRIEVER_MAX_LENGTH:-8196}"
RETRIEVER_BATCH_SIZE="${RETRIEVER_BATCH_SIZE:-4096}"
RETRIEVER_FAISS_GPU="${RETRIEVER_FAISS_GPU:-false}"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
RETRIEVAL_THRESHOLD="${RETRIEVAL_THRESHOLD:-0.5}"

EXECUTOR_API_CODE="${EXECUTOR_API_CODE:-executor_llm}"
EXECUTOR_AK="${EXECUTOR_AK:-EMPTY}"
EXECUTOR_MAX_CONCURRENCY="${EXECUTOR_MAX_CONCURRENCY:-512}"
EXECUTOR_TIMEOUT="${EXECUTOR_TIMEOUT:-1200}"
EXECUTOR_MAX_RETRIES="${EXECUTOR_MAX_RETRIES:-10}"
EXECUTOR_REPEAT_N="${EXECUTOR_REPEAT_N:-1}"
EXECUTOR_ENABLE_THINKING="${EXECUTOR_ENABLE_THINKING:-false}"
EXECUTOR_MAX_INPUT_TOKENS="${EXECUTOR_MAX_INPUT_TOKENS:-16384}"
EXECUTOR_MAX_TOKENS="${EXECUTOR_MAX_TOKENS:-16384}"
EXECUTOR_TEMPERATURE="${EXECUTOR_TEMPERATURE:-0.0}"
EXECUTOR_TOP_P="${EXECUTOR_TOP_P:-0.8}"
EXECUTOR_TOP_K="${EXECUTOR_TOP_K:-20}"
EXECUTOR_MIN_P="${EXECUTOR_MIN_P:-0}"

EXTRACTOR_W_FORMAT="${EXTRACTOR_W_FORMAT:-1.0}"
EXTRACTOR_W_QUALITY="${EXTRACTOR_W_QUALITY:-1.0}"

PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-20}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console']}"
TRAINER_ROLLOUT_SAVE_DIR="${TRAINER_ROLLOUT_SAVE_DIR-${SAVE_PATH}/rollouts}"

MEMORY_JSONL_PATH="${SAVE_PATH}/memory.jsonl"
MEMORY_INDEX_PATH="${SAVE_PATH}/memory.index"

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Work dir: ${WORK_DIR}"
echo "Output dir: ${SAVE_PATH}"
echo "Train data: ${TRAIN_FILE_PATH}"
echo "Validation data: ${TEST_FILE_PATH}"
echo "Model: ${MODEL_PATH}"
echo "Retriever model: ${RETRIEVER_MODEL_PATH}"

python3 -u -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  trainer.resume_mode=auto \
  data.train_files="${TRAIN_FILE_PATH}" \
  data.val_files="${TEST_FILE_PATH}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=24500 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss="${USE_KL}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_W}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_NUM}" \
  actor_rollout_ref.rollout.temperature="${TAU}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.reward_manager=extractor \
  reward_model.val_reward_manager=val_extractor \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.use_kl_in_reward=False \
  retriever.enable="${RETRIEVER_ENABLE}" \
  retriever.model_path="${RETRIEVER_MODEL_PATH}" \
  retriever.use_fp16="${RETRIEVER_USE_FP16}" \
  retriever.max_length="${RETRIEVER_MAX_LENGTH}" \
  retriever.batch_size="${RETRIEVER_BATCH_SIZE}" \
  retriever.faiss_gpu="${RETRIEVER_FAISS_GPU}" \
  retriever.topk="${RETRIEVER_TOPK}" \
  retriever.threshold="${RETRIEVAL_THRESHOLD}" \
  retriever.memory_jsonl_path="${MEMORY_JSONL_PATH}" \
  retriever.memory_index_path="${MEMORY_INDEX_PATH}" \
  executor.api_url="${EXECUTOR_API_URL}" \
  executor.api_code="${EXECUTOR_API_CODE}" \
  executor.ak="${EXECUTOR_AK}" \
  executor.max_concurrency="${EXECUTOR_MAX_CONCURRENCY}" \
  executor.timeout="${EXECUTOR_TIMEOUT}" \
  executor.max_retries="${EXECUTOR_MAX_RETRIES}" \
  executor.repeat_n="${EXECUTOR_REPEAT_N}" \
  executor.enable_thinking="${EXECUTOR_ENABLE_THINKING}" \
  executor.max_input_tokens="${EXECUTOR_MAX_INPUT_TOKENS}" \
  executor.max_tokens="${EXECUTOR_MAX_TOKENS}" \
  executor.temperature="${EXECUTOR_TEMPERATURE}" \
  executor.top_p="${EXECUTOR_TOP_P}" \
  executor.top_k="${EXECUTOR_TOP_K}" \
  executor.MinP="${EXECUTOR_MIN_P}" \
  extractor.w_format="${EXTRACTOR_W_FORMAT}" \
  extractor.w_quality="${EXTRACTOR_W_QUALITY}" \
  trainer.val_before_train=False \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name=umem \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.log_val_generations=0 \
  trainer.rollout_save_dir="${TRAINER_ROLLOUT_SAVE_DIR}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes="${WORLD_SIZE}" \
  trainer.default_local_dir="${SAVE_PATH}/save_ckpts" \
  trainer.default_hdfs_dir=null \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.ray_wait_register_center_timeout=1200 \
  trainer.total_epochs="${EPOCH}"
