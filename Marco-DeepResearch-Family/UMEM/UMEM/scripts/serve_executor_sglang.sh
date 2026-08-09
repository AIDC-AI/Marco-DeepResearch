#!/usr/bin/env bash
set -euo pipefail

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 1
  fi
}

require_env EXECUTOR_MODEL_PATH

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-executor_llm}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
SCHEDULE_CONSERVATIVENESS="${SCHEDULE_CONSERVATIVENESS:-0.3}"
ENABLE_METRICS="${ENABLE_METRICS:-true}"

cmd=(
  python3 -u -m sglang.launch_server
  --model-path "${EXECUTOR_MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --tp "${TENSOR_PARALLEL_SIZE}"
  --dp "${DATA_PARALLEL_SIZE}"
  --schedule-conservativeness "${SCHEDULE_CONSERVATIVENESS}"
  --host "${HOST}"
  --port "${PORT}"
)

if [[ "${ENABLE_METRICS}" == "true" ]]; then
  cmd+=(--enable-metrics)
fi

echo "Starting SGLang executor service"
echo "Model: ${EXECUTOR_MODEL_PATH}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Endpoint: http://${HOST}:${PORT}/v1/chat/completions"
echo "Tensor parallel: ${TENSOR_PARALLEL_SIZE}; data parallel: ${DATA_PARALLEL_SIZE}"

exec "${cmd[@]}"
