#!/usr/bin/env bash
set -euo pipefail

# One immutable DFlash2 recipe, two target checkpoints. This is an internal
# launcher; use ../start-original.sh or ../start-abliterated.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-}"

case "${PROFILE}" in
  original)
    TARGET_MODEL="/models/unsloth/Qwen3.8-27B-NVFP4"
    ;;
  abliterated)
    TARGET_MODEL="/models/Huihui/Qwen3.8-27B-abliterated-NVFP4"
    ;;
  *)
    echo "internal error: profile must be original or abliterated" >&2
    exit 2
    ;;
esac

readonly IMAGE="lmsysorg/sglang:dev-qwen38-27b-dflash2"
readonly DRAFT_MODEL="/models/incoai/Qwen3.8-27B-DFlash2"
readonly CONTAINER_NAME="qwen3.8-27b-sglang"
readonly SERVED_MODEL_NAME="qwen3.8-27b-sglang"
readonly PORT="8888"
readonly STATE_DIR="${ROOT_DIR}/state"
readonly LOG_FILE="${STATE_DIR}/${PROFILE}.log"

command -v docker >/dev/null 2>&1 || { echo "docker is not on PATH" >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is not on PATH" >&2; exit 1; }

[[ -f "${TARGET_MODEL}/config.json" ]] || {
  echo "target checkpoint is incomplete: ${TARGET_MODEL}" >&2
  exit 1
}
[[ -f "${DRAFT_MODEL}/config.json" ]] || {
  echo "DFlash2 draft is incomplete: ${DRAFT_MODEL}" >&2
  exit 1
}
SERVER_ARGS=(
  sglang serve
  --trust-remote-code
  --model-path "${TARGET_MODEL}"
  --kv-cache-dtype bfloat16
  --mem-fraction-static 0.80
  --attention-backend flashinfer
  --chunked-prefill-size 2048
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --mamba-full-memory-ratio 3.04
  --host 0.0.0.0
  --port "${PORT}"
  --speculative-algorithm DFLASH
  --speculative-draft-model-path "${DRAFT_MODEL}"
  --speculative-num-draft-tokens 8
  --mamba-radix-cache-strategy extra_buffer
  --mamba-ssm-dtype bfloat16
  --max-running-requests 10
  --max-mamba-cache-size 50
  --context-length 262144
  --served-model-name "${SERVED_MODEL_NAME}"
)

# Never allow the old appended/last-wins launch pattern to return.
for critical_flag in \
  --model-path \
  --mem-fraction-static \
  --mamba-radix-cache-strategy \
  --max-running-requests \
  --max-mamba-cache-size \
  --speculative-algorithm \
  --speculative-num-draft-tokens; do
  count=0
  for arg in "${SERVER_ARGS[@]}"; do
    [[ "${arg}" == "${critical_flag}" ]] && ((count += 1))
  done
  (( count == 1 )) || {
    echo "launch preflight failed: ${critical_flag} occurs ${count} times" >&2
    exit 1
  }
done

if [[ "${PRINT_SERVER_ARGS:-0}" == "1" ]]; then
  printf '%q ' "${SERVER_ARGS[@]}"
  printf '\n'
  exit 0
fi

if [[ "${PROFILE}" == "abliterated" ]]; then
  readonly ABLITERATED_REVISION="36276309d30211d9babf72f22d1605dc2dc6357c"
  readonly ABLITERATED_READY_MARKER="${TARGET_MODEL}/.sglang-checkpoint-ready"
  if [[ ! -f "${ABLITERATED_READY_MARKER}" ]] \
    || ! grep -qx "revision=${ABLITERATED_REVISION}" "${ABLITERATED_READY_MARKER}"; then
    echo "abliterated checkpoint has not passed integrity verification" >&2
    echo "run: ${ROOT_DIR}/install-abliterated.sh" >&2
    exit 1
  fi
fi

docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
  echo "required image is missing: ${IMAGE}" >&2
  echo "install it with: docker pull ${IMAGE}" >&2
  exit 1
}

mkdir -p "${STATE_DIR}" "${ROOT_DIR}/.cache/huggingface" "${ROOT_DIR}/.cache/triton"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  current_target="$({ docker inspect -f '{{range .Config.Cmd}}{{println .}}{{end}}' "${CONTAINER_NAME}" || true; } \
    | awk 'previous == "--model-path" { print; exit } { previous = $0 }')"
  if [[ "${current_target}" == "${TARGET_MODEL}" ]]; then
    printf '%s\n' "${PROFILE}" > "${STATE_DIR}/active-profile"
    echo "${PROFILE} instance is already running"
    echo "OpenAI base URL: http://127.0.0.1:${PORT}/v1"
    exit 0
  fi
  echo "Switching active instance from ${current_target:-unknown} to ${TARGET_MODEL}"
  docker stop "${CONTAINER_NAME}" >/dev/null
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

# DGX Spark GPU memory is unified with the host. Starting a second large model
# can kill the desktop even though Docker reports OOMKilled=false, so gate the
# launch using the host's real available-memory counter as the SGLang recipe
# requires. Override only for deliberate diagnostics.
readonly MIN_MEM_AVAILABLE_GIB="${QWEN_MIN_MEM_AVAILABLE_GIB:-105}"
mem_available_kib="$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo)"
min_mem_available_kib="$((MIN_MEM_AVAILABLE_GIB * 1024 * 1024))"
if (( mem_available_kib < min_mem_available_kib )); then
  printf 'launch refused: MemAvailable is %.1f GiB; Qwen requires at least %s GiB before startup\n' \
    "$(awk -v kib="${mem_available_kib}" 'BEGIN { printf "%.1f", kib / 1024 / 1024 }')" \
    "${MIN_MEM_AVAILABLE_GIB}" >&2
  echo "Stop the active large-model container first, then retry." >&2
  exit 1
fi

: > "${LOG_FILE}"
container_id="$(docker run -d \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:${PORT}" \
  --ipc host \
  --device nvidia.com/gpu=all \
  --shm-size 32g \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e HF_HUB_OFFLINE=1 \
  -v "${ROOT_DIR}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${ROOT_DIR}/.cache/triton:/root/.triton" \
  -v /models:/models:ro \
  "${IMAGE}" \
  "${SERVER_ARGS[@]}")"

printf '%s\n' "${container_id}" > "${STATE_DIR}/active-container"
printf '%s\n' "${PROFILE}" > "${STATE_DIR}/active-profile"
echo "Starting ${PROFILE} instance (${container_id:0:12}); startup normally takes 3-5 minutes"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

docker logs -f "${CONTAINER_NAME}" 2>&1 \
  | tee -a "${LOG_FILE}" \
  | grep --line-buffered -v "Enabled fused SiLU+mul+FP4-quant for dense MLP down_proj input" &
log_follow_pid=$!

for attempt in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "${PROFILE} instance is ready"
    echo "OpenAI base URL: http://127.0.0.1:${PORT}/v1"
    echo "Switchyard model: ${SERVED_MODEL_NAME}"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "SGLang exited before readiness; last 120 log lines:" >&2
    tail -n 120 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  (( attempt % 6 == 0 )) && echo "  still starting..."
  sleep 5
done

echo "startup timed out; inspect ${LOG_FILE}" >&2
exit 1
