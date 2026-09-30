#!/usr/bin/env bash
set -euo pipefail

readonly REPO="sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4"
readonly REVISION="36276309d30211d9babf72f22d1605dc2dc6357c"
readonly DESTINATION="/models/Huihui/Qwen3.8-27B-abliterated-NVFP4"
readonly READY_MARKER="${DESTINATION}/.sglang-checkpoint-ready"

command -v hf >/dev/null 2>&1 || {
  echo "Hugging Face CLI (hf) is not on PATH" >&2
  exit 1
}
command -v sha256sum >/dev/null 2>&1 || {
  echo "sha256sum is not on PATH" >&2
  exit 1
}

echo "Downloading revision ${REVISION} into ${DESTINATION}"
HF_XET_HIGH_PERFORMANCE=1 hf download "${REPO}" \
  --revision "${REVISION}" \
  --local-dir "${DESTINATION}"

verify_weight() {
  local filename="$1"
  local expected_size="$2"
  local expected_sha256="$3"
  local path="${DESTINATION}/${filename}"
  local actual_size

  [[ -f "${path}" ]] || {
    echo "checkpoint verification failed: missing ${path}" >&2
    exit 1
  }
  actual_size="$(stat -c '%s' "${path}")"
  [[ "${actual_size}" == "${expected_size}" ]] || {
    echo "checkpoint verification failed: ${filename} is ${actual_size} bytes; expected ${expected_size}" >&2
    exit 1
  }
  printf '%s  %s\n' "${expected_sha256}" "${path}" | sha256sum --check --status || {
    echo "checkpoint verification failed: SHA-256 mismatch for ${filename}" >&2
    exit 1
  }
}

verify_weight \
  model.safetensors \
  19709883808 \
  cb595ed2770d612737c1114dcdc79c191925eacafdf0b5c90ecdc4cf982ae058
verify_weight \
  model-mtp-bf16.safetensors \
  849400424 \
  90fa0e3eed5a647c035c6df9ecabc416c0f8d573ff84ac12485b085f00a7cdf2

printf 'repo=%s\nrevision=%s\n' "${REPO}" "${REVISION}" > "${READY_MARKER}"
echo "Verified abliterated checkpoint is ready."
