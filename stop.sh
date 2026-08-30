#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="qwen3.8-27b-sglang"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping active Qwen instance..."
  docker stop "${CONTAINER_NAME}"
elif docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Qwen SGLang container is already stopped."
else
  echo "Qwen SGLang container does not exist."
fi

rm -f "${SCRIPT_DIR}/state/active-container" "${SCRIPT_DIR}/state/active-profile"
echo "No Qwen SGLang instance is running; the stopped container was preserved."
