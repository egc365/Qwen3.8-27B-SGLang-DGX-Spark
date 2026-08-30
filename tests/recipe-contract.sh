#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command_line="$(PRINT_SERVER_ARGS=1 "${ROOT_DIR}/lib/serve-dflash.sh" abliterated)"

require() {
  [[ " ${command_line} " == *" $1 "* ]] || {
    echo "recipe contract failed: missing $1" >&2
    exit 1
  }
}

reject() {
  [[ " ${command_line} " != *" $1 "* ]] || {
    echo "recipe contract failed: forbidden $1" >&2
    exit 1
  }
}

require /models/Huihui-Qwen3.8-27B-abliterated-NVFP4-36276309
require --mem-fraction-static
require 0.80
require --chunked-prefill-size
require 2048
require --context-length
require 262144
require --mamba-radix-cache-strategy
require extra_buffer
require --mamba-full-memory-ratio
require 3.04
require --max-running-requests
require 10
require --max-mamba-cache-size
require 50
require --speculative-algorithm
require DFLASH
require --speculative-num-draft-tokens
require 8
require --reasoning-parser
require qwen3
require --tool-call-parser
require qwen3_coder
reject 0.90
reject 8192
require --kv-cache-dtype
require bfloat16
reject fp8_e4m3
reject --disable-prefill-cuda-graph

echo "Qwen DGX Spark recipe contract passed."
