# Qwen3.8-27B SGLang on DGX Spark

There are exactly two launch choices. They use one shared serving slot because
a single instance consumes almost all of the Spark's unified memory.

```bash
./install-abliterated.sh
./start-original.sh
./start-abliterated.sh
./stop.sh
```

Starting one profile automatically replaces the other. Re-running the active
profile is idempotent.

Both choices use the same fixed DGX Spark DFlash2 recipe:

- SGLang image: `lmsysorg/sglang:dev-qwen38-27b-dflash2`
- draft: `/models/Qwen3.8-27B-DFlash2`
- memory fraction `0.80`, chunk size `2048`, native context `262144`
- BF16 KV, BF16 GDN, low-latency `extra_buffer`, 8 draft tokens
- requested concurrency `10`, Mamba state pool `50` (10 × 5 slots)
- launch guard: at least `105 GiB` host `MemAvailable` before loading

Only the target checkpoint changes:

| Command | Target |
|---|---|
| `./start-original.sh` | `/models/Qwen3.8-27B-NVFP4` |
| `./start-abliterated.sh` | `/models/Huihui-Qwen3.8-27B-abliterated-NVFP4-36276309` |

The active instance is available at `http://127.0.0.1:8888/v1` and through
Switchyard as `qwen3.8-27b-sglang`. Runtime logs and the active-profile marker
live under `state/` and are ignored by git.

The abliterated target is the measured BF16-head NVFP4 checkpoint from
`sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4`, pinned to revision
`36276309d30211d9babf72f22d1605dc2dc6357c`. The installer checks the exact
published sizes and SHA-256 hashes before creating the readiness marker that
the launcher requires. The older custom ModelOpt export is deliberately not a
launch target because it produced corrupt arithmetic and tool output with this
official SGLang image, even when speculative decoding was disabled.

The one intentional Playground override is BF16 KV cache. Both local NVFP4
exports declare FP8 KV but contain no dedicated KV calibration tensors. With
FP8 selected, SGLang substitutes scale `1.0` and warns that results may be less
accurate. BF16 KV removes that fallback; it does not alter the abliterated
weights. The matching balanced Mamba ratio is `3.04`.

## Start, call, and stop

```bash
# Install the exact SGLang image once.
docker pull lmsysorg/sglang:dev-qwen38-27b-dflash2

# Install and verify the revision-pinned abliterated checkpoint once.
./install-abliterated.sh

# Start the abliterated checkpoint and wait for readiness.
./start-abliterated.sh

# Inspect the exact inner SGLang command without launching it.
PRINT_SERVER_ARGS=1 ./lib/serve-dflash.sh abliterated

# Call SGLang directly.
curl http://127.0.0.1:8888/v1/models

# Call the same server through Switchyard.
curl http://127.0.0.1:4000/v1/models

# Run the lightweight governed harness. Its tool/path permissions live in
# SQLite, and its system prompt is intentionally short.
python3 /Ai-workshop/qwen/qwen38_harness.py run-task "your task"

# Qwen Code is available for comparison, but even --safe-mode currently sends
# a roughly 24K-token built-in harness prompt before the user's task.
qwen --safe-mode --model qwen3.8-27b-sglang

# Stop and remove the model container.
./stop.sh
```

`--max-mamba-cache-size 50` means 50 GDN state slots (10 requests × 5),
not 50 GiB. The DGX Spark recipe's actual static-memory ceiling is controlled
by `--mem-fraction-static 0.80`.

The governed harness configuration is stored at
`/Ai-workshop/harness-experiment/qwen38_harness.db`. The active endpoint is
`http://127.0.0.1:4000/v1`, the model slug is `qwen3.8-27b-sglang`, and the
repository is currently registered as a READ-only root. Add WRITE or RUN roots
through the harness CLI instead of embedding permissions in a system prompt.

There is no `.env`, experimental flag hatch, MTP launcher, or DSpark launcher.
The `patch/` directory is retained only to rebuild the required DFlash2 Docker
image if that image is ever removed.
