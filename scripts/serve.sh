#!/usr/bin/env bash
# Serve Llama-3.2-1B + thesis LoRA adapters via vLLM's OpenAI-compatible server.
#
# Usage:  bash scripts/serve.sh
# Env overrides:
#   MODEL_ID     base model             (default: meta-llama/Llama-3.2-1B-Instruct)
#   ADAPTER_DIR  dir of LoRA adapters   (default: assets/adapters)
#   PORT         server port            (default: 8000)
#
# Every subdirectory of ADAPTER_DIR containing an adapter_config.json is served
# as its own model, named "mi-coach-<subdir>" — e.g. assets/adapters/pto-iter10
# becomes model "mi-coach-pto-iter10" (the thesis default; grpo-iter8 optional).
# Requires HF_TOKEN in the environment for the gated Llama download (first run only).

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-assets/adapters}"
PORT="${PORT:-8000}"

# WSL2: vLLM leaves pinned memory off by default and then fails with "UVA is
# not available"; modern WSL2 kernels (>= 4.19.121) support it, so opt in.
export VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}"

# No-sudo FFmpeg fallback for torchcodec (see scripts/setup_local_toolchain.sh).
if [[ -d .venv/ffmpeg-libs ]]; then
  export LD_LIBRARY_PATH="$PWD/.venv/ffmpeg-libs:$PWD/.venv/lib/python3.12/site-packages/av.libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# No-sudo C compiler for Triton/Inductor JIT (zig cc shim; see README setup).
if [[ -z "${CC:-}" && -x .venv/bin/zigcc ]] && ! command -v cc >/dev/null && ! command -v gcc >/dev/null; then
  export CC="$PWD/.venv/bin/zigcc"
  export CXX="$PWD/.venv/bin/zigcc"
fi

LORA_ARGS=()
MODULES=()
for dir in "$ADAPTER_DIR"/*/; do
  [[ -f "$dir/adapter_config.json" ]] || continue
  name="mi-coach-$(basename "$dir")"
  MODULES+=("$name=$dir")
done
if ((${#MODULES[@]})); then
  LORA_ARGS=(--enable-lora --lora-modules "${MODULES[@]}")
  echo "Serving LoRA adapters: ${MODULES[*]}" >&2
else
  echo "WARNING: no LoRA adapters under $ADAPTER_DIR — serving base model only." >&2
fi

exec .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --port "$PORT" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  "${LORA_ARGS[@]}"
