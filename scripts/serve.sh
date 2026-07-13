#!/usr/bin/env bash
# Serve Llama-3.2-1B (+ thesis LoRA adapter) via vLLM's OpenAI-compatible server.
#
# Usage:  bash scripts/serve.sh
# Env overrides:
#   MODEL_ID   base model            (default: meta-llama/Llama-3.2-1B-Instruct)
#   ADAPTER    LoRA adapter dir      (default: assets/adapter)
#   PORT       server port           (default: 8000)
#
# Requires HF_TOKEN in the environment for the gated Llama download (first run only).

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B-Instruct}"
ADAPTER="${ADAPTER:-assets/adapter}"
PORT="${PORT:-8000}"

# WSL2: vLLM leaves pinned memory off by default and then fails with "UVA is
# not available"; modern WSL2 kernels (>= 4.19.121) support it, so opt in.
export VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}"

# No-sudo FFmpeg fallback for torchcodec (see scripts/setup_ffmpeg_libs.sh).
if [[ -d .venv/ffmpeg-libs ]]; then
  export LD_LIBRARY_PATH="$PWD/.venv/ffmpeg-libs:$PWD/.venv/lib/python3.12/site-packages/av.libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

LORA_ARGS=()
if [[ -f "$ADAPTER/adapter_config.json" ]]; then
  # Adapter is exposed as model name "mi-coach"; base model stays available under its own name.
  LORA_ARGS=(--enable-lora --lora-modules "mi-coach=$ADAPTER")
else
  echo "WARNING: no LoRA adapter at $ADAPTER — serving base model only." >&2
fi

exec .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --port "$PORT" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  "${LORA_ARGS[@]}"
