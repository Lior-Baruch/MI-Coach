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
