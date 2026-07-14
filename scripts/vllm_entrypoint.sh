#!/usr/bin/env bash
# Container entrypoint for the vllm service: registers every adapter found
# under /assets/adapters (symlinks created by scripts/link_adapters.sh resolve
# because the whole ./assets tree is mounted) and starts the OpenAI server.

set -euo pipefail

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B}"

MODULES=()
for dir in /assets/adapters/*/; do
  [[ -f "$dir/adapter_config.json" ]] || continue
  MODULES+=("mi-coach-$(basename "$dir")=$dir")
done
TEMPLATE=$(ls /assets/adapters/*/chat_template.jinja 2>/dev/null | head -1)

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-loras 4 \
  --max-cpu-loras 24 \
  --enable-lora \
  --lora-modules "${MODULES[@]}" \
  ${TEMPLATE:+--chat-template "$TEMPLATE"}
