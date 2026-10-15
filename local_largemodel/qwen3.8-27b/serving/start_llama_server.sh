#!/usr/bin/env bash
# 启动 qwen3.8-27b 的 OpenAI 兼容本地服务（llama.cpp）。
# 与当前机器上正在运行的实例参数保持一致：单卡/多卡 GPU 全量 offload、8K 上下文、视觉投影。
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/mnt/workspace/yangt/vlm/phonevlm/models/Qwen3.8-27B}"
GGUF="${GGUF:-$MODEL_ROOT/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
MMPROJ="${MMPROJ:-$MODEL_ROOT/gguf/mmproj-F16.gguf}"
PORT="${PORT:-8084}"
LLAMA_SERVER="${LLAMA_SERVER:-/root/.unsloth/llama.cpp/build/bin/llama-server}"

exec "$LLAMA_SERVER" \
  -m "$GGUF" \
  --mmproj "$MMPROJ" \
  -c "${CTX:-8192}" \
  -ngl "${NGL:-99}" \
  --port "$PORT" --host 127.0.0.1 \
  -t "${THREADS:-16}" \
  --image-min-tokens "${IMAGE_MIN_TOKENS:-1024}"
