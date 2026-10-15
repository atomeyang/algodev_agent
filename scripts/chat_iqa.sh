#!/usr/bin/env bash
# One-shot launcher: open the algo-agent chat REPL attached to the IQA real-MOS task.
#
#   ./scripts/chat_iqa.sh              # attach to task iqa_mos
#   ALGO_TASK=iqa ./scripts/chat_iqa.sh   # attach to the pseudo track instead
#
# Fixes the two common traps: the right conda env (langgraph + torch) and the
# ALGO_STATE_DIR / PYTHONPATH needed to see the task and import algorithm_plugins.
set -euo pipefail
cd "$(dirname "$0")/.."

export ALGO_STATE_DIR="${ALGO_STATE_DIR:-/root/algo_agent_state_iqa_mos}"
export PYTHONPATH="runtimes/agent_runtime/src:${PYTHONPATH:-.}"
export IQA_BENCHMARK="${IQA_BENCHMARK:-koniq}"
export IQA_FEATURE_NPZ="${IQA_FEATURE_NPZ:-/root/iqa_assets/koniq_mobileclip2s0_256.npz}"

# Research/design + conversational model: DeepSeek-V4-Flash (from ./.env, with a
# safe default here so the model id is explicit even without a .env).
export ALGO_CLOUD_MODEL="${ALGO_CLOUD_MODEL:-deepseek-flash}"

ENV_BIN=/root/miniconda3/envs/algo_agent_runtime/bin
if [ ! -x "$ENV_BIN/algo-agent" ]; then
  ENV_BIN=/usr/local/bin
fi
exec "$ENV_BIN/algo-agent" --root . chat --task "${ALGO_TASK:-iqa_mos}" "$@"
