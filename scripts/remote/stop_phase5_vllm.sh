#!/usr/bin/env bash
set -euo pipefail

session="forge-phase5-vllm"
if ! tmux has-session -t "${session}" 2>/dev/null; then
  echo "Phase 5 vLLM session is not running"
  exit 0
fi
tmux send-keys -t "${session}" C-c
for _ in $(seq 1 120); do
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    echo "Phase 5 vLLM stopped cleanly; pod remains running"
    exit 0
  fi
  sleep 1
done
echo "Phase 5 vLLM did not stop after SIGINT; leaving the session untouched" >&2
exit 1
