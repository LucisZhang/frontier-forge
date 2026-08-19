#!/usr/bin/env bash
set -euo pipefail

session="forge-phase5-gateway"
if ! tmux has-session -t "${session}" 2>/dev/null; then
  echo "Phase 5 gateway session is not running"
  exit 0
fi
tmux send-keys -t "${session}" C-c
for _ in $(seq 1 60); do
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    echo "Phase 5 gateway stopped cleanly"
    exit 0
  fi
  sleep 1
done
echo "Phase 5 gateway did not stop after SIGINT; leaving the session untouched" >&2
exit 1
