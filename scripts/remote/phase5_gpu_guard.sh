#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Phase 5 GPU guard is remote Linux only" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
pid_path="results/phase5/runtime/vllm.pid"
if [[ ! -f "${pid_path}" ]]; then
  echo "Phase 5 vLLM PID receipt is missing" >&2
  exit 2
fi
expected_pid="$(tr -d '[:space:]' < "${pid_path}")"
if [[ ! "${expected_pid}" =~ ^[0-9]+$ ]] || ! kill -0 "${expected_pid}" 2>/dev/null; then
  echo "Phase 5 vLLM PID receipt is stale" >&2
  exit 2
fi

foreign="$(nvidia-smi pmon -c 1 | awk -v expected="${expected_pid}" \
  '$1 !~ /^#/ && $2 ~ /^[0-9]+$/ && $2 != expected {print $1, $2, $3, $10}')"
if [[ -n "${foreign//[[:space:]]/}" ]]; then
  echo "Foreign GPU process detected; refusing the Phase 5 launch:" >&2
  echo "${foreign}" >&2
  exit 3
fi

echo "GPU guard passed; only the recorded Phase 5 vLLM process uses the GPU"
