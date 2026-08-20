#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
for session in forge-phase7-1-gateway forge-phase7-1-vllm; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    tmux kill-session -t "${session}"
  fi
done
for _ in $(seq 1 60); do
  observed="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')"
  [[ -z "${observed}" ]] && break
  sleep 1
done
observed="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ -n "${observed}" ]]; then
  echo "GPU compute processes remain after service shutdown" >&2
  exit 1
fi
if ss -lntH 'sport = :8000 or sport = :9000' | grep -q .; then
  echo "a benchmark port is still listening" >&2
  exit 1
fi
jq -n \
  --arg stopped_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson gpu_compute_processes 0 \
  '{version:1,status:"complete",phase:"7.1",stopped_at:$stopped_at,gpu_compute_processes:$gpu_compute_processes,gateway_listening:false,vllm_listening:false}' \
  > results/phase7_1/service_shutdown.json
echo "Phase 7.1 vLLM and gateway stopped; A10 has no compute process"
