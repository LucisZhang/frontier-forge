#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
pid_path=results/phase7_1/runtime/vllm_gpu_pids
if [[ ! -s "${pid_path}" ]]; then
  echo "Phase 7.1 vLLM GPU PID receipt is missing" >&2
  exit 2
fi
mapfile -t expected < <(sort -nu "${pid_path}")
mapfile -t observed < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' | sed '/^$/d' | sort -nu)
if [[ ${#expected[@]} -eq 0 || "${expected[*]}" != "${observed[*]}" ]]; then
  echo "GPU process set differs from the recorded exclusive vLLM set" >&2
  printf 'expected=%s\nobserved=%s\n' "${expected[*]}" "${observed[*]}" >&2
  exit 3
fi
for pid in "${expected[@]}"; do
  kill -0 "${pid}"
done
echo "GPU guard passed; only recorded Phase 7.1 vLLM processes use the A10"
