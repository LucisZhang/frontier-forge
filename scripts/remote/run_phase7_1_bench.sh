#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "baseline" && "$1" != "gateway-matrix" && "$1" != "gateway-overload" ) ]]; then
  echo "usage: $0 baseline|gateway-matrix|gateway-overload" >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
./scripts/remote/phase7_1_gpu_guard.sh
session_receipt=results/phase7_1/runtime/session.json
if [[ ! -f "${session_receipt}" ]]; then
  echo "vLLM session receipt is missing" >&2
  exit 2
fi
export FORGE_GPU_HOURLY_USD="$(jq -r .hourly_usd "${session_receipt}")"
export FORGE_BENCH_GIT_SHA="$(jq -r .git_sha "${session_receipt}")"
export FORGE_VM_STARTED_AT="$(jq -r .vm_started_at "${session_receipt}")"
stage="$1"
if [[ "${stage}" == "baseline" ]]; then
  if tmux has-session -t forge-phase7-1-gateway 2>/dev/null || ss -lntH 'sport = :9000' | grep -q .; then
    echo "bare-vLLM baseline must finish before the gateway starts" >&2
    exit 2
  fi
else
  curl -fsS http://127.0.0.1:9000/readyz >/dev/null
fi
.venv-phase4/bin/python -m gateway.bench.phase7_1_bench --stage "${stage}"
if [[ "${stage}" == "gateway-overload" ]]; then
  .venv-phase4/bin/python -m gateway.bench.phase7_1_report
fi
