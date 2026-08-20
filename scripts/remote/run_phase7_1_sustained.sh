#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "verify-artifact" && "$1" != "bare" && "$1" != "gateway" ) ]]; then
  echo "usage: FORGE_PHASE7_SESSION_STARTED_AT=<UTC ISO-8601> $0 verify-artifact|bare|gateway" >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
./scripts/remote/phase7_1_gpu_guard.sh
curl -fsS http://127.0.0.1:8000/health >/dev/null

session_receipt=results/phase7_1/runtime/delegated-phase7-session.json
mkdir -p "$(dirname "${session_receipt}")"
if [[ ! -f "${session_receipt}" ]]; then
  : "${FORGE_PHASE7_SESSION_STARTED_AT:?Set the delegated session start in UTC ISO-8601 form}"
  : "${FORGE_GPU_HOURLY_USD:?Set FORGE_GPU_HOURLY_USD=1.53}"
  if [[ "${FORGE_GPU_HOURLY_USD}" != "1.53" ]]; then
    echo "Gate 7.1 pins FORGE_GPU_HOURLY_USD=1.53" >&2
    exit 2
  fi
  jq -n \
    --arg started_at "${FORGE_PHASE7_SESSION_STARTED_AT}" \
    --arg hourly_usd "${FORGE_GPU_HOURLY_USD}" \
    --arg git_sha "$(git rev-parse HEAD)" \
    '{version:1,started_at:$started_at,hourly_usd:$hourly_usd,git_sha:$git_sha}' \
    > "${session_receipt}"
fi
export FORGE_PHASE7_SESSION_STARTED_AT="$(jq -r .started_at "${session_receipt}")"
export FORGE_GPU_HOURLY_USD="$(jq -r .hourly_usd "${session_receipt}")"
export FORGE_BENCH_GIT_SHA="$(jq -r .git_sha "${session_receipt}")"
if [[ "$(git rev-parse HEAD)" != "${FORGE_BENCH_GIT_SHA}" ]]; then
  echo "sustained benchmark SHA differs from checked-out source" >&2
  exit 2
fi
if (( $(ulimit -n) < 1024 )); then
  echo "sustained benchmark requires an open-file limit of at least 1024" >&2
  exit 2
fi

stage="$1"
if [[ "${stage}" == "bare" ]]; then
  if tmux has-session -t forge-phase7-1-gateway 2>/dev/null || ss -lntH 'sport = :9000' | grep -q .; then
    echo "all sustained bare-vLLM cells must finish before the gateway starts" >&2
    exit 2
  fi
elif [[ "${stage}" == "gateway" ]]; then
  curl -fsS http://127.0.0.1:9000/readyz >/dev/null
fi

.venv-phase4/bin/python -m gateway.bench.phase7_1_sustained --stage "${stage}"
if [[ "${stage}" == "gateway" ]]; then
  .venv-phase4/bin/python -m gateway.bench.phase7_1_sustained_report
fi
