#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ( "$1" != "profile" && "$1" != "release" ) ]]; then
  echo "usage: $0 profile|release [profile-label]" >&2
  exit 2
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Phase 5 gateway launch is remote Linux-only" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
./scripts/remote/phase5_gpu_guard.sh
mode="$1"
label="${2:-release}"
binary="gateway/build/remote-${mode}/forge_gateway"
if [[ ! -x "${binary}" ]]; then
  echo "Phase 5 gateway binary is missing: ${binary}" >&2
  exit 2
fi
session="forge-phase5-gateway"
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  exit 2
fi
mkdir -p results/phase5/logs results/phase5/profile results/phase5/runtime
log_path="results/phase5/logs/gateway-${label}.log"
profile_env=()
if [[ "${mode}" == "profile" ]]; then
  profile_env+=("GMON_OUT_PREFIX=${repo_root}/results/phase5/profile/gmon-${label}")
fi
tmux new-session -d -s "${session}" \
  env "${profile_env[@]}" "FORGE_LOG_PATH=${repo_root}/${log_path}" \
  bash -lc 'exec "$@" >"${FORGE_LOG_PATH}" 2>&1' _ \
  "${binary}" \
  --listen-host 127.0.0.1 --listen-port 9000 \
  --io-threads 4 \
  --primary-host 127.0.0.1 --primary-port 8000 \
  --primary-model forge-r1b-bf16-native-mtp \
  --pool-size 64 \
  --primary-token-capacity 24000 \
  --max-queue-requests 24 --max-queue-tokens 48000 \
  --request-timeout-ms 60000 \
  --health-interval-ms 1000 \
  --disable-fallback
gateway_pid="$(tmux list-panes -t "${session}" -F '#{pane_pid}')"
printf '%s\n' "${gateway_pid}" > results/phase5/runtime/gateway.pid
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:9000/readyz >/dev/null 2>&1; then
    echo "Phase 5 gateway ready in ${session}"
    exit 0
  fi
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    echo "Phase 5 gateway exited before becoming ready" >&2
    tail -n 120 "${log_path}" >&2
    exit 1
  fi
  sleep 1
done
echo "Phase 5 gateway health timeout" >&2
exit 1
