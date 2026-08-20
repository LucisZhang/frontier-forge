#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
./scripts/remote/phase7_1_gpu_guard.sh
binary=gateway/build/phase7-1-release/forge_gateway
if [[ ! -x "${binary}" ]]; then
  echo "run build_phase7_1_gateway.sh first" >&2
  exit 2
fi
session=forge-phase7-1-gateway
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  exit 2
fi
mkdir -p results/phase7_1/logs results/phase7_1/runtime
log_path=results/phase7_1/logs/gateway-a10.log
tmux new-session -d -s "${session}" \
  env "FORGE_LOG_PATH=${repo_root}/${log_path}" \
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
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:9000/readyz >/dev/null 2>&1; then
    echo "Phase 7.1 gateway ready on 127.0.0.1:9000"
    exit 0
  fi
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    tail -n 160 "${log_path}" >&2
    exit 1
  fi
  sleep 1
done
echo "Phase 7.1 gateway readiness timeout" >&2
exit 1
