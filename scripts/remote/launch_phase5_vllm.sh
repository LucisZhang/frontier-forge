#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "usage: FORGE_GPU_HOURLY_USD=0.30 FORGE_BENCH_GIT_SHA=<sha> $0" >&2
  exit 2
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Phase 5 vLLM launch is remote Linux-only" >&2
  exit 2
fi
: "${FORGE_GPU_HOURLY_USD:?Set FORGE_GPU_HOURLY_USD=0.30}"
: "${FORGE_BENCH_GIT_SHA:?Set FORGE_BENCH_GIT_SHA to the benchmark implementation SHA}"
if [[ "${FORGE_GPU_HOURLY_USD}" != "0.30" ]]; then
  echo "Phase 5 pins FORGE_GPU_HOURLY_USD=0.30" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
if [[ ! -x .venv-phase4/bin/vllm || ! -x .venv-phase4/bin/python ]]; then
  echo "the existing in-repo Phase 4 serving environment is required" >&2
  exit 2
fi
gpu_processes="$(nvidia-smi pmon -c 1 | awk \
  '$1 !~ /^#/ && $2 ~ /^[0-9]+$/ {print $1, $2, $3, $10}')"
if [[ -n "${gpu_processes//[[:space:]]/}" ]]; then
  echo "GPU is occupied; refusing to launch the shared-pod Phase 5 server:" >&2
  echo "${gpu_processes}" >&2
  exit 3
fi
session="forge-phase5-vllm"
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  exit 2
fi

mkdir -p \
  results/phase5/logs \
  results/phase5/runtime \
  .tmp-phase5 \
  .cache-phase5/cuda \
  .cache-phase5/cupy \
  .cache-phase5/numba \
  .cache-phase5/torch-extensions \
  .cache-phase5/vllm \
  .cache-phase5/vllm-config \
  .cache-phase5/huggingface
mapfile -d '' command < <(.venv-phase4/bin/python -m forge.bench.server_args \
  --config configs/phase4/spec_r1b_bf16_mtp.yaml \
  --executable .venv-phase4/bin/vllm --null)
started_at="${FORGE_POD_STARTED_AT:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
log_path="results/phase5/logs/vllm-${started_at//:/}.log"
tmux new-session -d -s "${session}" \
  env \
  "FORGE_GPU_HOURLY_USD=${FORGE_GPU_HOURLY_USD}" \
  "FORGE_BENCH_GIT_SHA=${FORGE_BENCH_GIT_SHA}" \
  "TMPDIR=${repo_root}/.tmp-phase5" \
  "XDG_CACHE_HOME=${repo_root}/.cache-phase5" \
  "VLLM_CACHE_ROOT=${repo_root}/.cache-phase5/vllm" \
  "VLLM_CONFIG_ROOT=${repo_root}/.cache-phase5/vllm-config" \
  "CUDA_CACHE_PATH=${repo_root}/.cache-phase5/cuda" \
  "CUPY_CACHE_DIR=${repo_root}/.cache-phase5/cupy" \
  "NUMBA_CACHE_DIR=${repo_root}/.cache-phase5/numba" \
  "TORCH_EXTENSIONS_DIR=${repo_root}/.cache-phase5/torch-extensions" \
  "HF_HOME=${repo_root}/.cache-phase5/huggingface" \
  "HUGGINGFACE_HUB_CACHE=${repo_root}/.cache-phase5/huggingface/hub" \
  "HF_HUB_OFFLINE=1" \
  "FORGE_LOG_PATH=${repo_root}/${log_path}" \
  bash -lc 'exec "$@" >"${FORGE_LOG_PATH}" 2>&1' _ "${command[@]}"

for _ in $(seq 1 180); do
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    echo "Phase 5 vLLM exited before becoming ready" >&2
    tail -n 120 "${log_path}" >&2
    exit 1
  fi
  if .venv-phase4/bin/python - <<'PY' >/dev/null 2>&1
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
    assert response.status == 200
PY
  then
    break
  fi
  sleep 2
done
if ! .venv-phase4/bin/python - <<'PY' >/dev/null 2>&1
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
    assert response.status == 200
PY
then
  echo "Phase 5 vLLM health timeout" >&2
  tail -n 120 "${log_path}" >&2
  exit 1
fi
vllm_pid="$(nvidia-smi pmon -c 1 | awk '$1 !~ /^#/ && $2 ~ /^[0-9]+$/ {print $2; exit}')"
if [[ -z "${vllm_pid}" ]]; then
  echo "vLLM is healthy but its GPU PID was not observed" >&2
  exit 1
fi
printf '%s\n' "${vllm_pid}" > results/phase5/runtime/vllm.pid
printf '%s\n' "${started_at}" > results/phase5/runtime/pod_started_at
echo "Phase 5 R1b native-MTP vLLM ready; pod remains running"
