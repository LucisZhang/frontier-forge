#!/usr/bin/env bash
set -euo pipefail

: "${FORGE_GPU_HOURLY_USD:?Set FORGE_GPU_HOURLY_USD=1.53}"
: "${FORGE_BENCH_GIT_SHA:?Set FORGE_BENCH_GIT_SHA to the benchmark commit}"
if [[ "${FORGE_GPU_HOURLY_USD}" != "1.53" ]]; then
  echo "Phase 7.1 pins FORGE_GPU_HOURLY_USD=1.53" >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
if [[ "$(git rev-parse HEAD)" != "${FORGE_BENCH_GIT_SHA}" ]]; then
  echo "benchmark SHA differs from checked-out source" >&2
  exit 2
fi
if [[ ! -x .venv-phase4/bin/vllm ]]; then
  echo "run bootstrap_phase7_1_a10.sh first" >&2
  exit 2
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; then
  echo "A10 is occupied; refusing to launch vLLM" >&2
  exit 3
fi
session=forge-phase7-1-vllm
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  exit 2
fi
cache_root="${FORGE_CACHE_ROOT:-/mnt/frontier-forge/cache}"
mkdir -p results/phase7_1/logs results/phase7_1/runtime "${cache_root}"/{cuda,cupy,numba,torch-extensions,vllm,vllm-config,huggingface,tmp}
mapfile -d '' command < <(.venv-phase4/bin/python -m forge.bench.server_args \
  --config configs/phase4/spec_r1b_bf16_mtp.yaml \
  --executable .venv-phase4/bin/vllm --null)
vm_started_at="${FORGE_VM_STARTED_AT:-$(date -u -d "$(uptime -s)" +%Y-%m-%dT%H:%M:%SZ)}"
vllm_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
log_path="results/phase7_1/logs/vllm-${vllm_started_at//:/}.log"
tmux new-session -d -s "${session}" \
  env \
  "TMPDIR=${cache_root}/tmp" \
  "XDG_CACHE_HOME=${cache_root}" \
  "VLLM_CACHE_ROOT=${cache_root}/vllm" \
  "VLLM_CONFIG_ROOT=${cache_root}/vllm-config" \
  "CUDA_CACHE_PATH=${cache_root}/cuda" \
  "CUPY_CACHE_DIR=${cache_root}/cupy" \
  "NUMBA_CACHE_DIR=${cache_root}/numba" \
  "TORCH_EXTENSIONS_DIR=${cache_root}/torch-extensions" \
  "HF_HOME=${cache_root}/huggingface" \
  "HUGGINGFACE_HUB_CACHE=${cache_root}/huggingface/hub" \
  "HF_HUB_OFFLINE=1" \
  "FORGE_LOG_PATH=${repo_root}/${log_path}" \
  bash -lc 'exec "$@" >"${FORGE_LOG_PATH}" 2>&1' _ "${command[@]}"

for _ in $(seq 1 450); do
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    echo "Phase 7.1 vLLM exited before becoming ready" >&2
    tail -n 160 "${log_path}" >&2
    exit 1
  fi
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:8000/health >/dev/null
for _ in $(seq 1 30); do
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' | sed '/^$/d' | sort -nu > results/phase7_1/runtime/vllm_gpu_pids
  [[ -s results/phase7_1/runtime/vllm_gpu_pids ]] && break
  sleep 1
done
if [[ ! -s results/phase7_1/runtime/vllm_gpu_pids ]]; then
  echo "vLLM is healthy but no GPU PID was observed" >&2
  exit 1
fi
jq -n \
  --arg vm_started_at "${vm_started_at}" \
  --arg vllm_started_at "${vllm_started_at}" \
  --arg hourly_usd "${FORGE_GPU_HOURLY_USD}" \
  --arg git_sha "${FORGE_BENCH_GIT_SHA}" \
  --arg log_path "${log_path}" \
  '{vm_started_at:$vm_started_at,vllm_started_at:$vllm_started_at,hourly_usd:$hourly_usd,git_sha:$git_sha,log_path:$log_path}' \
  > results/phase7_1/runtime/session.json
./scripts/remote/phase7_1_gpu_guard.sh
echo "Phase 7.1 A10 vLLM ready on 127.0.0.1:8000"
