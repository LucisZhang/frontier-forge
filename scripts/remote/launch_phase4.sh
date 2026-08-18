#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "usage: FORGE_GPU_HOURLY_USD=0.30 FORGE_BENCH_GIT_SHA=<sha> $0" >&2
  exit 2
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "full Phase 4 launches are Linux CUDA pod-only" >&2
  exit 2
fi
: "${FORGE_GPU_HOURLY_USD:?Set FORGE_GPU_HOURLY_USD=0.30}"
: "${FORGE_BENCH_GIT_SHA:?Set FORGE_BENCH_GIT_SHA to the pushed implementation SHA}"
if [[ "${FORGE_GPU_HOURLY_USD}" != "0.30" ]]; then
  echo "this session pins FORGE_GPU_HOURLY_USD=0.30" >&2
  exit 2
fi
if [[ ! -x .venv-phase4/bin/python ]]; then
  echo "run scripts/remote/bootstrap_phase4.sh first" >&2
  exit 2
fi

gpu_processes="$(nvidia-smi pmon -c 1 | awk '$1 !~ /^#/ && $2 ~ /^[0-9]+$/ {print $1, $2, $3, $10}')"
if [[ -n "${gpu_processes//[[:space:]]/}" ]]; then
  echo "GPU is already in use; refusing to launch the shared-pod Phase 4 task:" >&2
  echo "${gpu_processes}" >&2
  exit 3
fi

session="forge-phase4"
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  exit 2
fi

repo_root="$(pwd)"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
mkdir -p \
  .cache/cuda \
  .cache/cupy \
  .cache/numba \
  .cache/torch-extensions \
  .cache/vllm \
  .cache/vllm-config
tmux new-session -d -s "${session}" \
  env \
  "FORGE_STARTED_AT=${started_at}" \
  "FORGE_GPU_HOURLY_USD=${FORGE_GPU_HOURLY_USD}" \
  "FORGE_BENCH_GIT_SHA=${FORGE_BENCH_GIT_SHA}" \
  "UV_CACHE_DIR=${repo_root}/.uv-cache-phase4" \
  "TMPDIR=${repo_root}/.tmp-phase4" \
  "XDG_CACHE_HOME=${repo_root}/.cache" \
  "VLLM_CACHE_ROOT=${repo_root}/.cache/vllm" \
  "VLLM_CONFIG_ROOT=${repo_root}/.cache/vllm-config" \
  "CUDA_CACHE_PATH=${repo_root}/.cache/cuda" \
  "CUPY_CACHE_DIR=${repo_root}/.cache/cupy" \
  "NUMBA_CACHE_DIR=${repo_root}/.cache/numba" \
  "TORCH_EXTENSIONS_DIR=${repo_root}/.cache/torch-extensions" \
  "HF_HOME=${repo_root}/.cache/huggingface" \
  "HUGGINGFACE_HUB_CACHE=${repo_root}/.cache/huggingface/hub" \
  "HF_HUB_OFFLINE=1" \
  ./scripts/remote/run_phase4.sh

echo "launched Phase 4 in tmux session ${session}"
echo "started_at=${started_at} (UTC)"
echo "actual_rate_usd_per_gpu_hour=${FORGE_GPU_HOURLY_USD}"
echo "the pod remains running"
