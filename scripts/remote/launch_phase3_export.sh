#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "usage: $0 [seed] [trl]" >&2
  exit 2
fi

seed="${1:-0}"
backend="${2:-trl}"
config="configs/r1b_sft_rule_20k.yaml"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "full Phase 3 exports are Linux CUDA pod-only" >&2
  exit 2
fi
if [[ "${seed}" != "0" || "${backend}" != "trl" ]]; then
  echo "Phase 3.1 exports only the approved R1b seed-0 TRL adapter" >&2
  exit 2
fi
: "${FORGE_GPU_HOURLY_USD:?Set FORGE_GPU_HOURLY_USD to the actual pod USD/hour rate}"
if [[ ! -x .venv/bin/python ]]; then
  echo "run scripts/remote/bootstrap.sh first" >&2
  exit 2
fi
.venv/bin/python -c \
  'import sys; rate=float(sys.argv[1]); assert rate > 0, "hourly rate must be positive"' \
  "${FORGE_GPU_HOURLY_USD}"

gpu_processes="$(nvidia-smi pmon -c 1 | awk '$1 !~ /^#/ && $2 ~ /^[0-9]+$/ {print $1, $2, $3, $10}')"
if [[ -n "${gpu_processes//[[:space:]]/}" ]]; then
  echo "GPU is already in use; refusing to launch a shared-pod export task:" >&2
  echo "${gpu_processes}" >&2
  exit 3
fi

.venv/bin/python -m forge.train.preflight \
  --config "${config}" --seed "${seed}" --backend "${backend}" --launch

session="forge-export-r1b-${backend}-s${seed}"
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  exit 2
fi
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
launch_env=(
  "FORGE_STARTED_AT=${started_at}"
  "FORGE_GPU_HOURLY_USD=${FORGE_GPU_HOURLY_USD}"
  "HF_HUB_OFFLINE=1"
  "TRANSFORMERS_OFFLINE=1"
  "PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7}"
)
for name in \
  http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY \
  REQUESTS_CA_BUNDLE SSL_CERT_FILE HF_HUB_DISABLE_XET; do
  if [[ -n "${!name:-}" ]]; then
    launch_env+=("${name}=${!name}")
  fi
done
tmux new-session -d -s "${session}" \
  env "${launch_env[@]}" \
  ./scripts/remote/run_phase3_export.sh "${config}" "${seed}" "${backend}"

echo "launched R1b BF16 + GPTQ export in tmux session ${session}"
echo "started_at=${started_at} (UTC)"
echo "actual_rate_usd_per_gpu_hour=${FORGE_GPU_HOURLY_USD}"
