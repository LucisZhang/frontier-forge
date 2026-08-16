#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 r0|r1|r1b|r2|r3|r4 [seed] [trl|unsloth|auto]" >&2
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit 2
fi

rung="$1"
seed="${2:-0}"
backend="${3:-auto}"

case "${rung}" in
  r0) config="configs/r0_base.yaml" ;;
  r1) config="configs/r1_sft_rule.yaml" ;;
  r1b) config="configs/r1b_sft_rule_20k.yaml" ;;
  r2) config="configs/r2_sft_distilled.yaml" ;;
  r3) config="configs/r3_dpo.yaml" ;;
  r4) config="configs/r4_grpo.yaml" ;;
  *) usage; exit 2 ;;
esac

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "full Phase 3 launches are Linux CUDA pod-only" >&2
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
if [[ "${backend}" == "auto" ]]; then
  backend="$(.venv/bin/python -c 'import json; print(json.load(open("results/phase3_backend_policy.json"))["default_backend"])')"
fi
if [[ "${backend}" != "trl" && "${backend}" != "unsloth" ]]; then
  usage
  exit 2
fi
if [[ "${rung}" == "r0" ]]; then
  backend="trl"
fi
train_python=".venv/bin/python"
if [[ "${backend}" == "unsloth" ]]; then
  train_python=".venv-unsloth/bin/python"
  if [[ ! -x "${train_python}" ]]; then
    echo "run scripts/remote/bootstrap.sh to create the locked Unsloth environment" >&2
    exit 2
  fi
fi

.venv/bin/python -m forge.train.preflight \
  --config "${config}" --seed "${seed}" --backend "${backend}" --launch

session="forge-${rung}-${backend}-s${seed}"
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  exit 2
fi
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
launch_env=(
  "FORGE_STARTED_AT=${started_at}"
  "FORGE_TRAIN_PYTHON=${train_python}"
  "FORGE_GPU_HOURLY_USD=${FORGE_GPU_HOURLY_USD}"
)
if [[ "${backend}" == "unsloth" ]]; then
  launch_env+=(
    "UNSLOTH_COMPILE_LOCATION=${UNSLOTH_COMPILE_LOCATION:-${TMPDIR:-/tmp}/frontier-forge-unsloth-compiled}"
  )
fi
for name in \
  http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY \
  REQUESTS_CA_BUNDLE SSL_CERT_FILE HF_HUB_DISABLE_XET; do
  if [[ -n "${!name:-}" ]]; then
    launch_env+=("${name}=${!name}")
  fi
done
tmux new-session -d -s "${session}" \
  env "${launch_env[@]}" \
  ./scripts/remote/run_phase3_rung.sh "${config}" "${seed}" "${backend}"

echo "launched ${rung} seed ${seed} with ${backend} in tmux session ${session}"
echo "started_at=${started_at} (UTC)"
echo "actual_rate_usd_per_gpu_hour=${FORGE_GPU_HOURLY_USD}"
echo "attach: tmux attach -t ${session}"
