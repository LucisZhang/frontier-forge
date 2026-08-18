#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CONFIG SEED BACKEND" >&2
  exit 2
fi

config="$1"
seed="$2"
backend="$3"
: "${FORGE_STARTED_AT:?FORGE_STARTED_AT must be passed by launch_phase3_export.sh}"
: "${FORGE_GPU_HOURLY_USD:?FORGE_GPU_HOURLY_USD must be passed by launch_phase3_export.sh}"
reference_python=".venv/bin/python"
if [[ ! -x "${reference_python}" ]]; then
  echo "locked Phase 3 Python environment is missing; run bootstrap.sh" >&2
  exit 2
fi

log_dir="results/phase3_remote_logs"
mkdir -p "${log_dir}"
log_path="${log_dir}/r1b-export-${backend}-s${seed}-${FORGE_STARTED_AT//:/}.log"
exec > >(tee -a "${log_path}") 2>&1

start_epoch="$(date -u +%s)"
completed=0

record_failure() {
  exit_code=$?
  if [[ "${completed}" -eq 0 ]]; then
    finish_epoch="$(date -u +%s)"
    finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    gpu_hours="$("${reference_python}" -c "print((${finish_epoch} - ${start_epoch}) / 3600.0)")"
    "${reference_python}" -m forge.train.ledger \
      --config "${config}" --backend "${backend}" --seed "${seed}" \
      --started-at "${FORGE_STARTED_AT}" --finished-at "${finished_at}" \
      --gpu-hours "${gpu_hours}" --hourly-usd "${FORGE_GPU_HOURLY_USD}" \
      --exit-code "${exit_code}" --operation export --status failed || true
  fi
  exit "${exit_code}"
}
trap record_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${reference_python}" -m forge.train.preflight \
  --config "${config}" --seed "${seed}" --backend "${backend}" --launch
"${reference_python}" -m forge.train.export \
  --config "${config}" --seed "${seed}" --backend "${backend}"

finish_epoch="$(date -u +%s)"
finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
gpu_hours="$("${reference_python}" -c "print((${finish_epoch} - ${start_epoch}) / 3600.0)")"
"${reference_python}" -m forge.train.ledger \
  --config "${config}" --backend "${backend}" --seed "${seed}" \
  --started-at "${FORGE_STARTED_AT}" --finished-at "${finished_at}" \
  --gpu-hours "${gpu_hours}" --hourly-usd "${FORGE_GPU_HOURLY_USD}" \
  --exit-code 0 --operation export --status complete
completed=1
"${reference_python}" -m forge.train.report

trap - EXIT INT TERM
echo "Phase 3.1 export complete: r1b ${backend} seed ${seed}, gpu_hours=${gpu_hours}"
