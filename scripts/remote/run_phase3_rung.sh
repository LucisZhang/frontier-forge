#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CONFIG SEED BACKEND" >&2
  exit 2
fi

config="$1"
seed="$2"
backend="$3"
: "${FORGE_STARTED_AT:?FORGE_STARTED_AT must be passed by launch_phase3.sh}"
: "${FORGE_TRAIN_PYTHON:?FORGE_TRAIN_PYTHON must be passed by launch_phase3.sh}"
: "${FORGE_GPU_HOURLY_USD:?FORGE_GPU_HOURLY_USD must be passed by launch_phase3.sh}"
reference_python=".venv/bin/python"
train_python="${FORGE_TRAIN_PYTHON}"
if [[ ! -x "${reference_python}" || ! -x "${train_python}" ]]; then
  echo "locked Phase 3 Python environment is missing; run bootstrap.sh" >&2
  exit 2
fi

rung="$("${reference_python}" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["rung"])' "${config}")"
log_dir="results/phase3_remote_logs"
mkdir -p "${log_dir}"
log_path="${log_dir}/${rung}-${backend}-s${seed}-${FORGE_STARTED_AT//:/}.log"
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
      --exit-code "${exit_code}" || true
  fi
  exit "${exit_code}"
}
trap record_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${reference_python}" -m forge.train.preflight \
  --config "${config}" --seed "${seed}" --backend "${backend}" --launch

case "${rung}" in
  r0) ;;
  r1|r1b|r2)
    "${train_python}" -m forge.train.sft \
      --config "${config}" --seed "${seed}" --backend "${backend}"
    ;;
  r3)
    "${train_python}" -m forge.train.dpo \
      --config "${config}" --seed "${seed}" --backend "${backend}"
    ;;
  r4)
    "${train_python}" -m forge.train.grpo \
      --config "${config}" --seed "${seed}" --backend "${backend}"
    ;;
  *) echo "unsupported rung ${rung}" >&2; exit 2 ;;
esac

FORGE_DEFER_FINALIZE=1 "${reference_python}" -m forge.train.evaluate \
  --config "${config}" --seed "${seed}" --backend "${backend}"

if [[ "${rung}" == "r4" && "${seed}" == "0" ]]; then
  "${reference_python}" -m forge.train.export \
    --config "${config}" --seed "${seed}" --backend "${backend}"
fi

finish_epoch="$(date -u +%s)"
finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
gpu_hours="$("${reference_python}" -c "print((${finish_epoch} - ${start_epoch}) / 3600.0)")"
"${reference_python}" -m forge.train.finalize \
  --config "${config}" --backend "${backend}" --seed "${seed}" \
  --started-at "${FORGE_STARTED_AT}" --finished-at "${finished_at}" \
  --gpu-hours "${gpu_hours}" --hourly-usd "${FORGE_GPU_HOURLY_USD}"

if [[ "${rung}" == "r1" && "${backend}" == "unsloth" ]]; then
  "${reference_python}" -m forge.train.crosscheck --seed "${seed}"
fi
"${reference_python}" -m forge.train.report

completed=1
trap - EXIT INT TERM
echo "Phase 3 rung complete: ${rung} ${backend} seed ${seed}, gpu_hours=${gpu_hours}"
