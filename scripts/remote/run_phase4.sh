#!/usr/bin/env bash
set -euo pipefail

: "${FORGE_STARTED_AT:?FORGE_STARTED_AT must be passed by launch_phase4.sh}"
: "${FORGE_GPU_HOURLY_USD:?FORGE_GPU_HOURLY_USD must be passed by launch_phase4.sh}"
: "${FORGE_BENCH_GIT_SHA:?FORGE_BENCH_GIT_SHA must be passed by launch_phase4.sh}"
if [[ "${FORGE_GPU_HOURLY_USD}" != "0.30" ]]; then
  echo "Phase 4 hourly rate must be 0.30" >&2
  exit 2
fi

python_bin=".venv-phase4/bin/python"
vllm_bin=".venv-phase4/bin/vllm"
if [[ ! -x "${python_bin}" || ! -x "${vllm_bin}" ]]; then
  echo "Phase 4 environment is missing; run bootstrap_phase4.sh" >&2
  exit 2
fi

mkdir -p results/phase4/logs
worker_log="results/phase4/logs/phase4-worker-${FORGE_STARTED_AT//:/}.log"
exec > >(tee -a "${worker_log}") 2>&1

server_pid=""
stop_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM -- "-${server_pid}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      if ! kill -0 "${server_pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${server_pid}" 2>/dev/null; then
      kill -KILL -- "-${server_pid}" 2>/dev/null || true
    fi
    wait "${server_pid}" 2>/dev/null || true
  fi
  server_pid=""
}
trap 'stop_server; exit 130' INT
trap 'stop_server; exit 143' TERM
trap stop_server EXIT

gpu_processes() {
  nvidia-smi pmon -c 1 | awk '$1 !~ /^#/ && $2 ~ /^[0-9]+$/ {print $1, $2, $3, $10}'
}

wait_for_gpu_idle() {
  local processes
  while true; do
    processes="$(gpu_processes)"
    if [[ -z "${processes//[[:space:]]/}" ]]; then
      return
    fi
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) GPU occupied by another process; waiting without launching:" >&2
    echo "${processes}" >&2
    sleep 60
  done
}

wait_for_server() {
  local log_path="$1"
  for _ in $(seq 1 120); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "vLLM exited before health became ready" >&2
      tail -n 120 "${log_path}" >&2
      return 1
    fi
    if "${python_bin}" - <<'PY' >/dev/null 2>&1
import urllib.request
with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
    assert response.status == 200
PY
    then
      return
    fi
    sleep 5
  done
  echo "vLLM health timeout" >&2
  tail -n 120 "${log_path}" >&2
  return 1
}

"${python_bin}" -m forge.bench.preflight --verify-artifacts

configs=(
  configs/phase4/serve_r1b_bf16.yaml
  configs/phase4/serve_r1b_gptq_int4.yaml
  configs/phase4/serve_r3eq_bf16.yaml
  configs/phase4/serve_r3eq_gptq_int4.yaml
  configs/phase4/spec_r1b_bf16_baseline.yaml
  configs/phase4/spec_r1b_bf16_qwen08b.yaml
  configs/phase4/structured_r1b_bf16_xgrammar.yaml
  configs/phase4/structured_r1b_bf16_outlines.yaml
)

for config in "${configs[@]}"; do
  run_id="$("${python_bin}" -c 'import sys; from forge.bench.config import load_phase4_config; print(load_phase4_config(sys.argv[1])["run_id"])' "${config}")"
  if [[ -f "results/phase4/raw/${run_id}.json" ]]; then
    echo "raw artifact already exists; validating immutable receipt ${run_id}"
    "${python_bin}" -m forge.bench.runner --config "${config}" --validate-existing
    continue
  fi
  wait_for_gpu_idle
  server_log="results/phase4/logs/${run_id}-${FORGE_STARTED_AT//:/}.server.log"
  mapfile -d '' command < <("${python_bin}" -m forge.bench.server_args \
    --config "${config}" --executable "${vllm_bin}" --null)
  echo "starting ${run_id}; command arguments are pinned by ${config}"
  setsid "${command[@]}" >"${server_log}" 2>&1 &
  server_pid=$!
  wait_for_server "${server_log}"
  case "${run_id}" in
    phase4_serve_*) make serve-bench CONFIG="${config}" SERVER_URL=http://127.0.0.1:8000 ;;
    phase4_spec_decode_*) make spec-decode-bench CONFIG="${config}" SERVER_URL=http://127.0.0.1:8000 ;;
    phase4_structured_*) make structured-bench CONFIG="${config}" SERVER_URL=http://127.0.0.1:8000 ;;
    *) echo "unsupported Phase 4 run id: ${run_id}" >&2; exit 2 ;;
  esac
  stop_server
done

make bench-report
trap - EXIT INT TERM
echo "Phase 4 complete; pod remains running"
