#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Phase 7.1 environment prefill is Linux-only" >&2
  exit 2
fi
: "${FORGE_BENCH_GIT_SHA:?Set FORGE_BENCH_GIT_SHA to the benchmark commit}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_real="$(readlink -f "${repo_root}")"
if [[ "${repo_real}" != /mnt/frontier-forge/* ]] || ! mountpoint -q /mnt/frontier-forge; then
  echo "repo and caches must live on the mounted /mnt/frontier-forge data disk" >&2
  exit 2
fi
cd "${repo_root}"
if [[ "$(git rev-parse HEAD)" != "${FORGE_BENCH_GIT_SHA}" ]]; then
  echo "checked-out commit differs from FORGE_BENCH_GIT_SHA" >&2
  exit 2
fi
if [[ ! -x .venv-phase4/bin/python ]]; then
  echo "bootstrap must create .venv-phase4 before the cache prefill" >&2
  exit 2
fi

cache_root="${FORGE_CACHE_ROOT:-/mnt/frontier-forge/cache}"
mkdir -p "${cache_root}/pip" "${cache_root}/tmp" results/phase7_1
export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR="${cache_root}/uv"
export PIP_CACHE_DIR="${cache_root}/pip"
export TMPDIR="${cache_root}/tmp"

requirements=results/phase7_1/a10_remote_serve_requirements.txt
pip_report=results/phase7_1/a10_mirror_prefill_pip_report.json
receipt=results/phase7_1/a10_mirror_prefill_receipt.json
mirror=https://mirrors.aliyun.com/pypi/simple

uv export --locked --no-default-groups --group remote-serve --no-emit-project \
  --format requirements-txt --output-file "${requirements}" >/dev/null
requirements_sha256="$(sha256sum "${requirements}" | awk '{print $1}')"
lock_sha256="$(sha256sum uv.lock | awk '{print $1}')"

PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=180 \
  .venv-phase4/bin/python -m pip install \
    --require-hashes --no-deps --index-url "${mirror}" --progress-bar off \
    --report "${pip_report}" -r "${requirements}"
.venv-phase4/bin/python -m pip check

pip_report_sha256="$(sha256sum "${pip_report}" | awk '{print $1}')"
installed_artifacts="$(jq '.install | length' "${pip_report}")"
jq -n \
  --arg git_sha "${FORGE_BENCH_GIT_SHA}" \
  --arg lock_sha256 "${lock_sha256}" \
  --arg requirements "${requirements}" \
  --arg requirements_sha256 "${requirements_sha256}" \
  --arg mirror "${mirror}" \
  --arg pip_report "${pip_report}" \
  --arg pip_report_sha256 "${pip_report_sha256}" \
  --argjson installed_artifacts "${installed_artifacts}" \
  --arg finished_at "$(date --utc --iso-8601=seconds)" \
  '{version:1,status:"complete",phase:"7.1",git_sha:$git_sha,lock_sha256:$lock_sha256,requirements:$requirements,requirements_sha256:$requirements_sha256,index_url:$mirror,hash_enforcement:"pip --require-hashes",dependency_resolution:"disabled; complete uv lock export installed with --no-deps",pip_report:$pip_report,pip_report_sha256:$pip_report_sha256,installed_artifacts:$installed_artifacts,pip_check:"pass",canonical_finalizer:"uv sync --active --locked against the lockfile canonical index",finished_at:$finished_at}' \
  > "${receipt}"

echo "Phase 7.1 hash-verified mirror prefill complete; canonical uv --locked finalization is still required"
