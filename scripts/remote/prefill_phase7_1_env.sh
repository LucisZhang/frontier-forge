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
wheelhouse="${cache_root}/wheelhouse/phase7-1-remote-serve"
wheelhouse_manifest=results/phase7_1/a10_mirror_wheelhouse.sha256
mirror=https://mirrors.aliyun.com/pypi/simple
download_jobs="${FORGE_PREFILL_DOWNLOAD_JOBS:-8}"
if [[ ! "${download_jobs}" =~ ^[1-9][0-9]*$ ]] || (( download_jobs > 16 )); then
  echo "FORGE_PREFILL_DOWNLOAD_JOBS must be an integer from 1 through 16" >&2
  exit 2
fi

uv export --locked --no-default-groups --group remote-serve --no-emit-project \
  --format requirements-txt --output-file "${requirements}" >/dev/null
requirements_sha256="$(sha256sum "${requirements}" | awk '{print $1}')"
lock_sha256="$(sha256sum uv.lock | awk '{print $1}')"
requirement_parts="${cache_root}/prefill-requirements/${requirements_sha256}"
mkdir -p "${requirement_parts}" "${wheelhouse}"
.venv-phase4/bin/python - "${requirements}" "${requirement_parts}" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
blocks: list[list[str]] = []
current: list[str] = []
for line in source.read_text().splitlines(keepends=True):
    if line.strip() and not line[0].isspace() and not line.startswith("#"):
        if current:
            blocks.append(current)
        current = [line]
    elif current:
        current.append(line)
if current:
    blocks.append(current)
for index, block in enumerate(blocks):
    (destination / f"{index:04d}.txt").write_text("".join(block))
print(f"split {len(blocks)} hash-pinned requirements")
PY

export mirror wheelhouse
export prefill_python="${repo_root}/.venv-phase4/bin/python"
find "${requirement_parts}" -maxdepth 1 -type f -name '*.txt' -print0 \
  | sort -z \
  | xargs -0 -n 1 -P "${download_jobs}" bash -c '
      PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=180 \
        "${prefill_python}" -m pip download --require-hashes --no-deps \
          --dest "${wheelhouse}" --index-url "${mirror}" --progress-bar off -r "$1"
    ' _
find "${wheelhouse}" -maxdepth 1 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${wheelhouse_manifest}"
wheelhouse_manifest_sha256="$(sha256sum "${wheelhouse_manifest}" | awk '{print $1}')"
wheelhouse_artifacts="$(wc -l < "${wheelhouse_manifest}" | tr -d ' ')"

PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=180 \
  .venv-phase4/bin/python -m pip install \
    --require-hashes --no-deps --no-index --find-links "${wheelhouse}" --progress-bar off \
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
  --arg wheelhouse "${wheelhouse}" \
  --arg wheelhouse_manifest "${wheelhouse_manifest}" \
  --arg wheelhouse_manifest_sha256 "${wheelhouse_manifest_sha256}" \
  --argjson wheelhouse_artifacts "${wheelhouse_artifacts}" \
  --argjson download_jobs "${download_jobs}" \
  --arg pip_report "${pip_report}" \
  --arg pip_report_sha256 "${pip_report_sha256}" \
  --argjson installed_artifacts "${installed_artifacts}" \
  --arg finished_at "$(date --utc --iso-8601=seconds)" \
  '{version:1,status:"complete",phase:"7.1",git_sha:$git_sha,lock_sha256:$lock_sha256,requirements:$requirements,requirements_sha256:$requirements_sha256,index_url:$mirror,download_jobs:$download_jobs,hash_enforcement:"parallel pip download and final pip install both use --require-hashes --no-deps",dependency_resolution:"disabled; complete uv lock export installed with --no-deps",wheelhouse:{path:$wheelhouse,artifacts:$wheelhouse_artifacts,manifest:$wheelhouse_manifest,manifest_sha256:$wheelhouse_manifest_sha256},pip_report:$pip_report,pip_report_sha256:$pip_report_sha256,installed_artifacts:$installed_artifacts,pip_check:"pass",canonical_finalizer:"uv sync --active --locked against the lockfile canonical index",finished_at:$finished_at}' \
  > "${receipt}"

echo "Phase 7.1 hash-verified mirror prefill complete; canonical uv --locked finalization is still required"
