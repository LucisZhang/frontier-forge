#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Phase 7.1 A10 bootstrap is Linux-only" >&2
  exit 2
fi
: "${FORGE_GPU_HOURLY_USD:?Set FORGE_GPU_HOURLY_USD=1.53}"
: "${FORGE_BENCH_GIT_SHA:?Set FORGE_BENCH_GIT_SHA to the benchmark commit}"
if [[ "${FORGE_GPU_HOURLY_USD}" != "1.53" ]]; then
  echo "Phase 7.1 pins FORGE_GPU_HOURLY_USD=1.53" >&2
  exit 2
fi

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
cache_root="${FORGE_CACHE_ROOT:-/mnt/frontier-forge/cache}"
mkdir -p "${cache_root}/uv" "${cache_root}/huggingface" "${cache_root}/tmp"
export UV_CACHE_DIR="${cache_root}/uv"
export HF_HOME="${cache_root}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TMPDIR="${cache_root}/tmp"
hf_endpoint="${FORGE_HF_ENDPOINT:-https://huggingface.co}"
case "${hf_endpoint}" in
  https://huggingface.co | https://hf-mirror.com) ;;
  *)
    echo "FORGE_HF_ENDPOINT must be https://huggingface.co or https://hf-mirror.com" >&2
    exit 2
    ;;
esac
hf_repo_id=Luciss007/frontier-forge-r1b
hf_revision=fd4ae1e1989dcb1641a496bf796031491518983e
hf_subdir=bf16-mtp-preserved
expected_artifact_sha256=7878b55f6fe6a9ecb12b9504b1a88d7bc6fef7ba72d91289b6e8d694f6bc75ce
export HF_ENDPOINT="${hf_endpoint}"
export FORGE_HF_REPO_ID="${hf_repo_id}"
export FORGE_HF_REVISION="${hf_revision}"
export FORGE_HF_SUBDIR="${hf_subdir}"
echo "Phase 7.1 model transport endpoint: ${HF_ENDPOINT}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
uv python install 3.12
if [[ ! -x .venv-phase4/bin/python ]]; then
  uv venv .venv-phase4 --python 3.12 --seed
fi
VIRTUAL_ENV="${repo_root}/.venv-phase4" \
  uv sync --active --locked --no-default-groups --group remote-serve

.venv-phase4/bin/python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo_root = Path.cwd()
snapshot = Path(
    snapshot_download(
        repo_id=os.environ["FORGE_HF_REPO_ID"],
        revision=os.environ["FORGE_HF_REVISION"],
        allow_patterns=[f'{os.environ["FORGE_HF_SUBDIR"]}/*'],
        cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
    )
)
source = snapshot / os.environ["FORGE_HF_SUBDIR"]
target = repo_root / "checkpoints/full/r1b/trl/s0/export/merged_bf16_mtp_preserved"
if not source.is_dir():
    raise RuntimeError(f"archive subdirectory is missing: {source}")
target.mkdir(parents=True, exist_ok=True)
for item in source.rglob("*"):
    relative = item.relative_to(source)
    destination = target / relative
    if item.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
    elif item.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != item.resolve():
                raise RuntimeError(f"artifact target conflicts with archive cache: {destination}")
        else:
            destination.symlink_to(os.path.relpath(item, destination.parent))
PY

workload=data/full/phase4/workload-9ed3d99a9d75c357.jsonl
if [[ ! -f "${workload}" ]]; then
  echo "the exact Phase 5 request set must be synced before bootstrap: ${workload}" >&2
  exit 2
fi
if [[ "$(sha256sum "${workload}" | awk '{print $1}')" != "4f042b56aacd6e596e112e290511717bd84737d805df67acbfbafd0845865e23" ]]; then
  echo "Phase 5 request-set hash mismatch" >&2
  exit 2
fi

.venv-phase4/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.version.cuda)'
.venv-phase4/bin/vllm --version
.venv-phase4/bin/python -m gateway.bench.phase7_1_bench --stage verify-artifact
artifact_receipt=results/phase7_1/artifact_verification.json
actual_artifact_sha256="$(jq -r .artifact.sha256 "${artifact_receipt}")"
if [[ "${actual_artifact_sha256}" != "${expected_artifact_sha256}" ]]; then
  echo "verified artifact receipt differs from the fixed archive identity" >&2
  exit 2
fi
artifact_receipt_sha256="$(sha256sum "${artifact_receipt}" | cut -c1-64)"
jq -n \
  --arg git_sha "${FORGE_BENCH_GIT_SHA}" \
  --arg endpoint "${HF_ENDPOINT}" \
  --arg repo_id "${hf_repo_id}" \
  --arg revision "${hf_revision}" \
  --arg allow_pattern "${hf_subdir}/*" \
  --arg artifact_sha256 "${actual_artifact_sha256}" \
  --arg artifact_receipt "${artifact_receipt}" \
  --arg artifact_receipt_sha256 "${artifact_receipt_sha256}" \
  --arg finished_at "$(date --utc --iso-8601=seconds)" \
  '{version:1,status:"complete",phase:"7.1",git_sha:$git_sha,transport:{endpoint:$endpoint,role:"transport_only"},identity:{repo_id:$repo_id,fixed_revision:$revision,allow_pattern:$allow_pattern,verified_tree_sha256:$artifact_sha256},artifact_verification:{path:$artifact_receipt,sha256:$artifact_receipt_sha256},finished_at:$finished_at}' \
  > results/phase7_1/a10_model_transfer.json
.venv-phase4/bin/python -m pip freeze > results/phase7_1/a10_python_packages.txt
echo "Phase 7.1 A10 serving environment and model artifact are verified on /mnt/frontier-forge"
