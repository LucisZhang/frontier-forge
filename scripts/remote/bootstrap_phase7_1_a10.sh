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
        repo_id="Luciss007/frontier-forge-r1b",
        revision="fd4ae1e1989dcb1641a496bf796031491518983e",
        allow_patterns=["bf16-mtp-preserved/*"],
        cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
    )
)
source = snapshot / "bf16-mtp-preserved"
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
.venv-phase4/bin/python -m pip freeze > results/phase7_1/a10_python_packages.txt
echo "Phase 7.1 A10 serving environment and model artifact are verified on /mnt/frontier-forge"
