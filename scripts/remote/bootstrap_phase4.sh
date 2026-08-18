#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Phase 4 vLLM bootstrap is Linux CUDA pod-only" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
uv_bin="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "${uv_bin}" && -x /root/.local/bin/uv ]]; then
  uv_bin=/root/.local/bin/uv
fi
if [[ ! -x "${uv_bin}" ]]; then
  echo "an existing uv executable is required; Phase 4 will not install outside the repo" >&2
  exit 2
fi
mkdir -p .uv-cache-phase4 .tmp-phase4 .cache/huggingface
export UV_CACHE_DIR="${repo_root}/.uv-cache-phase4"
export TMPDIR="${repo_root}/.tmp-phase4"
export HF_HOME="${repo_root}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

if [[ ! -x .venv-phase4/bin/python ]]; then
  "${uv_bin}" venv .venv-phase4 --python 3.12 --seed
fi

if [[ -n "${FORGE_UV_MIRROR_URL:-}" ]]; then
  requirements_path=".tmp-phase4/remote-serve-requirements.txt"
  "${uv_bin}" export \
    --locked \
    --no-default-groups \
    --group remote-serve \
    --format requirements-txt \
    --no-emit-project \
    --output-file "${requirements_path}"
  VIRTUAL_ENV="${repo_root}/.venv-phase4" \
    "${uv_bin}" pip install \
      --python .venv-phase4/bin/python \
      --index-url "${FORGE_UV_MIRROR_URL}" \
      --require-hashes \
      --requirements "${requirements_path}"
fi

VIRTUAL_ENV="${repo_root}/.venv-phase4" \
  "${uv_bin}" sync --active --locked --no-default-groups --group remote-serve

.venv-phase4/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-0.8B-Base",
    revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
    allow_patterns=[
        "*.json",
        "*.jinja",
        "*.safetensors",
        "*.model",
        "*.txt",
    ],
)
PY

.venv-phase4/bin/python -m forge.bench.preflight
.venv-phase4/bin/vllm --version
echo "Phase 4 vLLM environment ready inside ${repo_root}/.venv-phase4"
