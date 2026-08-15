#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "bootstrap.sh is for a Linux GPU pod" >&2
  exit 2
fi

if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install --yes ca-certificates curl git rsync tmux
else
  apt-get update
  apt-get install --yes ca-certificates curl git rsync tmux
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv python install 3.12
uv sync --locked --python 3.12

FORGE_TORCH_VERSION="${FORGE_TORCH_VERSION:-2.8.0}"
FORGE_CUDA_INDEX_URL="${FORGE_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
uv pip install \
  --python .venv/bin/python \
  --index-url "${FORGE_CUDA_INDEX_URL}" \
  "torch==${FORGE_TORCH_VERSION}"

uv run python -c \
  'import torch; assert torch.version.cuda is not None; print(f"torch CUDA {torch.version.cuda}")'

if ! tmux has-session -t frontier-forge 2>/dev/null; then
  tmux new-session -d -s frontier-forge
fi

echo "GPU pod bootstrap complete; tmux session: frontier-forge"
