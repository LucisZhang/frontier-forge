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
uv sync --locked --python 3.12 --no-default-groups \
  --group train --group remote-reference
env UV_PROJECT_ENVIRONMENT=.venv-unsloth \
  uv sync --locked --python 3.12 --no-default-groups --group unsloth-train

uv pip check --python .venv/bin/python
uv pip check --python .venv-unsloth/bin/python

.venv/bin/python -c \
  'import torch; assert torch.version.cuda is not None; assert torch.cuda.is_available(); print(f"torch CUDA {torch.version.cuda}: {torch.cuda.get_device_name(0)}")'
.venv-unsloth/bin/python -c \
  'import torch; assert torch.version.cuda is not None; assert torch.cuda.is_available(); print(f"Unsloth torch CUDA {torch.version.cuda}: {torch.cuda.get_device_name(0)}")'

.venv/bin/python -c \
  'import importlib.metadata as m; expected={"torch":"2.13.0","transformers":"5.15.0","trl":"1.10.0","bitsandbytes":"0.50.1","gptqmodel":"7.3.2","optimum":"2.2.0"}; actual={k:m.version(k) for k in expected}; assert actual == expected, (actual, expected); print(actual)'
.venv/bin/python -c \
  'import inspect; from trl import GRPOConfig; grpo=inspect.signature(GRPOConfig).parameters; assert "chat_template_kwargs" in grpo; assert "log_completions" in grpo; print("Reference GRPO contracts verified")'
.venv-unsloth/bin/python -c \
  'import importlib.metadata as m; expected={"torch":"2.11.0","torchvision":"0.26.0","transformers":"5.5.0","trl":"0.24.0","unsloth":"2026.8.18","bitsandbytes":"0.50.1"}; actual={k:m.version(k) for k in expected}; assert actual == expected, (actual, expected); print(actual)'
.venv-unsloth/bin/python -c \
  'import inspect; from trl import DPOConfig, GRPOConfig, SFTConfig; assert "max_length" in inspect.signature(SFTConfig).parameters; assert "max_prompt_length" in inspect.signature(DPOConfig).parameters; assert "max_prompt_length" in inspect.signature(GRPOConfig).parameters; print("Unsloth TRL trainer contracts verified")'

if ! tmux has-session -t frontier-forge 2>/dev/null; then
  tmux new-session -d -s frontier-forge
fi

echo "GPU pod bootstrap complete; tmux session: frontier-forge"
