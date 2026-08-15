#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: FORGE_REMOTE_ROOT=user@host:/path/to/frontier-forge $0 up|down" >&2
}

if [[ $# -ne 1 ]] || [[ "$1" != "up" && "$1" != "down" ]]; then
  usage
  exit 2
fi

direction="$1"
if [[ "${SMOKE:-0}" == "1" ]]; then
  printf '[stub] sync-%s\n' "${direction}"
  exit 0
fi

: "${FORGE_REMOTE_ROOT:?Set FORGE_REMOTE_ROOT to user@host:/absolute/project/path}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
local_root="$(cd "${script_dir}/../.." && pwd)/"
remote_root="${FORGE_REMOTE_ROOT%/}/"

filters=(
  --include=/configs/
  --include=/configs/***
  --include=/results/
  --include=/results/***
  --include=/checkpoints/
  --include=/checkpoints/***
  --include=/data/
  --include=/data/manifests/
  --include=/data/manifests/***
  --exclude=*
)

if [[ "${direction}" == "up" ]]; then
  rsync --archive --compress --prune-empty-dirs "${filters[@]}" "${local_root}" "${remote_root}"
else
  rsync --archive --compress --prune-empty-dirs "${filters[@]}" "${remote_root}" "${local_root}"
fi
