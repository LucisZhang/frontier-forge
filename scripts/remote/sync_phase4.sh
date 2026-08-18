#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || "$1" != "down" ]]; then
  echo "usage: FORGE_REMOTE_ROOT=user@host:/path/to/frontier-forge $0 down" >&2
  exit 2
fi
: "${FORGE_REMOTE_ROOT:?Set FORGE_REMOTE_ROOT to user@host:/absolute/project/path}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
local_root="$(cd "${script_dir}/../.." && pwd)/"
remote_root="${FORGE_REMOTE_ROOT%/}/"

rsync --archive --compress --prune-empty-dirs \
  --include=/results/ \
  --include=/results/runs.jsonl \
  --include=/results/phase4/*** \
  --include=/results/phase4_serving_report.md \
  --include=/results/phase4_spec_decode_boundary.svg \
  --include=/results/phase4_report_manifest.json \
  --exclude='*' \
  "${remote_root}" "${local_root}"
