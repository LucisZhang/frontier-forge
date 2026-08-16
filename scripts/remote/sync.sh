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

common_filters=(
  --include=/configs/
  --include=/configs/***
  --include=/results/
  --include=/results/***
  --include=/checkpoints/
  --include=/checkpoints/***
)

up_filters=(
  "${common_filters[@]}"
  --include=/data/
  --include=/data/phase1_2/
  --include=/data/phase1_2/manifest.json
  --include=/data/phase1_2/splits/
  --include=/data/phase1_2/splits/test_iid.parquet
  --include=/data/phase1_2/splits/test_drift.parquet
  --include=/data/phase2/
  --include=/data/phase2/manifest.json
  --include=/data/phase2/sft_rule.jsonl
  --include=/data/phase2/sft_distilled.jsonl
  --include=/data/phase2/dpo_pairs.jsonl
  --include=/data/phase3/
  --include=/data/phase3/manifest.json
  --include=/data/phase3/r1b_sft_rule.jsonl
  --exclude=*
)

down_filters=(
  "${common_filters[@]}"
  --include=/data/
  --include=/data/full/
  --include=/data/full/phase3/
  --include=/data/full/phase3/***
  --exclude=*
)

if [[ "${direction}" == "up" ]]; then
  rsync --archive --compress --prune-empty-dirs \
    "${up_filters[@]}" "${local_root}" "${remote_root}"
else
  rsync --archive --compress --prune-empty-dirs \
    "${down_filters[@]}" "${remote_root}" "${local_root}"
fi
