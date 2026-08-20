#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "up" && "$1" != "down" ) ]]; then
  echo "usage: FORGE_SSH_CONFIG=.remote/ssh_config FORGE_REMOTE_ALIAS=frontier-forge-a10 $0 up|down" >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ssh_config="${FORGE_SSH_CONFIG:-${repo_root}/.remote/ssh_config}"
remote_alias="${FORGE_REMOTE_ALIAS:-frontier-forge-a10}"
remote_root="${FORGE_REMOTE_ROOT:-/mnt/frontier-forge/repo}"
if [[ ! -f "${ssh_config}" ]]; then
  echo "SSH config is missing: ${ssh_config}" >&2
  exit 2
fi
ssh_transport="ssh -F ${ssh_config}"
if [[ "$1" == "up" ]]; then
  rsync --archive --compress --prune-empty-dirs -e "${ssh_transport}" \
    --include=/data/ \
    --include=/data/full/ \
    --include=/data/full/phase4/ \
    --include=/data/full/phase4/workload-9ed3d99a9d75c357.jsonl \
    --exclude='*' \
    "${repo_root}/" "${remote_alias}:${remote_root}/"
else
  rsync --archive --compress --prune-empty-dirs -e "${ssh_transport}" \
    --exclude=/results/phase7_1/runtime/*** \
    --include=/results/ \
    --include=/results/phase7_1/*** \
    --include=/results/phase7_1_gateway_a10_report.md \
    --exclude='*' \
    "${remote_alias}:${remote_root}/" "${repo_root}/"
fi
