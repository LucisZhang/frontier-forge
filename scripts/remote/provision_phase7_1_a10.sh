#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Phase 7.1 A10 provisioning is Linux-only" >&2
  exit 2
fi
: "${FORGE_CONFIRM_FORMAT_DEVICE:?Set FORGE_CONFIRM_FORMAT_DEVICE=/dev/vdb}"
: "${FORGE_GIT_SHA:?Set FORGE_GIT_SHA to the exact benchmark commit}"
if [[ "${FORGE_CONFIRM_FORMAT_DEVICE}" != "/dev/vdb" ]]; then
  echo "refusing: the only authorized data device is /dev/vdb" >&2
  exit 2
fi
if [[ ! "${FORGE_GIT_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "FORGE_GIT_SHA must be a full lowercase Git SHA" >&2
  exit 2
fi
sudo -n true
if [[ "$(id -un)" != "ecs-user" ]]; then
  echo "run this script as ecs-user; it elevates only the required host operations" >&2
  exit 2
fi

device=/dev/vdb
mount_path=/mnt/frontier-forge
repo_path="${mount_path}/repo"
repo_url="${FORGE_REPO_URL:-https://github.com/LucisZhang/frontier-forge.git}"
if [[ ! -b "${device}" ]]; then
  echo "authorized block device is missing: ${device}" >&2
  exit 2
fi
root_source="$(findmnt -n -o SOURCE /)"
if [[ "${root_source}" == "${device}" || "${root_source}" == "${device}"* ]]; then
  echo "refusing to format the root filesystem device" >&2
  exit 2
fi
mounted_target="$(findmnt -rn -S "${device}" -o TARGET | head -n 1 || true)"
if [[ -n "${mounted_target}" && "${mounted_target}" != "${mount_path}" ]]; then
  echo "refusing a device mounted at an unexpected target: ${mounted_target}" >&2
  exit 2
fi
mapfile -t device_tree < <(lsblk -nrpo NAME "${device}")
if [[ ${#device_tree[@]} -ne 1 || "${device_tree[0]}" != "${device}" ]]; then
  echo "refusing to format a device with partitions or an unexpected topology" >&2
  lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS "${device}" >&2
  exit 2
fi

existing_type="$(sudo blkid -s TYPE -o value "${device}" 2>/dev/null || true)"
formatted_now=false
if [[ -z "${existing_type}" && -z "${mounted_target}" ]]; then
  sudo mkfs.ext4 -F -L frontier-forge "${device}"
  formatted_now=true
elif [[ "${existing_type}" != "ext4" ]]; then
  echo "refusing unexpected existing filesystem ${existing_type} on ${device}" >&2
  exit 2
fi

sudo mkdir -p "${mount_path}"
uuid="$(sudo blkid -s UUID -o value "${device}")"
fstab_line="UUID=${uuid} ${mount_path} ext4 defaults,nofail 0 2"
if ! grep -Fqx "${fstab_line}" /etc/fstab; then
  printf '%s\n' "${fstab_line}" | sudo tee -a /etc/fstab >/dev/null
fi
if ! mountpoint -q "${mount_path}"; then
  sudo mount "${mount_path}"
fi
sudo chown ecs-user:ecs-user "${mount_path}"

sudo modprobe br_netfilter
printf '%s\n' br_netfilter | sudo tee /etc/modules-load.d/frontier-forge.conf >/dev/null
if ! lsmod | awk '$1 == "br_netfilter" {found=1} END {exit !found}'; then
  echo "br_netfilter did not load" >&2
  exit 1
fi
if systemctl list-unit-files nvidia-fabricmanager.service --no-legend 2>/dev/null | grep -q nvidia-fabricmanager; then
  sudo systemctl disable --now nvidia-fabricmanager.service
  sudo systemctl reset-failed nvidia-fabricmanager.service || true
  fabric_enabled="$(systemctl is-enabled nvidia-fabricmanager.service 2>/dev/null || true)"
  fabric_active="$(systemctl is-active nvidia-fabricmanager.service 2>/dev/null || true)"
  if [[ "${fabric_enabled}" != "disabled" || "${fabric_active}" != "inactive" ]]; then
    echo "nvidia-fabricmanager.service was not fully disabled" >&2
    exit 1
  fi
fi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install --yes \
  build-essential ca-certificates cmake curl git jq libboost-all-dev ninja-build \
  pkg-config rsync tmux

if [[ ! -d "${repo_path}/.git" ]]; then
  git clone "${repo_url}" "${repo_path}"
fi
git -C "${repo_path}" remote set-url origin "${repo_url}"
git -C "${repo_path}" fetch origin "${FORGE_GIT_SHA}"
git -C "${repo_path}" checkout --detach "${FORGE_GIT_SHA}"
if [[ "$(git -C "${repo_path}" rev-parse HEAD)" != "${FORGE_GIT_SHA}" ]]; then
  echo "remote repository did not resolve to the requested commit" >&2
  exit 1
fi

mkdir -p "${repo_path}/results/phase7_1"
gpu_json="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -n 1)"
listen_json="$(ss -lntH | awk '{print $4}' | jq -Rsc 'split("\n") | map(select(length > 0))')"
filesystem_created_text="$(sudo tune2fs -l "${device}" | sed -n 's/^Filesystem created:[[:space:]]*//p')"
filesystem_created_epoch="$(date --date="${filesystem_created_text}" +%s)"
booted_at_text="$(uptime -s)"
booted_at_epoch="$(date --date="${booted_at_text}" +%s)"
filesystem_created_at="$(date --utc --date="@${filesystem_created_epoch}" --iso-8601=seconds)"
booted_at="$(date --utc --date="@${booted_at_epoch}" --iso-8601=seconds)"
formatted_during_current_boot=false
if (( filesystem_created_epoch >= booted_at_epoch )); then
  formatted_during_current_boot=true
fi
jq -n \
  --arg device "${device}" \
  --arg filesystem ext4 \
  --arg uuid "${uuid}" \
  --arg mount "${mount_path}" \
  --arg repo "${repo_path}" \
  --arg git_sha "${FORGE_GIT_SHA}" \
  --arg gpu "${gpu_json}" \
  --argjson formatted_now "${formatted_now}" \
  --arg filesystem_created_at "${filesystem_created_at}" \
  --arg booted_at "${booted_at}" \
  --argjson formatted_during_current_boot "${formatted_during_current_boot}" \
  --argjson listening_addresses "${listen_json}" \
  '{version:1,status:"complete",phase:"7.1",device:$device,filesystem:$filesystem,uuid:$uuid,mount:$mount,formatted_now:$formatted_now,filesystem_created_at:$filesystem_created_at,booted_at:$booted_at,formatted_during_current_boot:$formatted_during_current_boot,repo:$repo,git_sha:$git_sha,gpu:$gpu,br_netfilter_loaded:true,br_netfilter_persisted:true,nvidia_fabricmanager_disabled:true,security_group_changes:"none; this script has no cloud-control-plane operations",listening_addresses:$listening_addresses}' \
  > "${repo_path}/results/phase7_1/host_provisioning.json"

echo "Phase 7.1 host provisioned at ${repo_path}; no security-group operation was performed"
