#!/usr/bin/env bash
set -euo pipefail

# Fail-closed installer for the pinned single-node Phase 7.2 runtime. Download
# the official installer, air-gap archive, and chart archives separately; this
# script verifies the immutable inputs and never changes firewall/security-group
# rules or creates externally reachable Services.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
values_dir="${repo_root}/deploy/phase7_2/charts"

: "${K3S_INSTALLER:=/tmp/forge-phase7-2-install-k3s-cn.sh}"
: "${K3S_INSTALLER_SHA256:=3944aa467eb945b5ff2151a8e4f8d4a5f3a210d31ab39aec81f37606936d0863}"
: "${K3S_AIRGAP_IMAGES:=/tmp/k3s-airgap-images-amd64.tar.gz}"
: "${K3S_AIRGAP_SHA256:=e09e718f931ef094294781d3c35e58b84e2170acd5cc2edae075a20f58d1f89d}"
: "${PHASE7_2_CHART_DIR:=/tmp/forge-phase7-2-charts}"

k3s_version="v1.36.3+k3s1"
nvidia_chart="${PHASE7_2_CHART_DIR}/nvidia-device-plugin-0.20.0.tgz"
monitoring_chart="${PHASE7_2_CHART_DIR}/kube-prometheus-stack-88.5.2.tgz"
adapter_chart="${PHASE7_2_CHART_DIR}/prometheus-adapter-5.3.0.tgz"
keda_chart="${PHASE7_2_CHART_DIR}/keda-2.20.2.tgz"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "required immutable input missing: $1" >&2
    exit 1
  fi
}

verify_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch for ${path}: ${actual} != ${expected}" >&2
    exit 1
  fi
}

systemctl is-system-running >/dev/null
if ! grep -q '^0::/' /proc/1/cgroup || grep -Eq 'docker|containerd|kubepods' /proc/1/cgroup; then
  echo "PID 1 is not in the real VM root cgroup" >&2
  exit 1
fi
if [[ "$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)" != "0" ]]; then
  echo "swap must be disabled" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits \
  | grep -Eq '^NVIDIA A10, (23028|23040)$'
command -v nvidia-container-runtime >/dev/null
command -v helm >/dev/null

for path in "${nvidia_chart}" "${monitoring_chart}" "${adapter_chart}" "${keda_chart}"; do
  require_file "${path}"
done
verify_sha256 "${nvidia_chart}" \
  9e5642bcdd3805c6e931f9d3ca542677d1f714d68c14c1364892011e67d9a606
verify_sha256 "${monitoring_chart}" \
  aac57bfb6fb53c3a9c6f4f0526b61145507dd81ff5312c67305ba5e31b7732d6
verify_sha256 "${adapter_chart}" \
  aa6752b6207ed788522714c3ef7f67f27d423eb4d9512fecb52dc641b6434f31
verify_sha256 "${keda_chart}" \
  db60b1fa60f0565fefa4ce2f87e855b6a8854363ee82962fbe28a12a70d64c4c

if command -v k3s >/dev/null; then
  k3s --version | grep -Fq "${k3s_version}"
else
  require_file "${K3S_INSTALLER}"
  require_file "${K3S_AIRGAP_IMAGES}"
  verify_sha256 "${K3S_INSTALLER}" "${K3S_INSTALLER_SHA256}"
  verify_sha256 "${K3S_AIRGAP_IMAGES}" "${K3S_AIRGAP_SHA256}"
  sudo env \
    INSTALL_K3S_VERSION="${k3s_version}" \
    INSTALL_K3S_MIRROR=cn \
    INSTALL_K3S_SKIP_START=true \
    INSTALL_K3S_EXEC='server --disable traefik --disable servicelb --write-kubeconfig-mode 600' \
    sh "${K3S_INSTALLER}"
  sudo mkdir -p /var/lib/rancher/k3s/agent/images
  sudo install -m 0644 "${K3S_AIRGAP_IMAGES}" \
    /var/lib/rancher/k3s/agent/images/k3s-airgap-images-amd64.tar.gz
  sudo systemctl enable --now k3s
fi

for _ in $(seq 1 120); do
  if sudo k3s kubectl get node >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
sudo k3s kubectl wait --for=condition=Ready node --all --timeout=180s
node_name="$(sudo k3s kubectl get node -o jsonpath='{.items[0].metadata.name}')"
sudo k3s kubectl label node "${node_name}" \
  nvidia.com/gpu.present=true \
  forge.openai.com/model-store=true \
  --overwrite

helm_args=(--kubeconfig /etc/rancher/k3s/k3s.yaml)
sudo helm "${helm_args[@]}" upgrade --install nvidia-device-plugin "${nvidia_chart}" \
  --namespace kube-system \
  --values "${values_dir}/nvidia-device-plugin-values.yaml" \
  --wait --timeout 10m
sudo helm "${helm_args[@]}" upgrade --install monitoring "${monitoring_chart}" \
  --namespace monitoring --create-namespace \
  --values "${values_dir}/kube-prometheus-stack-values.yaml" \
  --wait --timeout 15m
sudo helm "${helm_args[@]}" upgrade --install prometheus-adapter "${adapter_chart}" \
  --namespace monitoring \
  --values "${values_dir}/prometheus-adapter-values.yaml" \
  --wait --timeout 10m
sudo helm "${helm_args[@]}" upgrade --install keda "${keda_chart}" \
  --namespace keda --create-namespace \
  --values "${values_dir}/keda-values.yaml" \
  --wait --timeout 10m

sudo k3s kubectl get node "${node_name}" \
  -o jsonpath='{.status.capacity.nvidia\.com/gpu\.shared}{"\n"}' \
  | grep -Fxq '2'
sudo k3s kubectl get pods -A
