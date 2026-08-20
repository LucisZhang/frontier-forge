#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

: "${KIND_CLUSTER_NAME:=forge-phase7-2}"
: "${GATEWAY_IMAGE:=frontier-forge/gateway:phase7-2}"

cleanup() {
  if [[ -n "${port_forward_pid:-}" ]]; then
    kill "${port_forward_pid}" 2>/dev/null || true
  fi
  kind delete cluster --name "${KIND_CLUSTER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

kind create cluster --name "${KIND_CLUSTER_NAME}" --wait 120s
docker build --file gateway/Dockerfile.runtime --tag "${GATEWAY_IMAGE}" .
kind load docker-image --name "${KIND_CLUSTER_NAME}" "${GATEWAY_IMAGE}"
kubectl apply -k deploy/phase7_2/kind
kubectl -n forge-system rollout status deployment/forge-mock-upstream --timeout=120s
kubectl -n forge-system rollout status deployment/forge-gateway --timeout=120s

kubectl -n forge-system port-forward service/forge-gateway 19000:9000 \
  >"${TMPDIR:-/tmp}/forge-phase7-2-kind-port-forward.log" 2>&1 &
port_forward_pid=$!
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:19000/readyz >/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS http://127.0.0.1:19000/readyz >/dev/null
response="$(curl -fsS -H 'Content-Type: application/json' \
  -d '{"model":"forge-r1b","messages":[{"role":"user","content":"route"}],"max_tokens":64,"stream":false}' \
  http://127.0.0.1:19000/v1/chat/completions)"
jq -e '.choices[0].message.content | fromjson | .tool_call.name == "route_to_company"' \
  <<<"${response}" >/dev/null
curl -fsS http://127.0.0.1:19000/metrics | grep -q 'forge_gateway_responses_total'
kubectl get services --all-namespaces -o json \
  | jq -e '[.items[] | select(.spec.type == "NodePort" or .spec.type == "LoadBalancer")] | length == 0' \
  >/dev/null
echo "Phase 7.2 kind manifest smoke passed"
