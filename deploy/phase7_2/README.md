# Phase 7.2 deployment

This directory contains the reproducible single-node k3s serving runtime for
Frontier Forge. It deploys the C++ gateway, R1b GPTQ-int4 and BF16 vLLM variants,
NVIDIA device-plugin time slicing, kube-prometheus-stack, Prometheus Adapter,
KEDA, controlled fault upstreams, SLO rules, and the versioned Grafana dashboard.

The topology is intentionally narrow: one real VM, one physical A10, two
time-sliced GPU allocations, and horizontally scalable CPU-only gateway pods.
It is not cloud-production or multi-GPU evidence. Kafka is out of scope.

All application and dashboard Services are `ClusterIP`. The operational scripts
use loopback-bound `kubectl port-forward` processes reached through SSH tunnels;
no security-group port is opened.

## Layout

- `base/`: gateway and namespace shared by real and CPU-only kind deployments.
- `real/`: vLLM variants, weighted release router, controlled faults, KEDA, and
  monitoring custom resources.
- `kind/`: mock-upstream overlay used by CI without a GPU.
- `charts/`: pinned values for the four third-party charts.
- `dashboards/`: source-controlled Grafana JSON.
- `k6/`: load scenarios used by scaling, overload, and alert drills.
- `versions.lock.yaml`: exact cluster, chart, image, and model archive identities.

Run `make phase7-2-manifest-test` for static validation and
`make phase7-2-kind-smoke` for the full CPU kind smoke.
