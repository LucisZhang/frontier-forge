# Phase 7.2 single-node k3s acceptance report

Status: **Gate 7.2 PASS**. Run `phase7_2_single_node_k3s_a10` at git
`167edca1d9184fb3f97c7039b62d24d10405dba7`. Raw receipt:
`results/phase7_2/raw/phase7_2_acceptance.json`.

## Scope and disclosure

This is a real-VM rehearsal on one k3s node and one physical NVIDIA A10. The
NVIDIA device plugin exposes two time-sliced `nvidia.com/gpu.shared` allocations
so GPTQ-int4 and BF16 can coexist. It is not cloud-production or multi-GPU
evidence; Kafka remains out of scope. Every application and dashboard Service is
`ClusterIP`, and operator access used loopback-only kubectl port-forwards through
an SSH tunnel. No security-group port was opened.

The namespace retained Pod Security `baseline`. Administrator-owned Local
PersistentVolumes, pinned to the labeled model-store node, exposed the two
already-hashed model trees to vLLM through read-only PVC mounts; the Pods did not
request direct `hostPath` access.

Both serving trees were verified before launch: GPTQ-int4
`c99b42cf0e062cc75f2df8588725d0c29383666f3db0c1ae837ce15bfe6d39d2`
and BF16/MTP-preserved
`7878b55f6fe6a9ecb12b9504b1a88d7bc6fef7ba72d91289b6e8d694f6bc75ce`.

## Autoscaling and cold start

- Gateway custom-metric KEDA scale: 1 → **3**
  ready replicas, then back to 1 after cooldown.
- Saturation evidence: queue high-watermark **24/24**;
  k6 observed **3799** HTTP 429 responses and every
  one carried `Retry-After`.
- GPU scale-to-zero/from-zero: **n=10**, measured from
  the Prometheus demand trigger through the first task-success-verified gateway
  response.

| cold-start seconds | min | p50 | p95 | max | mean |
|---|---:|---:|---:|---:|---:|
| trigger → verified response | 116.539 | 124.617 | 127.112 | 128.050 | 123.971 |

## One-GPU canary

Both deployments were Ready concurrently under the two time-sliced allocations.
The router exposed the selected upstream address in a response header so the
weighted stages were measured rather than inferred from configuration alone.
These attribution probes were sequential because both servers time-share one
physical GPU; concurrent saturation is covered separately by the k6/KEDA drill.

| router stage | int4 responses | BF16 responses | unknown | SLO guard |
|---|---:|---:|---:|---:|
| canary-10 | 45 | 5 | 0 | pass |
| canary-50 | 20 | 20 | 0 | pass |
| canary-100 | 0 | 20 | 0 | pass |

After 100% BF16 promotion, a controlled HTTP-500 upstream deliberately degraded
the promoted path. `ForgeAvailabilityBurnRate` entered `firing`; the router was
then rolled back to stable GPTQ-int4 and a task-success-verified request recovered.
`drill_bad_canary_rollback.json` adds drill metadata to this same execution in
`canary_release.json`; the two files are not receipts for independent canary runs.

## Alerts and runbook drills

- Availability multi-window burn alert: fired under the `fault-500` k6 scenario
  (64 observed HTTP 500 responses).
- Latency multi-window burn alert: fired under the three-second latency injector
  (573 observed HTTP 200 responses), isolating
  latency from availability.
- Kill-vLLM drill: old pod `e4876457-773d-48db-a84e-f8f0721987b0` was replaced by
  `60240ea3-99f0-49d5-88bf-d1da53849a24`; the gateway failed closed with HTTP 503 and recovered
  a verified request in **120.708 s**.
- Saturation and bad-canary rollback drills reuse the same k6/KEDA and alert
  receipts above; all three are scripted by
  `scripts/remote/phase7_2_acceptance.py` and documented in
  `docs/runbooks/phase7_2.md`.

## CI and cost

- CPU-only kind manifest smoke and the full test job passed in
  https://github.com/LucisZhang/frontier-forge/actions/runs/32412121467 at `167edca1d9184fb3f97c7039b62d24d10405dba7`.
- VM rate: `FORGE_GPU_HOURLY_USD=1.53` (¥11/hour at 7.2 CNY/USD).
- Phase 7.2 delegated VM interval: **3.4672 h =
  $5.3048**. This records the whole Phase 7.2 interval after
  Gate 7.1 finalization, including cluster/image setup and idle orchestration time.

## Gate 7.2

- [x] inventory complete
- [x] keda scale event receipt
- [x] cold start n at least 10
- [x] canary promote and rollback
- [x] availability alert fault fired
- [x] latency alert fault fired
- [x] three runbook drills complete
- [x] kind ci green
- [x] single node disclosure

## Reproduction

```sh
export FORGE_GPU_HOURLY_USD=1.53
export PYTHONPATH=src:.
.venv-phase4/bin/python scripts/remote/phase7_2_acceptance.py inventory
.venv-phase4/bin/python scripts/remote/phase7_2_acceptance.py cold-start --iterations 10
.venv-phase4/bin/python scripts/remote/phase7_2_acceptance.py kill-drill
.venv-phase4/bin/python scripts/remote/phase7_2_acceptance.py gateway-scale
.venv-phase4/bin/python scripts/remote/phase7_2_acceptance.py canary
.venv-phase4/bin/python scripts/remote/phase7_2_acceptance.py latency-alert
```
