# Phase 7.2 single-node serving runbook

This runbook is deliberately scoped to one k3s node and one physical A10 GPU.
It is an operational rehearsal, not evidence of cloud production or multi-GPU
autoscaling. All services are `ClusterIP`; operators reach Grafana, Prometheus,
and the gateway only through loopback-bound SSH tunnels.

## Preconditions

1. Verify the node is a real systemd VM with cgroup v2, no swap, and an idle A10.
2. Verify both immutable model trees against `deploy/phase7_2/versions.lock.yaml`.
3. Verify the NVIDIA device plugin reports exactly two
   `nvidia.com/gpu.shared` time-sliced allocations backed by the one physical GPU.
4. Verify no `NodePort`, `LoadBalancer`, or `hostPort` exists in the Phase 7.2
   manifests.

## Dashboard access

On the operator machine, bind only loopback addresses:

```sh
ssh -F .remote/ssh_config \
  -L 13000:127.0.0.1:13000 \
  -L 19090:127.0.0.1:19090 \
  frontier-forge-a10
```

On the VM, run loopback-only port-forwards for Grafana and Prometheus. Never
create a NodePort or change a security-group rule for dashboard access.

## Drill 1: kill the active vLLM pod

Goal: prove the GPU workload can be recreated and the gateway fails closed while
the upstream is unavailable.

1. Record the current pod UID and a successful verified request.
2. Delete that exact `vllm-int4` pod.
3. Confirm the gateway returns `503/unavailable`, not a fabricated success.
4. Wait for a different pod UID to become Ready and send the same verified
   request again.
5. Save timestamps, UIDs, statuses, and recovery latency in the drill receipt.

## Drill 2: saturate gateway admission

Goal: prove CPU replica scaling and bounded-overload behavior.

1. Run the `saturation` k6 scenario.
2. Capture queue depth, HPA/KEDA events, gateway replica counts, and 429 samples.
3. Require at least one scale-up event and at least one 429 carrying
   `Retry-After` after the bounded queue saturates.
4. Stop load and confirm the gateway returns to the configured minimum replica
   count after its cooldown.

## Drill 3: bad canary and rollback

Goal: exercise a release decision, not merely edit a manifest.

1. Start GPTQ-int4 and BF16 together using the two time-sliced GPU allocations.
2. Move traffic through the recorded 10%, 50%, and 100% router configurations,
   checking the SLO queries at every step.
3. Route the promoted path to the controlled HTTP-500 upstream.
4. Require the availability burn-rate alert to enter `firing`.
5. Roll back to the stable GPTQ-int4 router configuration and require a verified
   request to recover.
6. Preserve the unhealthy interval and alert payload; never erase the failed
   canary evidence.

## Availability burn

`ForgeAvailabilityBurnRate` combines 30-second and 2-minute error ratios against
the 99% availability objective. The injected HTTP-500 upstream keeps `/health`
healthy so the request reaches the upstream and is counted as a real gateway 5xx.

## Latency burn

`ForgeLatencyBurnRate` combines 30-second and 2-minute ratios of requests above
2.5 seconds against a 95% latency objective. The 2.5-second boundary is an
actually exported gateway histogram bucket; the controlled latency upstream
sleeps for three seconds and still returns HTTP 200, isolating latency from
availability.

## Shutdown

1. Restore the stable router configuration and set GPU demand to zero.
2. Stop all port-forwards and test Jobs.
3. Scale both vLLM deployments to zero and verify `nvidia-smi` has no compute PID.
4. Stop k3s and any service started by this phase.
5. Capture the final listener/GPU/process receipt, sync evidence, then power off
   the VM with `sudo shutdown -h now`.
