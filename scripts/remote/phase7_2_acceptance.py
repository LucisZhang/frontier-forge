#!/usr/bin/env python3
"""Run and seal the real-VM Phase 7.2 acceptance drills.

This runner deliberately talks only to loopback-bound kubectl port-forwards.
It never creates public Services, NodePorts, hostPorts, or security-group rules.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import statistics
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from forge.bench.loadgen import normalize_verifier_input
from forge.teacher.filters import breakdown_dict
from forge.train.artifacts import append_jsonl_once, sha256_tree, write_json_atomic
from forge.train.config import REPO_ROOT, relative_path, sha256_file
from forge.verify.verifier import score

RESULTS = REPO_ROOT / "results/phase7_2"
RAW = RESULTS / "raw"
LOGS = RESULTS / "logs"
K6_RESULTS = RESULTS / "k6"
NAMESPACE = "forge-system"
GATEWAY_URL = "http://127.0.0.1:19000"
PROMETHEUS_URL = "http://127.0.0.1:19090"
PUSHGATEWAY_URL = "http://127.0.0.1:19091"
GRAFANA_URL = "http://127.0.0.1:13000"
PREFERRED_REQUEST_ID = "p4-0088-5946494"
WORKLOAD_RELATIVE = "data/full/phase4/workload-9ed3d99a9d75c357.jsonl"
WORKLOAD = REPO_ROOT / WORKLOAD_RELATIVE
if not WORKLOAD.is_file():
    WORKLOAD = (
        Path(os.environ.get("FORGE_PHASE7_2_SOURCE_REPO", "/mnt/frontier-forge/repo"))
        / WORKLOAD_RELATIVE
    )
HOURLY_USD = 1.53


def now() -> str:
    return datetime.now(UTC).isoformat()


def run(
    command: list[str],
    *,
    input_text: str | None = None,
    timeout: float = 300,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def kube(*args: str, input_text: str | None = None, timeout: float = 300) -> str:
    return run(
        ["sudo", "k3s", "kubectl", *args],
        input_text=input_text,
        timeout=timeout,
    ).stdout


def kube_json(*args: str) -> dict[str, Any]:
    return json.loads(kube(*args, "-o", "json"))


def git_sha() -> str:
    return run(["git", "rev-parse", "HEAD"]).stdout.strip()


def wait_until(
    description: str,
    predicate: Callable[[], Any],
    *,
    timeout_s: float,
    interval_s: float = 2,
) -> Any:
    deadline = time.monotonic() + timeout_s
    last: Any = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval_s)
    raise TimeoutError(f"timed out waiting for {description}; last={last!r}")


def deployment_snapshot(name: str) -> dict[str, Any]:
    value = kube_json("-n", NAMESPACE, "get", "deployment", name)
    status = value.get("status", {})
    return {
        "name": name,
        "generation": value["metadata"].get("generation"),
        "replicas": int(status.get("replicas", 0)),
        "ready_replicas": int(status.get("readyReplicas", 0)),
        "available_replicas": int(status.get("availableReplicas", 0)),
        "desired_replicas": int(value["spec"].get("replicas", 0)),
        "image": value["spec"]["template"]["spec"]["containers"][0]["image"],
    }


def wait_deployment(name: str, replicas: int, *, timeout_s: float = 900) -> dict[str, Any]:
    def ready() -> dict[str, Any] | None:
        item = deployment_snapshot(name)
        if item["desired_replicas"] == replicas and item["ready_replicas"] == replicas:
            return item
        return None

    return wait_until(f"deployment/{name} ready replicas={replicas}", ready, timeout_s=timeout_s)


def wait_gpu_zero(*, timeout_s: float = 300) -> dict[str, Any]:
    def zero() -> dict[str, Any] | None:
        pods = kube_json("-n", NAMESPACE, "get", "pods", "-l", "forge.openai.com/precision")[
            "items"
        ]
        processes = run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
        ).stdout.strip()
        if not pods and not processes:
            return {"at": now(), "pods": 0, "compute_processes": ""}
        return None

    return wait_until(
        "zero vLLM pods and zero GPU compute processes",
        zero,
        timeout_s=timeout_s,
        interval_s=2,
    )


def apply_object(value: dict[str, Any]) -> None:
    kube("apply", "-f", "-", input_text=json.dumps(value, separators=(",", ":")))


def prometheus(query: str) -> dict[str, Any]:
    response = httpx.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=10,
        trust_env=False,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {payload}")
    return payload["data"]


def prom_scalar(query: str, *, default: float = 0.0) -> float:
    result = prometheus(query).get("result", [])
    if not result:
        return default
    return float(result[0]["value"][1])


def push_gpu_demand(value: int) -> dict[str, Any]:
    body = f"# TYPE forge_gpu_scale_demand gauge\nforge_gpu_scale_demand {value}\n"
    response = httpx.put(
        f"{PUSHGATEWAY_URL}/metrics/job/forge_gpu_scale",
        content=body,
        headers={"Content-Type": "text/plain; version=0.0.4"},
        timeout=10,
        trust_env=False,
    )
    response.raise_for_status()
    return {"at": now(), "value": value, "http_status": response.status_code}


def load_probe_rows() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in WORKLOAD.read_text().splitlines() if line]
    preferred = [row for row in rows if row.get("request_id") == PREFERRED_REQUEST_ID]
    if len(preferred) != 1:
        raise RuntimeError(f"missing preferred verifier row {PREFERRED_REQUEST_ID}")
    return preferred + [row for row in rows if row.get("request_id") != PREFERRED_REQUEST_ID]


def request_once(row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": "forge-r1b",
        "messages": row["messages"],
        "temperature": 0,
        "max_tokens": min(256, int(row.get("max_tokens", 256))),
        "stream": False,
    }
    started = time.monotonic()
    try:
        response = httpx.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json=payload,
            headers={
                "X-Client-ID": "phase7-2-acceptance",
                "X-Allow-Degrade": "0",
                "X-Request-Timeout-Ms": "120000",
            },
            timeout=130,
            trust_env=False,
        )
    except httpx.HTTPError as error:
        return {
            "at": now(),
            "elapsed_s": time.monotonic() - started,
            "request_id": row["request_id"],
            "http_status": None,
            "error": f"{type(error).__name__}: {error}",
            "verified": False,
        }
    result: dict[str, Any] = {
        "at": now(),
        "elapsed_s": time.monotonic() - started,
        "request_id": row["request_id"],
        "http_status": response.status_code,
        "retry_after": response.headers.get("retry-after"),
        "forge_route": response.headers.get("x-forge-route"),
        "forge_upstream": response.headers.get("x-forge-upstream"),
        "verified": False,
    }
    try:
        value = response.json()
    except ValueError:
        result["body"] = response.text[:2000]
        return result
    if response.status_code != 200:
        result["body"] = value
        return result
    try:
        output = value["choices"][0]["message"]["content"]
        verifier_input = normalize_verifier_input(output)
        scored = breakdown_dict(score({"label": row["label"]}, verifier_input))
    except (KeyError, IndexError, TypeError, ValueError) as error:
        result["parse_error"] = f"{type(error).__name__}: {error}"
        result["body"] = value
        return result
    result.update(
        {
            "output": output,
            "verifier_input": verifier_input,
            "verifier": scored,
            "verified": bool(scored["task_success"]),
        }
    )
    return result


def wait_verified_request(*, timeout_s: float = 300) -> dict[str, Any]:
    rows = load_probe_rows()
    deadline = time.monotonic() + timeout_s
    attempts: list[dict[str, Any]] = []
    index = 0
    while time.monotonic() < deadline:
        try:
            ready = (
                httpx.get(f"{GATEWAY_URL}/readyz", timeout=3, trust_env=False).status_code == 200
            )
        except httpx.HTTPError:
            ready = False
        if not ready:
            time.sleep(1)
            continue
        attempt = request_once(rows[index % min(20, len(rows))])
        attempts.append(attempt)
        if attempt["verified"]:
            return {"successful": attempt, "attempts": attempts}
        index += 1
        time.sleep(0.5)
    raise TimeoutError(f"no verified request before timeout; attempts={attempts[-5:]}")


def set_router(variant: str) -> dict[str, Any]:
    path = REPO_ROOT / f"deploy/phase7_2/real/router/{variant}.conf"
    if not path.is_file():
        raise FileNotFoundError(path)
    apply_object(
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "vllm-router-config", "namespace": NAMESPACE},
            "data": {"default.conf": path.read_text()},
        }
    )
    kube("-n", NAMESPACE, "rollout", "restart", "deployment/vllm-router")
    kube(
        "-n",
        NAMESPACE,
        "rollout",
        "status",
        "deployment/vllm-router",
        "--timeout=180s",
        timeout=200,
    )
    return {"at": now(), "variant": variant, "sha256": sha256_file(path)}


def alerts() -> list[dict[str, Any]]:
    response = httpx.get(f"{PROMETHEUS_URL}/api/v1/alerts", timeout=10, trust_env=False)
    response.raise_for_status()
    return response.json()["data"]["alerts"]


def firing_alert(name: str) -> dict[str, Any] | None:
    for item in alerts():
        if item.get("labels", {}).get("alertname") == name and item.get("state") == "firing":
            return item
    return None


def wait_alert(name: str, *, timeout_s: float = 180) -> dict[str, Any]:
    return wait_until(
        f"alert {name} firing", lambda: firing_alert(name), timeout_s=timeout_s, interval_s=5
    )


def wait_slo_clear(*, timeout_s: float = 300) -> list[dict[str, Any]]:
    names = {"ForgeAvailabilityBurnRate", "ForgeLatencyBurnRate"}

    def clear() -> dict[str, list[dict[str, Any]]] | None:
        active = [
            item
            for item in alerts()
            if item.get("labels", {}).get("alertname") in names and item.get("state") == "firing"
        ]
        return {"active": []} if not active else None

    result = wait_until("both Forge SLO alerts clear", clear, timeout_s=timeout_s, interval_s=5)
    return result["active"]


def apply_k6_config() -> None:
    script = (REPO_ROOT / "deploy/phase7_2/k6/scenarios.js").read_text()
    apply_object(
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "forge-k6-scenarios", "namespace": NAMESPACE},
            "data": {"scenarios.js": script},
        }
    )


def run_k6(
    name: str,
    *,
    scenario: str,
    duration: str,
    rate: int,
    sample: Callable[[], dict[str, Any]] | None = None,
    timeout_s: float = 600,
) -> dict[str, Any]:
    apply_k6_config()
    kube("-n", NAMESPACE, "delete", "job", name, "--ignore-not-found=true", "--wait=true")
    apply_object(
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": name, "namespace": NAMESPACE},
            "spec": {
                "backoffLimit": 0,
                "ttlSecondsAfterFinished": 3600,
                "template": {
                    "metadata": {
                        "labels": {"app.kubernetes.io/name": "forge-k6", "scenario": scenario}
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "automountServiceAccountToken": False,
                        "securityContext": {
                            "runAsNonRoot": True,
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                        "containers": [
                            {
                                "name": "k6",
                                "image": "grafana/k6:2.2.0",
                                "imagePullPolicy": "IfNotPresent",
                                "args": [
                                    "run",
                                    "--summary-export=/tmp/summary.json",
                                    "/scripts/scenarios.js",
                                ],
                                "env": [
                                    {"name": "SCENARIO", "value": scenario},
                                    {"name": "DURATION", "value": duration},
                                    {"name": "RATE", "value": str(rate)},
                                ],
                                "securityContext": {
                                    "allowPrivilegeEscalation": False,
                                    "capabilities": {"drop": ["ALL"]},
                                },
                                "volumeMounts": [
                                    {
                                        "name": "script",
                                        "mountPath": "/scripts",
                                        "readOnly": True,
                                    }
                                ],
                                "resources": {
                                    "requests": {"cpu": "100m", "memory": "64Mi"},
                                    "limits": {"cpu": "2", "memory": "512Mi"},
                                },
                            }
                        ],
                        "volumes": [
                            {"name": "script", "configMap": {"name": "forge-k6-scenarios"}}
                        ],
                    },
                },
            },
        }
    )
    started_at = now()
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    condition = ""
    while time.monotonic() - started < timeout_s:
        job = kube_json("-n", NAMESPACE, "get", "job", name)
        for item in job.get("status", {}).get("conditions", []):
            if item.get("status") == "True" and item.get("type") in {"Complete", "Failed"}:
                condition = item["type"]
        if sample is not None:
            try:
                samples.append({"at": now(), **sample()})
            except (RuntimeError, httpx.HTTPError, ValueError) as error:
                samples.append({"at": now(), "sample_error": str(error)})
        if condition:
            break
        time.sleep(2)
    if not condition:
        raise TimeoutError(f"k6 job/{name} did not finish")
    pod = kube_json("-n", NAMESPACE, "get", "pods", "-l", f"job-name={name}")["items"][0][
        "metadata"
    ]["name"]
    log = kube("-n", NAMESPACE, "logs", pod, timeout=60)
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{name}.log").write_text(log)
    marker = "FORGE_K6_SUMMARY_JSON "
    summary_lines = [line for line in log.splitlines() if line.startswith(marker)]
    summary = json.loads(summary_lines[-1][len(marker) :]) if summary_lines else None
    receipt = {
        "name": name,
        "scenario": scenario,
        "duration": duration,
        "rate": rate,
        "started_at": started_at,
        "finished_at": now(),
        "wall_seconds": time.monotonic() - started,
        "condition": condition,
        "pod": pod,
        "samples": samples,
        "summary": summary,
        "log": relative_path(LOGS / f"{name}.log"),
    }
    K6_RESULTS.mkdir(parents=True, exist_ok=True)
    write_json_atomic(K6_RESULTS / f"{name}.json", receipt)
    if condition != "Complete":
        raise RuntimeError(f"k6 job/{name} failed; see {receipt['log']}")
    return receipt


def k6_count(receipt: dict[str, Any], metric: str) -> int:
    summary = receipt.get("summary") or {}
    values = summary.get("metrics", {}).get(metric, {}).get("values", {})
    return int(values.get("count", 0))


def command_k6_smoke(_: argparse.Namespace) -> None:
    receipt = run_k6(
        "forge-k6-smoke",
        scenario="fault",
        duration="5s",
        rate=1,
        timeout_s=90,
    )
    if receipt.get("summary") is None:
        raise RuntimeError("k6 smoke completed without the machine-readable summary marker")
    write_json_atomic(
        RAW / "k6_smoke.json",
        {
            "version": 1,
            "phase": "7.2",
            "status": "complete",
            "captured_at": now(),
            "k6_receipt": relative_path(K6_RESULTS / "forge-k6-smoke.json"),
        },
    )


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile input")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def command_inventory(_: argparse.Namespace) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    services = kube_json("get", "services", "--all-namespaces")["items"]
    node = kube_json("get", "node")
    node_item = node["items"][0]
    namespace = kube_json("get", "namespace", NAMESPACE)
    persistent_volumes = [
        item
        for item in kube_json("get", "persistentvolumes")["items"]
        if item["metadata"]["name"] in {"forge-model-checkpoints", "forge-huggingface-archive"}
    ]
    persistent_volume_claims = kube_json("-n", NAMESPACE, "get", "persistentvolumeclaims")["items"]
    grafana_secret = kube_json("-n", "monitoring", "get", "secret", "monitoring-grafana")
    username = base64.b64decode(grafana_secret["data"]["admin-user"]).decode()
    password = base64.b64decode(grafana_secret["data"]["admin-password"]).decode()
    dashboard_response = httpx.get(
        f"{GRAFANA_URL}/api/search",
        params={"query": "Frontier Forge"},
        auth=(username, password),
        timeout=10,
        trust_env=False,
    )
    dashboard_response.raise_for_status()
    images = run(["sudo", "k3s", "ctr", "images", "list"]).stdout
    listeners = [
        line
        for line in run(["ss", "-H", "-ltn"]).stdout.splitlines()
        if any(f":{port}" in line for port in (13000, 19000, 19090, 19091))
    ]
    systemd = run(["systemctl", "is-system-running"], check=False)
    cgroup = Path("/proc/1/cgroup").read_text().strip()
    swap_total_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("SwapTotal:")
    )
    runtime = {
        "git_sha": git_sha(),
        "k3s": run(["k3s", "--version"]).stdout.strip(),
        "kubernetes": kube("version", "-o", "json"),
        "helm_releases": run(
            [
                "sudo",
                "helm",
                "--kubeconfig",
                "/etc/rancher/k3s/k3s.yaml",
                "list",
                "-A",
                "-o",
                "json",
            ]
        ).stdout,
        "node": {
            "name": node_item["metadata"]["name"],
            "labels": node_item["metadata"].get("labels", {}),
            "capacity": node_item["status"]["capacity"],
            "allocatable": node_item["status"]["allocatable"],
        },
        "nvidia_smi": run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ).stdout.strip(),
        "host_preflight": {
            "systemd_state": systemd.stdout.strip(),
            "systemd_returncode": systemd.returncode,
            "pid1_cgroup": cgroup,
            "swap_total_kib": swap_total_kib,
        },
        "operator_port_forward_listeners": listeners,
        "model_storage": {
            "namespace_pod_security_enforce": namespace["metadata"]
            .get("labels", {})
            .get("pod-security.kubernetes.io/enforce"),
            "persistent_volumes": [
                {
                    "name": item["metadata"]["name"],
                    "phase": item.get("status", {}).get("phase"),
                    "local_path": item["spec"].get("local", {}).get("path"),
                    "claim_ref": item["spec"].get("claimRef"),
                    "access_modes": item["spec"].get("accessModes", []),
                    "reclaim_policy": item["spec"].get("persistentVolumeReclaimPolicy"),
                }
                for item in persistent_volumes
            ],
            "persistent_volume_claims": [
                {
                    "name": item["metadata"]["name"],
                    "phase": item.get("status", {}).get("phase"),
                    "volume_name": item["spec"].get("volumeName"),
                    "access_modes": item["spec"].get("accessModes", []),
                }
                for item in persistent_volume_claims
                if item["metadata"]["name"]
                in {"forge-model-checkpoints", "forge-huggingface-archive"}
            ],
        },
        "services": [
            {
                "name": item["metadata"]["name"],
                "namespace": item["metadata"]["namespace"],
                "type": item["spec"].get("type", "ClusterIP"),
                "ports": item["spec"].get("ports", []),
            }
            for item in services
        ],
        "phase_images": [
            line
            for line in images.splitlines()
            if any(
                key in line
                for key in ("frontier-forge", "vllm", "nginx", "pushgateway", "grafana/k6")
            )
        ],
        "prometheus_targets": httpx.get(
            f"{PROMETHEUS_URL}/api/v1/targets", timeout=10, trust_env=False
        ).json(),
        "prometheus_rules": httpx.get(
            f"{PROMETHEUS_URL}/api/v1/rules", timeout=10, trust_env=False
        ).json(),
        "grafana_dashboards": dashboard_response.json(),
        "model_artifacts": {
            "gptq_int4": {
                "path": "checkpoints/full/r1b/trl/s0/export/gptq_int4",
                "sha256": sha256_tree(REPO_ROOT / "checkpoints/full/r1b/trl/s0/export/gptq_int4"),
            },
            "bf16_mtp_preserved": {
                "path": "checkpoints/full/r1b/trl/s0/export/merged_bf16_mtp_preserved",
                "sha256": sha256_tree(
                    REPO_ROOT / "checkpoints/full/r1b/trl/s0/export/merged_bf16_mtp_preserved"
                ),
            },
        },
        "workload": {
            "path": WORKLOAD_RELATIVE,
            "resolved_path": str(WORKLOAD.resolve()),
            "sha256": sha256_file(WORKLOAD),
        },
        "disclosure": {
            "scope": "single-node k3s on one real VM and one physical NVIDIA A10",
            "gpu_sharing": "two time-sliced nvidia.com/gpu.shared allocations",
            "not_cloud_production": True,
            "not_multi_gpu": True,
            "kafka_out_of_scope": True,
            "dashboard_access": "ClusterIP plus loopback kubectl port-forward plus SSH tunnel only",
        },
    }
    checks = {
        "real_systemd_vm": runtime["host_preflight"]["systemd_returncode"] == 0,
        "root_cgroup_not_container": runtime["host_preflight"]["pid1_cgroup"].startswith("0::/")
        and not any(
            marker in runtime["host_preflight"]["pid1_cgroup"]
            for marker in ("docker", "containerd", "kubepods")
        ),
        "swap_disabled": runtime["host_preflight"]["swap_total_kib"] == 0,
        "one_physical_a10": runtime["nvidia_smi"].startswith("NVIDIA A10,"),
        "two_time_sliced_allocations": node_item["status"]["capacity"].get("nvidia.com/gpu.shared")
        == "2",
        "pod_security_baseline": runtime["model_storage"]["namespace_pod_security_enforce"]
        == "baseline",
        "model_store_node_labeled": runtime["node"]["labels"].get("forge.openai.com/model-store")
        == "true",
        "local_model_pvs_bound": len(runtime["model_storage"]["persistent_volumes"]) == 2
        and all(
            item["phase"] == "Bound"
            and item["local_path"]
            in {
                "/mnt/frontier-forge/repo/checkpoints",
                "/mnt/frontier-forge/cache/huggingface",
            }
            and item["reclaim_policy"] == "Retain"
            for item in runtime["model_storage"]["persistent_volumes"]
        ),
        "model_pvcs_bound": len(runtime["model_storage"]["persistent_volume_claims"]) == 2
        and all(
            item["phase"] == "Bound"
            and item["volume_name"] in {"forge-model-checkpoints", "forge-huggingface-archive"}
            for item in runtime["model_storage"]["persistent_volume_claims"]
        ),
        "all_services_cluster_ip": all(item["type"] == "ClusterIP" for item in runtime["services"]),
        "operator_ports_loopback_only": len(listeners) == 4
        and all("127.0.0.1:" in line for line in listeners),
        "gptq_hash": runtime["model_artifacts"]["gptq_int4"]["sha256"]
        == "c99b42cf0e062cc75f2df8588725d0c29383666f3db0c1ae837ce15bfe6d39d2",
        "bf16_hash": runtime["model_artifacts"]["bf16_mtp_preserved"]["sha256"]
        == "7878b55f6fe6a9ecb12b9504b1a88d7bc6fef7ba72d91289b6e8d694f6bc75ce",
        "dashboard_loaded": any(
            item.get("uid") == "frontier-forge-phase7-2" for item in runtime["grafana_dashboards"]
        ),
    }
    receipt = {
        "version": 1,
        "phase": "7.2",
        "status": "complete" if all(checks.values()) else "failed",
        "captured_at": now(),
        "checks": checks,
        "runtime": runtime,
    }
    write_json_atomic(RAW / "inventory.json", receipt)
    if receipt["status"] != "complete":
        raise RuntimeError(f"inventory checks failed: {checks}")


def command_cold_start(args: argparse.Namespace) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    set_router("stable")
    kube("-n", NAMESPACE, "scale", "deployment/vllm-bf16", "--replicas=0")
    wait_deployment("vllm-bf16", 0)
    runs: list[dict[str, Any]] = []
    path = RAW / "gpu_cold_start.json"
    for index in range(1, args.iterations + 1):
        push_gpu_demand(0)
        wait_deployment("vllm-int4", 0, timeout_s=300)
        zero_state = wait_gpu_zero(timeout_s=300)
        trigger_wall = now()
        trigger = time.monotonic()
        pushed = push_gpu_demand(1)
        wait_deployment("vllm-int4", 1, timeout_s=900)
        verified = wait_verified_request(timeout_s=300)
        elapsed = time.monotonic() - trigger
        pods = kube_json("-n", NAMESPACE, "get", "pods", "-l", "app.kubernetes.io/name=vllm-int4")[
            "items"
        ]
        run_receipt = {
            "iteration": index,
            "trigger_at": trigger_wall,
            "pre_trigger_zero_state": zero_state,
            "metric_push": pushed,
            "first_verified_at": now(),
            "trigger_to_first_verified_s": elapsed,
            "pod": {
                "name": pods[0]["metadata"]["name"],
                "uid": pods[0]["metadata"]["uid"],
                "node": pods[0]["spec"]["nodeName"],
            },
            "verified_request": verified,
            "nvidia_smi": run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory,process_name",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
            ).stdout.strip(),
        }
        runs.append(run_receipt)
        write_json_atomic(
            path,
            {
                "version": 1,
                "phase": "7.2",
                "status": "running",
                "iterations_required": args.iterations,
                "runs": runs,
            },
        )
        push_gpu_demand(0)
        wait_deployment("vllm-int4", 0, timeout_s=300)
        wait_gpu_zero(timeout_s=300)
    durations = [float(item["trigger_to_first_verified_s"]) for item in runs]
    receipt = {
        "version": 1,
        "phase": "7.2",
        "status": "complete",
        "iterations_required": args.iterations,
        "iterations_completed": len(runs),
        "trigger": "Prometheus forge_gpu_scale_demand via KEDA",
        "endpoint": "gateway first task-success verified response",
        "distribution_s": {
            "min": min(durations),
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "max": max(durations),
            "mean": statistics.fmean(durations),
        },
        "runs": runs,
        "finished_at": now(),
    }
    write_json_atomic(path, receipt)
    push_gpu_demand(1)
    wait_deployment("vllm-int4", 1, timeout_s=900)


def command_gateway_scale(_: argparse.Namespace) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    push_gpu_demand(1)
    wait_deployment("vllm-int4", 1)
    wait_verified_request()
    wait_slo_clear()
    before = {
        "reject_overload": prom_scalar(
            'sum(forge_gateway_routing_decisions_total{decision="reject_overload"})'
        ),
        "replicas": deployment_snapshot("forge-gateway"),
    }

    def sample() -> dict[str, Any]:
        return {
            "replicas": deployment_snapshot("forge-gateway"),
            "queue_depth": prom_scalar("max(forge_gateway_queue_depth)"),
            "queue_high_watermark": prom_scalar("max(forge_gateway_queue_high_watermark)"),
        }

    k6 = run_k6(
        "forge-k6-saturation",
        scenario="saturation",
        duration="120s",
        rate=40,
        sample=sample,
        timeout_s=300,
    )
    after = {
        "reject_overload": prom_scalar(
            'sum(forge_gateway_routing_decisions_total{decision="reject_overload"})'
        ),
        "replicas": deployment_snapshot("forge-gateway"),
    }
    scaled_down = wait_deployment("forge-gateway", 1, timeout_s=300)
    replica_values = [
        int(item.get("replicas", {}).get("ready_replicas", 0))
        for item in k6["samples"]
        if isinstance(item.get("replicas"), dict)
    ]
    queue_values = [float(item["queue_depth"]) for item in k6["samples"] if "queue_depth" in item]
    high_watermarks = [
        float(item["queue_high_watermark"])
        for item in k6["samples"]
        if "queue_high_watermark" in item
    ]
    count_429 = k6_count(k6, "forge_http_429")
    retry_after = k6_count(k6, "forge_retry_after_429")
    checks = {
        "custom_metric_scale_up": max(replica_values, default=0) >= 2,
        "queue_saturated": max(high_watermarks, default=0) >= 24,
        "overload_decision_increased": after["reject_overload"] > before["reject_overload"],
        "k6_observed_429": count_429 > 0,
        "every_k6_429_had_retry_after": count_429 == retry_after,
        "scaled_back_to_one": scaled_down["ready_replicas"] == 1,
    }
    events = kube("-n", NAMESPACE, "get", "events", "--sort-by=.lastTimestamp")
    (LOGS / "keda_gateway_events.log").write_text(events)
    receipt = {
        "version": 1,
        "phase": "7.2",
        "status": "complete" if all(checks.values()) else "failed",
        "started_at": k6["started_at"],
        "finished_at": now(),
        "checks": checks,
        "before": before,
        "after": after,
        "scaled_down": scaled_down,
        "max_ready_replicas": max(replica_values, default=0),
        "max_queue_depth": max(queue_values, default=0),
        "max_queue_high_watermark": max(high_watermarks, default=0),
        "k6_counts": {"http_429": count_429, "retry_after_429": retry_after},
        "k6_receipt": relative_path(K6_RESULTS / "forge-k6-saturation.json"),
        "events_log": relative_path(LOGS / "keda_gateway_events.log"),
    }
    write_json_atomic(RAW / "gateway_keda_scale.json", receipt)
    write_json_atomic(
        RAW / "drill_saturate_arrival.json",
        {
            **receipt,
            "drill": "saturate_arrival_rate",
            "runbook": "docs/runbooks/phase7_2.md#drill-2-saturate-gateway-admission",
        },
    )
    if receipt["status"] != "complete":
        raise RuntimeError(f"gateway KEDA scale checks failed: {checks}")


def command_kill_drill(_: argparse.Namespace) -> None:
    push_gpu_demand(1)
    wait_deployment("vllm-int4", 1)
    before_request = wait_verified_request()
    pods = kube_json("-n", NAMESPACE, "get", "pods", "-l", "app.kubernetes.io/name=vllm-int4")[
        "items"
    ]
    old = {"name": pods[0]["metadata"]["name"], "uid": pods[0]["metadata"]["uid"]}
    deleted_at = now()
    started = time.monotonic()
    kube("-n", NAMESPACE, "delete", "pod", old["name"], "--wait=false")

    unavailable: list[dict[str, Any]] = []

    def failed_closed() -> dict[str, Any] | None:
        attempt = request_once(load_probe_rows()[0])
        unavailable.append(attempt)
        return attempt if attempt.get("http_status") == 503 else None

    observed_503 = wait_until(
        "gateway fail-closed 503 during vLLM replacement",
        failed_closed,
        timeout_s=120,
        interval_s=1,
    )

    def replacement_ready() -> dict[str, Any] | None:
        items = kube_json("-n", NAMESPACE, "get", "pods", "-l", "app.kubernetes.io/name=vllm-int4")[
            "items"
        ]
        for item in items:
            ready = any(
                condition.get("type") == "Ready" and condition.get("status") == "True"
                for condition in item.get("status", {}).get("conditions", [])
            )
            if item["metadata"]["uid"] != old["uid"] and ready:
                return {
                    "name": item["metadata"]["name"],
                    "uid": item["metadata"]["uid"],
                }
        return None

    new = wait_until("replacement vLLM pod", replacement_ready, timeout_s=900)
    recovered = wait_verified_request(timeout_s=300)
    receipt = {
        "version": 1,
        "phase": "7.2",
        "status": "complete",
        "drill": "kill_active_vllm_pod",
        "runbook": "docs/runbooks/phase7_2.md#drill-1-kill-the-active-vllm-pod",
        "deleted_at": deleted_at,
        "recovered_at": now(),
        "recovery_seconds": time.monotonic() - started,
        "old_pod": old,
        "new_pod": new,
        "before_verified_request": before_request,
        "failed_closed_observation": observed_503,
        "all_unavailable_probes": unavailable,
        "after_verified_request": recovered,
    }
    write_json_atomic(RAW / "drill_kill_vllm.json", receipt)


async def routing_requests(count: int) -> list[dict[str, Any]]:
    # This gate proves weighted routing while two models coexist on one
    # time-sliced physical GPU; the separate k6 drill owns concurrent load.
    # Sequential probes avoid turning this canary attribution check into an
    # undocumented multi-process GPU contention benchmark.
    semaphore = asyncio.Semaphore(1)
    payload = {
        "model": "forge-r1b",
        "messages": [{"role": "user", "content": "Reply with the single token OK."}],
        "temperature": 0,
        "max_tokens": 1,
        "stream": False,
    }

    async def one(client: httpx.AsyncClient, index: int) -> dict[str, Any]:
        async with semaphore:
            started = time.monotonic()
            try:
                response = await client.post(
                    f"{GATEWAY_URL}/v1/chat/completions",
                    json=payload,
                    headers={"X-Client-ID": f"phase7-2-canary-{index}"},
                )
                return {
                    "index": index,
                    "status": response.status_code,
                    "elapsed_s": time.monotonic() - started,
                    "upstream": response.headers.get("x-forge-upstream"),
                    "retry_after": response.headers.get("retry-after"),
                }
            except httpx.HTTPError as error:
                return {
                    "index": index,
                    "status": None,
                    "elapsed_s": time.monotonic() - started,
                    "error": f"{type(error).__name__}: {error}",
                }

    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
        return await asyncio.gather(*(one(client, index) for index in range(count)))


def slo_guard() -> dict[str, Any]:
    firing = [
        item
        for item in alerts()
        if item.get("labels", {}).get("alertname")
        in {"ForgeAvailabilityBurnRate", "ForgeLatencyBurnRate"}
        and item.get("state") == "firing"
    ]
    return {
        "at": now(),
        "availability_5xx_ratio_30s": prom_scalar(
            'sum(rate(forge_gateway_responses_total{class="5xx"}[30s])) '
            "/ clamp_min(sum(rate(forge_gateway_responses_total[30s])), 0.001)"
        ),
        "p95_latency_seconds_1m": prom_scalar(
            "histogram_quantile(0.95, sum(rate("
            "forge_gateway_request_duration_seconds_bucket[1m])) by (le))"
        ),
        "firing_alerts": firing,
        "pass": not firing,
    }


def backend_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for variant in ("vllm-int4", "vllm-bf16"):
        service = kube_json("-n", NAMESPACE, "get", "service", variant)
        mapping[f"{service['spec']['clusterIP']}:8000"] = variant
        items = kube_json(
            "-n", NAMESPACE, "get", "pods", "-l", f"app.kubernetes.io/name={variant}"
        )["items"]
        for item in items:
            mapping[f"{item['status']['podIP']}:8000"] = variant
    return mapping


def wait_gateway_ready_stable(*, timeout_s: float = 120) -> dict[str, Any]:
    consecutive = 0

    def stable() -> dict[str, Any] | None:
        nonlocal consecutive
        try:
            response = httpx.get(f"{GATEWAY_URL}/readyz", timeout=3, trust_env=False)
            if response.status_code == 200:
                consecutive += 1
            else:
                consecutive = 0
        except httpx.HTTPError:
            consecutive = 0
        if consecutive >= 3:
            return {"at": now(), "consecutive_ready_probes": consecutive}
        return None

    return wait_until(
        "gateway ready for three consecutive probes after router rollout",
        stable,
        timeout_s=timeout_s,
        interval_s=1,
    )


def route_stage(variant: str, count: int) -> dict[str, Any]:
    config = set_router(variant)
    gateway_ready = wait_gateway_ready_stable()
    wait_slo_clear()
    observations = asyncio.run(routing_requests(count))
    mapping = backend_map()
    counts: dict[str, int] = {"vllm-int4": 0, "vllm-bf16": 0, "unknown": 0}
    for item in observations:
        backend = mapping.get(str(item.get("upstream")), "unknown")
        if item.get("status") == 200:
            counts[backend] += 1
    guard = slo_guard()
    return {
        "variant": variant,
        "router_config": config,
        "gateway_ready_after_rollout": gateway_ready,
        "request_concurrency": 1,
        "backend_map": mapping,
        "counts": counts,
        "http_statuses": {
            str(status): sum(1 for item in observations if item.get("status") == status)
            for status in sorted({item.get("status") for item in observations}, key=str)
        },
        "observations": observations,
        "slo_guard": guard,
    }


def run_fault_alert(
    *,
    router: str,
    alert_name: str,
    job_name: str,
    rate: int,
) -> dict[str, Any]:
    wait_slo_clear()
    config = set_router(router)
    k6 = run_k6(
        job_name,
        scenario="fault",
        duration="145s",
        rate=rate,
        timeout_s=300,
    )
    fired = wait_alert(alert_name, timeout_s=120)
    receipt = {
        "version": 1,
        "phase": "7.2",
        "status": "complete",
        "alert": alert_name,
        "injected_fault": router,
        "router_config": config,
        "firing_payload": fired,
        "all_alerts_at_capture": alerts(),
        "k6_receipt": relative_path(K6_RESULTS / f"{job_name}.json"),
        "k6_counts": {
            metric: k6_count(k6, metric)
            for metric in ("forge_http_200", "forge_http_500", "forge_http_503")
        },
        "captured_at": now(),
    }
    write_json_atomic(RAW / f"alert_{alert_name}.json", receipt)
    return receipt


def command_canary(_: argparse.Namespace) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    push_gpu_demand(1)
    kube(
        "-n",
        NAMESPACE,
        "annotate",
        "scaledobject/vllm-int4",
        "autoscaling.keda.sh/paused=true",
        "--overwrite",
    )
    kube("-n", NAMESPACE, "scale", "deployment/vllm-int4", "--replicas=1")
    kube("-n", NAMESPACE, "scale", "deployment/vllm-bf16", "--replicas=1")
    wait_deployment("vllm-int4", 1)
    wait_deployment("vllm-bf16", 1)
    wait_verified_request()
    wait_slo_clear()
    coexistence = {
        "at": now(),
        "int4": deployment_snapshot("vllm-int4"),
        "bf16": deployment_snapshot("vllm-bf16"),
        "pods": kube_json("-n", NAMESPACE, "get", "pods", "-l", "forge.openai.com/precision"),
        "nvidia_smi": run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory,process_name",
                "--format=csv,noheader,nounits",
            ]
        ).stdout.strip(),
    }
    stages = [
        route_stage("canary-10", 50),
        route_stage("canary-50", 40),
        route_stage("canary-100", 20),
    ]
    promoted_verified = wait_verified_request()
    stage_checks = {
        "canary_10_both_and_stable_majority": stages[0]["counts"]["vllm-bf16"] > 0
        and stages[0]["counts"]["vllm-int4"] > stages[0]["counts"]["vllm-bf16"],
        "canary_50_both": stages[1]["counts"]["vllm-bf16"] > 0
        and stages[1]["counts"]["vllm-int4"] > 0,
        "canary_100_bf16_only": stages[2]["counts"]["vllm-bf16"] > 0
        and stages[2]["counts"]["vllm-int4"] == 0,
        "all_promotion_guards_passed": all(item["slo_guard"]["pass"] for item in stages),
        "promoted_bf16_verified": promoted_verified["successful"]["verified"],
        "coexisting_ready": coexistence["int4"]["ready_replicas"] == 1
        and coexistence["bf16"]["ready_replicas"] == 1,
    }
    if not all(stage_checks.values()):
        write_json_atomic(
            RAW / "canary_release.json",
            {
                "version": 1,
                "phase": "7.2",
                "status": "failed",
                "checks": stage_checks,
                "coexistence": coexistence,
                "promotion_stages": stages,
            },
        )
        raise RuntimeError(f"canary promotion checks failed: {stage_checks}")
    availability = run_fault_alert(
        router="fault-500",
        alert_name="ForgeAvailabilityBurnRate",
        job_name="forge-k6-fault-availability",
        rate=3,
    )
    rollback_config = set_router("stable")
    rollback_verified = wait_verified_request(timeout_s=300)
    rollback = {
        "at": now(),
        "router_config": rollback_config,
        "verified_request": rollback_verified,
        "pass": rollback_verified["successful"]["verified"],
    }
    checks = {**stage_checks, "availability_alert_fired": True, "rollback_verified": True}
    receipt = {
        "version": 1,
        "phase": "7.2",
        "status": "complete" if all(checks.values()) else "failed",
        "checks": checks,
        "coexistence": coexistence,
        "promotion_stages": stages,
        "promoted_verified_request": promoted_verified,
        "degraded_canary": availability,
        "rollback": rollback,
        "finished_at": now(),
    }
    write_json_atomic(RAW / "canary_release.json", receipt)
    write_json_atomic(
        RAW / "drill_bad_canary_rollback.json",
        {
            **receipt,
            "drill": "bad_canary_rollback",
            "runbook": "docs/runbooks/phase7_2.md#drill-3-bad-canary-and-rollback",
        },
    )
    kube("-n", NAMESPACE, "scale", "deployment/vllm-bf16", "--replicas=0")
    wait_deployment("vllm-bf16", 0)
    kube(
        "-n",
        NAMESPACE,
        "annotate",
        "scaledobject/vllm-int4",
        "autoscaling.keda.sh/paused-",
    )
    push_gpu_demand(1)
    wait_deployment("vllm-int4", 1)


def command_latency_alert(_: argparse.Namespace) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    push_gpu_demand(1)
    wait_deployment("vllm-int4", 1)
    receipt = run_fault_alert(
        router="fault-latency",
        alert_name="ForgeLatencyBurnRate",
        job_name="forge-k6-fault-latency",
        rate=2,
    )
    receipt["restore"] = set_router("stable")
    receipt["restored_verified_request"] = wait_verified_request(timeout_s=300)
    write_json_atomic(RAW / "alert_ForgeLatencyBurnRate.json", receipt)


def write_report(final: dict[str, Any], payloads: dict[str, dict[str, Any]]) -> Path:
    cold = payloads["cold_start"]
    gateway = payloads["gateway_scale"]
    canary = payloads["canary"]
    availability = payloads["availability_alert"]
    latency = payloads["latency_alert"]
    kill = payloads["drill_kill"]
    distribution = cold["distribution_s"]
    stage_rows = []
    for stage in canary["promotion_stages"]:
        stage_rows.append(
            "| {variant} | {int4} | {bf16} | {unknown} | {guard} |".format(
                variant=stage["variant"],
                int4=stage["counts"]["vllm-int4"],
                bf16=stage["counts"]["vllm-bf16"],
                unknown=stage["counts"]["unknown"],
                guard="pass" if stage["slo_guard"]["pass"] else "fail",
            )
        )
    checks = final["gate"]["checks"]
    checklist = "\n".join(
        f"- [{'x' if passed else ' '}] {name.replace('_', ' ')}" for name, passed in checks.items()
    )
    cold_row = (
        f"| trigger → verified response | {distribution['min']:.3f} | "
        f"{distribution['p50']:.3f} | {distribution['p95']:.3f} | "
        f"{distribution['max']:.3f} | {distribution['mean']:.3f} |"
    )
    report = f"""# Phase 7.2 single-node k3s acceptance report

Status: **Gate 7.2 PASS**. Run `{final["run_id"]}` at git
`{final["git_sha"]}`. Raw receipt:
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

- Gateway custom-metric KEDA scale: 1 → **{gateway["max_ready_replicas"]}**
  ready replicas, then back to 1 after cooldown.
- Saturation evidence: queue high-watermark **{gateway["max_queue_high_watermark"]:.0f}/24**;
  k6 observed **{gateway["k6_counts"]["http_429"]}** HTTP 429 responses and every
  one carried `Retry-After`.
- GPU scale-to-zero/from-zero: **n={cold["iterations_completed"]}**, measured from
  the Prometheus demand trigger through the first task-success-verified gateway
  response.

| cold-start seconds | min | p50 | p95 | max | mean |
|---|---:|---:|---:|---:|---:|
{cold_row}

## One-GPU canary

Both deployments were Ready concurrently under the two time-sliced allocations.
The router exposed the selected upstream address in a response header so the
weighted stages were measured rather than inferred from configuration alone.
These attribution probes were sequential because both servers time-share one
physical GPU; concurrent saturation is covered separately by the k6/KEDA drill.

| router stage | int4 responses | BF16 responses | unknown | SLO guard |
|---|---:|---:|---:|---:|
{chr(10).join(stage_rows)}

After 100% BF16 promotion, a controlled HTTP-500 upstream deliberately degraded
the promoted path. `ForgeAvailabilityBurnRate` entered `firing`; the router was
then rolled back to stable GPTQ-int4 and a task-success-verified request recovered.

## Alerts and runbook drills

- Availability multi-window burn alert: fired under the `fault-500` k6 scenario
  ({availability["k6_counts"]["forge_http_500"]} observed HTTP 500 responses).
- Latency multi-window burn alert: fired under the three-second latency injector
  ({latency["k6_counts"]["forge_http_200"]} observed HTTP 200 responses), isolating
  latency from availability.
- Kill-vLLM drill: old pod `{kill["old_pod"]["uid"]}` was replaced by
  `{kill["new_pod"]["uid"]}`; the gateway failed closed with HTTP 503 and recovered
  a verified request in **{kill["recovery_seconds"]:.3f} s**.
- Saturation and bad-canary rollback drills reuse the same k6/KEDA and alert
  receipts above; all three are scripted by
  `scripts/remote/phase7_2_acceptance.py` and documented in
  `docs/runbooks/phase7_2.md`.

## CI and cost

- CPU-only kind manifest smoke and the full test job passed in
  {final["ci"]["run_url"]} at `{final["ci"]["tested_git_sha"]}`.
- VM rate: `FORGE_GPU_HOURLY_USD=1.53` (¥11/hour at 7.2 CNY/USD).
- Phase 7.2 delegated VM interval: **{final["cost"]["gpu_hours"]:.4f} h =
  ${final["cost"]["usd"]:.4f}**. This records the whole Phase 7.2 interval after
  Gate 7.1 finalization, including cluster/image setup and idle orchestration time.

## Gate 7.2

{checklist}

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
"""
    path = REPO_ROOT / "results/phase7_2_k3s_report.md"
    path.write_text(report)
    return path


def command_finalize(args: argparse.Namespace) -> None:
    required = {
        "inventory": RAW / "inventory.json",
        "cold_start": RAW / "gpu_cold_start.json",
        "gateway_scale": RAW / "gateway_keda_scale.json",
        "canary": RAW / "canary_release.json",
        "availability_alert": RAW / "alert_ForgeAvailabilityBurnRate.json",
        "latency_alert": RAW / "alert_ForgeLatencyBurnRate.json",
        "drill_kill": RAW / "drill_kill_vllm.json",
        "drill_saturate": RAW / "drill_saturate_arrival.json",
        "drill_canary": RAW / "drill_bad_canary_rollback.json",
    }
    payloads = {name: json.loads(path.read_text()) for name, path in required.items()}
    checks = {
        "inventory_complete": payloads["inventory"]["status"] == "complete",
        "keda_scale_event_receipt": payloads["gateway_scale"]["status"] == "complete",
        "cold_start_n_at_least_10": payloads["cold_start"]["status"] == "complete"
        and payloads["cold_start"]["iterations_completed"] >= 10,
        "canary_promote_and_rollback": payloads["canary"]["status"] == "complete",
        "availability_alert_fault_fired": payloads["availability_alert"]["status"] == "complete",
        "latency_alert_fault_fired": payloads["latency_alert"]["status"] == "complete",
        "three_runbook_drills_complete": all(
            payloads[name]["status"] == "complete"
            for name in ("drill_kill", "drill_saturate", "drill_canary")
        ),
        "kind_ci_green": args.ci_conclusion == "success",
        "single_node_disclosure": payloads["inventory"]["runtime"]["disclosure"][
            "not_cloud_production"
        ],
    }
    if not all(checks.values()):
        raise RuntimeError(f"cannot finalize failed Gate 7.2 checks: {checks}")
    finished_at = now()
    started = datetime.fromisoformat(args.session_start)
    finished = datetime.fromisoformat(finished_at)
    gpu_hours = (finished - started).total_seconds() / 3600
    artifacts = {
        name: {"path": relative_path(path), "sha256": sha256_file(path)}
        for name, path in required.items()
    }
    final = {
        "version": 1,
        "phase": "7.2",
        "run_id": "phase7_2_single_node_k3s_a10",
        "status": "complete",
        "git_sha": git_sha(),
        "started_at": args.session_start,
        "finished_at": finished_at,
        "gate": {"status": "pass", "checks": checks},
        "ci": {
            "conclusion": args.ci_conclusion,
            "run_url": args.ci_run_url,
            "tested_git_sha": args.ci_git_sha,
        },
        "cold_start_distribution_s": payloads["cold_start"]["distribution_s"],
        "gateway_scale": {
            "max_ready_replicas": payloads["gateway_scale"]["max_ready_replicas"],
            "http_429": payloads["gateway_scale"]["k6_counts"]["http_429"],
        },
        "canary_checks": payloads["canary"]["checks"],
        "alerts": ["ForgeAvailabilityBurnRate", "ForgeLatencyBurnRate"],
        "artifacts": artifacts,
        "disclosure": payloads["inventory"]["runtime"]["disclosure"],
        "cost": {
            "hourly_usd": HOURLY_USD,
            "gpu_hours": gpu_hours,
            "usd": gpu_hours * HOURLY_USD,
            "rate_source": "FORGE_GPU_HOURLY_USD=1.53; CNY 11/h at 7.2 CNY/USD",
            "scope": "entire Phase 7.2 delegated VM session after Gate 7.1 finalization",
        },
    }
    report_path = write_report(final, payloads)
    final["artifacts"]["report"] = {
        "path": relative_path(report_path),
        "sha256": sha256_file(report_path),
    }
    final_path = RAW / "phase7_2_acceptance.json"
    write_json_atomic(final_path, final)
    ledger = {
        "ledger_id": "phase7_2_single_node_k3s_a10",
        "phase": "7.2",
        "operation": "single_node_k3s_gpu_runtime_acceptance",
        "status": "complete",
        "gate_status": "pass",
        "git_sha": git_sha(),
        "started_at": args.session_start,
        "finished_at": finished_at,
        "gpu_type": "NVIDIA A10",
        "gpu_hours": gpu_hours,
        "hourly_usd": HOURLY_USD,
        "usd": gpu_hours * HOURLY_USD,
        "rate_source": final["cost"]["rate_source"],
        "receipt": relative_path(final_path),
        "notes": "Single-node/one-GPU k3s evidence; not cloud-production evidence.",
    }
    append_jsonl_once(RESULTS / "gpu_ledger.jsonl", ledger, key="ledger_id")
    run_record = {
        "run_id": final["run_id"],
        "phase": "7.2",
        "status": "complete",
        "git_sha": git_sha(),
        "gate": final["gate"],
        "metrics": {
            "cold_start_distribution_s": final["cold_start_distribution_s"],
            "gateway_max_ready_replicas": final["gateway_scale"]["max_ready_replicas"],
            "gateway_k6_http_429": final["gateway_scale"]["http_429"],
        },
        "cost": final["cost"],
        "raw_artifact": relative_path(final_path),
        "disclosure": final["disclosure"],
    }
    append_jsonl_once(REPO_ROOT / "results/runs.jsonl", run_record, key="run_id")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    subparsers = value.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inventory").set_defaults(function=command_inventory)
    subparsers.add_parser("k6-smoke").set_defaults(function=command_k6_smoke)
    cold = subparsers.add_parser("cold-start")
    cold.add_argument("--iterations", type=int, default=10)
    cold.set_defaults(function=command_cold_start)
    subparsers.add_parser("gateway-scale").set_defaults(function=command_gateway_scale)
    subparsers.add_parser("kill-drill").set_defaults(function=command_kill_drill)
    subparsers.add_parser("canary").set_defaults(function=command_canary)
    subparsers.add_parser("latency-alert").set_defaults(function=command_latency_alert)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--session-start", required=True)
    finalize.add_argument("--ci-conclusion", choices=("success", "failure"), required=True)
    finalize.add_argument("--ci-run-url", required=True)
    finalize.add_argument("--ci-git-sha", required=True)
    finalize.set_defaults(function=command_finalize)
    return value


def main() -> None:
    if os.environ.get("FORGE_GPU_HOURLY_USD") != "1.53":
        raise RuntimeError("FORGE_GPU_HOURLY_USD=1.53 is required")
    args = parser().parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    args.function(args)


if __name__ == "__main__":
    main()
