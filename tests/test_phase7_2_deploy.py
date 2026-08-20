from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
DEPLOY = ROOT / "deploy/phase7_2"


def _documents(path: Path) -> list[dict[str, object]]:
    return [item for item in yaml.safe_load_all(path.read_text()) if isinstance(item, dict)]


def _real_documents() -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    for path in sorted((DEPLOY / "real").rglob("*.yaml")):
        if path.name == "kustomization.yaml":
            continue
        documents.extend(_documents(path))
    for path in sorted((DEPLOY / "base").glob("*.yaml")):
        if path.name != "kustomization.yaml":
            documents.extend(_documents(path))
    return documents


def _named(documents: list[dict[str, object]], kind: str, name: str) -> dict[str, object]:
    for item in documents:
        metadata = item.get("metadata", {})
        if item.get("kind") == kind and isinstance(metadata, dict) and metadata.get("name") == name:
            return item
    raise AssertionError(f"missing {kind}/{name}")


def test_all_services_are_cluster_internal() -> None:
    services = [item for item in _real_documents() if item.get("kind") == "Service"]
    assert services
    assert all(item["spec"].get("type", "ClusterIP") == "ClusterIP" for item in services)
    assert all("nodePort" not in port for item in services for port in item["spec"]["ports"])


def test_no_public_host_bindings_in_pod_specs() -> None:
    for item in _real_documents():
        if item.get("kind") not in {"Deployment", "DaemonSet", "StatefulSet"}:
            continue
        pod_spec = item["spec"]["template"]["spec"]
        assert not pod_spec.get("hostNetwork", False)
        for container in pod_spec.get("containers", []):
            assert all("hostPort" not in port for port in container.get("ports", []))


def test_single_gpu_is_time_sliced_for_exactly_two_workloads() -> None:
    values = yaml.safe_load((DEPLOY / "charts/nvidia-device-plugin-values.yaml").read_text())
    config = values["config"]["map"]["phase7-2"]
    assert "renameByDefault: true" in config
    assert "replicas: 2" in config
    documents = _real_documents()
    for name in ("vllm-int4", "vllm-bf16"):
        deployment = _named(documents, "Deployment", name)
        assert deployment["spec"]["replicas"] == 0
        resources = deployment["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert resources["requests"]["nvidia.com/gpu.shared"] == "1"
        assert resources["limits"]["nvidia.com/gpu.shared"] == "1"


def test_keda_scopes_cpu_and_gpu_scaling_honestly() -> None:
    documents = _documents(DEPLOY / "real/keda-scalers.yaml")
    gateway = _named(documents, "ScaledObject", "forge-gateway")
    gpu = _named(documents, "ScaledObject", "vllm-int4")
    assert gateway["spec"]["minReplicaCount"] == 1
    assert gateway["spec"]["maxReplicaCount"] == 3
    assert "forge_gateway_queue_depth" in gateway["spec"]["triggers"][0]["metadata"]["query"]
    assert gpu["spec"]["minReplicaCount"] == 0
    assert gpu["spec"]["maxReplicaCount"] == 1
    assert "forge_gpu_scale_demand" in gpu["spec"]["triggers"][0]["metadata"]["query"]


def test_both_slo_alerts_use_two_windows() -> None:
    rules = _documents(DEPLOY / "real/monitoring/prometheus-rules.yaml")[0]
    alerts = rules["spec"]["groups"][0]["rules"]
    assert {item["alert"] for item in alerts} == {
        "ForgeAvailabilityBurnRate",
        "ForgeLatencyBurnRate",
    }
    for item in alerts:
        assert "[30s]" in item["expr"]
        assert "[2m]" in item["expr"]
        assert item["for"] == "0m"
    latency = next(item for item in alerts if item["alert"] == "ForgeLatencyBurnRate")
    assert 'le="2.500"' in latency["expr"]
    assert 'le="2.000"' not in latency["expr"]


def test_dashboard_and_router_variants_are_versioned() -> None:
    dashboard = json.loads((DEPLOY / "dashboards/frontier-forge-runtime.json").read_text())
    assert dashboard["uid"] == "frontier-forge-phase7-2"
    assert len(dashboard["panels"]) >= 5
    expected = {"stable", "canary-10", "canary-50", "canary-100", "fault-500", "fault-latency"}
    assert expected == {path.stem for path in (DEPLOY / "real/router").glob("*.conf")}
    for path in (DEPLOY / "real/router").glob("*.conf"):
        assert "X-Forge-Upstream" in path.read_text()


def test_k6_exports_status_and_retry_after_evidence() -> None:
    script = (DEPLOY / "k6/scenarios.js").read_text()
    for metric in (
        "forge_http_200",
        "forge_http_429",
        "forge_http_500",
        "forge_http_503",
        "forge_retry_after_429",
    ):
        assert metric in script
    assert "FORGE_K6_SUMMARY_JSON" in script


def test_kind_overlay_and_ci_use_the_mock_upstream() -> None:
    kustomization = yaml.safe_load((DEPLOY / "kind/kustomization.yaml").read_text())
    assert "mock-upstream-deployment.yaml" in kustomization["resources"]
    patch = (DEPLOY / "kind/gateway-patch.yaml").read_text()
    assert "forge-mock-upstream" in patch
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "phase-7-2-kind" in workflow
    assert "make phase7-2-kind-smoke" in workflow


def test_version_lock_matches_immutable_model_manifests() -> None:
    locked = yaml.safe_load((DEPLOY / "versions.lock.yaml").read_text())
    phase3 = json.loads((ROOT / "results/phase3_export_manifest_r1b_trl_s0.json").read_text())
    phase4 = json.loads((ROOT / "results/phase4/r1b_mtp_reexport_manifest.json").read_text())
    assert (
        locked["model_archive"]["gptq_int4_tree_sha256"]
        == phase3["deployment_int4_export"]["sha256"]
    )
    assert (
        locked["model_archive"]["bf16_mtp_preserved_tree_sha256"]
        == phase4["full_precision_export"]["sha256"]
    )
    assert locked["image_source_manifests_linux_amd64"]["vllm"].startswith("sha256:")
    assert locked["images"]["k6"] == "grafana/k6:2.2.0"
