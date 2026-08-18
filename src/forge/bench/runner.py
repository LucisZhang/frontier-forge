"""Execute one immutable Phase 4 experiment against an already-running server."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from forge.train.artifacts import append_jsonl_once, write_json_atomic, write_jsonl_atomic
from forge.train.config import REPO_ROOT, relative_path, sha256_file

from .config import (
    load_phase4_config,
    phase4_raw_path,
    phase4_requests_path,
    phase4_workload_path,
    workload_contract,
    workload_contract_hash,
)
from .loadgen import VERIFIER_INPUT_NORMALIZATION, run_load_benchmark
from .mtp_audit import require_speculative_method_evidence
from .preflight import (
    VERIFICATION_PATH,
    benchmark_git_sha,
    host_disclosure,
    require_verified_artifact,
)
from .structured import run_structured_benchmark
from .system_load import metrics_contaminated
from .workload import build_workload


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


async def _server_identity(base_url: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        version_response = await client.get(f"{base_url.rstrip('/')}/version")
        models_response = await client.get(f"{base_url.rstrip('/')}/v1/models")
        version_response.raise_for_status()
        models_response.raise_for_status()
        return {
            "version_endpoint": version_response.json(),
            "models_endpoint": models_response.json(),
        }


def _max_stable_concurrency(points: list[dict[str, Any]]) -> int | None:
    values = [point["max_observed_in_flight"] for point in points if point.get("stable")]
    return max(values) if values else None


def _full_run_record(receipt: dict[str, Any]) -> dict[str, Any]:
    metrics = receipt["metrics"]
    summary: dict[str, Any]
    if receipt["experiment"] in {"serve", "spec_decode"}:
        points = metrics["points"]
        summary = {
            "max_stable_concurrency": _max_stable_concurrency(points),
            "points": points,
        }
    else:
        summary = metrics
    return {
        "phase": 4,
        "run_id": receipt["run_id"],
        "status": "complete",
        "experiment": receipt["experiment"],
        "config_path": receipt["config_path"],
        "config_hash": receipt["config_hash"],
        "git_sha": receipt["git_sha"],
        "dataset_hash": receipt["workload"]["sha256"],
        "model": receipt["model"],
        "speculative": receipt.get("speculative"),
        "speculative_method_evidence": receipt.get("speculative_method_evidence"),
        "metrics": summary,
        "cost": receipt["cost"],
        "started_at": receipt["started_at"],
        "finished_at": receipt["finished_at"],
        "raw_artifact": receipt["raw_artifact"],
        "verifier_disclosure": receipt["verifier_disclosure"],
        "supersedes": receipt.get("supersedes"),
        "contaminated_sweeps": receipt.get("contaminated_sweeps", []),
    }


def _require_hourly_rate(config: dict[str, Any], *, smoke: bool) -> float:
    expected = float(config["hardware"]["hourly_usd"])
    if smoke:
        return expected
    supplied = os.environ.get("FORGE_GPU_HOURLY_USD")
    if supplied is None:
        raise RuntimeError("FORGE_GPU_HOURLY_USD is required for full benchmarks")
    actual = float(supplied)
    if actual != expected:
        raise RuntimeError(f"hourly rate mismatch: config={expected}, environment={actual}")
    return actual


def validate_existing_receipt(config_path: str | Path) -> dict[str, Any]:
    """Validate a completed full-run receipt without rewriting its provenance."""

    config = load_phase4_config(config_path)
    raw_path = phase4_raw_path(config, smoke=False)
    if not raw_path.is_file():
        raise RuntimeError(f"Phase 4 raw artifact is missing: {raw_path}")
    receipt = json.loads(raw_path.read_text())
    expected_fields = {
        "status": "complete",
        "mode": "full",
        "phase": 4,
        "experiment": config["experiment"],
        "run_id": config["run_id"],
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "model": config["model"],
        "raw_artifact": relative_path(raw_path),
    }
    for field, expected in expected_fields.items():
        if receipt.get(field) != expected:
            raise RuntimeError(
                f"existing Phase 4 receipt has conflicting {field}: "
                f"expected {expected!r}, got {receipt.get(field)!r}"
            )
    git_sha = receipt.get("git_sha")
    if not isinstance(git_sha, str) or len(git_sha) != 40:
        raise RuntimeError("existing Phase 4 receipt must preserve a full git SHA")

    request_artifact = receipt.get("request_artifact")
    if not isinstance(request_artifact, Mapping):
        raise RuntimeError("existing Phase 4 receipt is missing request artifact metadata")
    requests_path = phase4_requests_path(config, smoke=False)
    if request_artifact.get("path") != relative_path(requests_path):
        raise RuntimeError("existing Phase 4 request artifact path conflicts with the config")
    if not requests_path.is_file() or request_artifact.get("sha256") != sha256_file(requests_path):
        raise RuntimeError("existing Phase 4 request artifact hash verification failed")

    workload = receipt.get("workload")
    if not isinstance(workload, Mapping):
        raise RuntimeError("existing Phase 4 receipt is missing workload metadata")
    workload_path = phase4_workload_path(config, smoke=False)
    if workload.get("path") != relative_path(workload_path):
        raise RuntimeError("existing Phase 4 workload path conflicts with the config")
    if workload.get("contract_hash") != workload_contract_hash(config):
        raise RuntimeError("existing Phase 4 workload contract hash conflicts with the config")
    if not workload_path.is_file() or workload.get("sha256") != sha256_file(workload_path):
        raise RuntimeError("existing Phase 4 workload hash verification failed")

    expected_rate = float(config["hardware"]["hourly_usd"])
    if receipt.get("cost", {}).get("hourly_usd") != expected_rate:
        raise RuntimeError("existing Phase 4 receipt hourly rate conflicts with the config")
    print(
        "Validated existing Phase 4 receipt without rewriting provenance: "
        f"{relative_path(raw_path)} ({git_sha})"
    )
    return receipt


def run(config_path: str | Path, *, base_url: str, smoke: bool) -> dict[str, Any]:
    config = load_phase4_config(config_path)
    raw_path = phase4_raw_path(config, smoke=smoke)
    if raw_path.is_file() and not smoke:
        existing = json.loads(raw_path.read_text())
        if (
            existing.get("status") == "complete"
            and existing.get("config_hash") == config["_config_hash"]
            and (smoke or existing.get("git_sha") == benchmark_git_sha())
        ):
            print(f"Phase 4 benchmark already complete: {relative_path(raw_path)}")
            return existing
        raise RuntimeError(f"existing Phase 4 raw artifact conflicts: {raw_path}")
    hourly_usd = _require_hourly_rate(config, smoke=smoke)
    workload_receipt = build_workload(config_path, smoke=smoke)
    artifact_verification = None if smoke else require_verified_artifact(config)
    speculative = config.get("speculative")
    speculative_method_evidence = None
    if not smoke and isinstance(speculative, Mapping) and speculative.get("enabled"):
        speculative_method_evidence = require_speculative_method_evidence(config)
    started_at = datetime.now(UTC)
    start_clock = time.perf_counter()
    identity = asyncio.run(_server_identity(base_url))
    contaminated_sweeps: list[dict[str, Any]] = []
    attempt = 0
    while True:
        attempt += 1
        if config["experiment"] in {"serve", "spec_decode"}:
            points, request_records = asyncio.run(
                run_load_benchmark(config, base_url=base_url, smoke=smoke)
            )
            metrics: dict[str, Any] = {
                "points": points,
                "max_stable_concurrency": _max_stable_concurrency(points),
            }
        else:
            structured, request_records = asyncio.run(
                run_structured_benchmark(config, base_url=base_url, smoke=smoke)
            )
            metrics = structured
        if smoke or not metrics_contaminated(metrics, experiment=str(config["experiment"])):
            break
        contaminated_path = raw_path.with_name(
            f"{raw_path.stem}.contaminated-{attempt:02d}.requests.jsonl"
        )
        write_jsonl_atomic(contaminated_path, request_records)
        contaminated_sweeps.append(
            {
                "attempt": attempt,
                "reason": "sampled_load1_exceeded_half_logical_core_count",
                "metrics": metrics,
                "request_artifact": {
                    "path": relative_path(contaminated_path),
                    "sha256": sha256_file(contaminated_path),
                    "rows": len(request_records),
                },
            }
        )
        cores = os.cpu_count() or 1
        threshold = cores / 2
        retained_path = relative_path(contaminated_path)
        print(f"Phase 4 sweep contaminated on attempt {attempt}; retained {retained_path}")
        while os.getloadavg()[0] > threshold:
            print(
                f"host load1={os.getloadavg()[0]:.2f} exceeds threshold={threshold:.2f}; "
                "waiting 30s before clean rerun"
            )
            time.sleep(30)
    elapsed = time.perf_counter() - start_clock
    finished_at = datetime.now(UTC)
    requests_path = phase4_requests_path(config, smoke=smoke)
    write_jsonl_atomic(requests_path, request_records)
    receipt = {
        "version": 1,
        "status": "complete",
        "phase": 4,
        "mode": "smoke" if smoke else "full",
        "experiment": config["experiment"],
        "run_id": config["run_id"],
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "git_sha": benchmark_git_sha(),
        "model": config["model"],
        "speculative": speculative,
        "speculative_method_evidence": speculative_method_evidence,
        "server": {
            "base_url": base_url,
            "identity": identity,
            "declared_vllm_version": config["server"]["vllm_version"],
            "engine_args": config["server"],
            "packages": {
                name: _package_version(name)
                for name in ("vllm", "torch", "transformers", "xgrammar", "outlines")
            },
        },
        "hardware": {
            **host_disclosure(),
            "hourly_usd": hourly_usd,
            "rate_source": "FORGE_GPU_HOURLY_USD" if not smoke else "smoke_config_only",
        },
        "workload": {
            **workload_contract(config),
            "contract_hash": workload_contract_hash(config),
            "path": workload_receipt["path"],
            "sha256": workload_receipt["sha256"],
            "input_length_measurement": (
                "UTF-8 byte-length divided by four (SMOKE_ONLY)"
                if smoke
                else "tokenizer.apply_chat_template exact token ids"
            ),
            "output_length_control": "per-request max_tokens fixed weighted distribution",
            "arrival_process": "Poisson with fixed seed",
        },
        "timing_disclosure": {
            "client": "monotonic wall clock around streamed OpenAI HTTP request",
            "server": "delta of vLLM Prometheus histograms over each measurement point",
            "vram": "device-total nvidia-smi samples on the server host",
            "co_tenancy": (
                "server-host getloadavg plus /proc/stat CPU utilization; any load1 sample "
                "above half logical core count contaminates and reruns the entire sweep"
            ),
            "warmup_excluded": True,
        },
        "verifier_disclosure": {
            "implementation": "forge.verify.verifier.score",
            "input_normalization": VERIFIER_INPUT_NORMALIZATION,
            "raw_and_normalized_outputs_preserved": True,
        },
        "supersedes": config.get("supersedes"),
        "artifact_verification": artifact_verification,
        "artifact_verification_path": (None if smoke else relative_path(VERIFICATION_PATH)),
        "metrics": metrics,
        "contaminated_sweeps": contaminated_sweeps,
        "request_artifact": {
            "path": relative_path(requests_path),
            "sha256": sha256_file(requests_path),
            "rows": len(request_records),
        },
        "cost": {
            "gpu_hours": elapsed / 3600 if not smoke else 0.0,
            "hourly_usd": hourly_usd,
            "usd": elapsed / 3600 * hourly_usd if not smoke else 0.0,
            "api_usd": 0.0,
        },
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "raw_artifact": relative_path(raw_path),
    }
    write_json_atomic(raw_path, receipt)
    if not smoke:
        append_jsonl_once(
            REPO_ROOT / "results/runs.jsonl",
            _full_run_record(receipt),
            key="run_id",
        )
    print(f"Phase 4 {config['run_id']} complete: {relative_path(raw_path)}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    if args.validate_existing:
        if args.smoke:
            parser.error("--validate-existing is only for immutable full-run receipts")
        validate_existing_receipt(args.config)
    else:
        run(args.config, base_url=args.base_url, smoke=args.smoke)


if __name__ == "__main__":
    main()
