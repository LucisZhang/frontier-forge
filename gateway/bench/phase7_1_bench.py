#!/usr/bin/env python3
"""Run the staged Phase 7.1 A10 baseline, gateway matrix, and overload rerun."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import httpx
import yaml

from forge.bench.config import load_phase4_config
from forge.bench.preflight import benchmark_git_sha, host_disclosure
from forge.train.artifacts import (
    append_jsonl_once,
    sha256_tree,
    write_json_atomic,
    write_jsonl_atomic,
)
from forge.train.config import REPO_ROOT, relative_path, sha256_file
from gateway.bench import phase5_bench as core

RESULTS_DIR = REPO_ROOT / "results/phase7_1"
RAW_DIR = RESULTS_DIR / "raw"
REQUESTS_DIR = RESULTS_DIR / "requests"
FINAL_RECEIPT = RAW_DIR / "phase7_1_gateway_bench.json"
ARTIFACT_RECEIPT = RESULTS_DIR / "artifact_verification.json"
RUNS_PATH = REPO_ROOT / "results/runs.jsonl"
LEDGER_PATH = RESULTS_DIR / "gpu_ledger.jsonl"
CONFIG_PATH = REPO_ROOT / "configs/phase7_1/gateway_r1b_mtp_a10.yaml"
PHASE5_CONFIG_PATH = REPO_ROOT / "configs/phase5/gateway_r1b_mtp.yaml"
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_CONTRACT_KEYS = (
    "model",
    "server",
    "gateway",
    "workload",
    "direct_gateway",
    "capacity",
    "overload",
    "profile",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _canonical_hash(value: Any) -> str:
    return core._canonical_hash(value)


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = _load_yaml(config_path)
    if str(config.get("phase")) != "7.1":
        raise ValueError('Phase 7.1 benchmark config must pin phase: "7.1"')
    if config.get("benchmark_contract") != "phase5_exact_matrix_a10_rerun":
        raise ValueError("Phase 7.1 must pin the exact Phase 5 rerun contract")

    phase5_path = REPO_ROOT / str(config.get("source_phase5_config", ""))
    if phase5_path.resolve() != PHASE5_CONFIG_PATH.resolve():
        raise ValueError("source_phase5_config must name the archived Phase 5 config")
    phase5 = _load_yaml(phase5_path)
    for key in _CONTRACT_KEYS:
        if config.get(key) != phase5.get(key):
            raise ValueError(f"Phase 7.1 changed the archived Phase 5 {key} contract")

    declared_rate = Decimal(str(config["hardware"]["hourly_usd"]))
    supplied_rate = os.environ.get("FORGE_GPU_HOURLY_USD")
    if declared_rate != Decimal("1.53"):
        raise ValueError("Phase 7.1 A10 config must pin hourly_usd to 1.53")
    if supplied_rate is None or Decimal(supplied_rate) != declared_rate:
        raise RuntimeError("FORGE_GPU_HOURLY_USD=1.53 is required")
    if config["hardware"].get("gpu_type") != "NVIDIA A10":
        raise ValueError("Phase 7.1 hardware must be NVIDIA A10")

    pricing = config["pricing"]
    converted = Decimal(str(pricing["assumed_cny_per_hour"])) / Decimal(
        str(pricing["assumed_cny_per_usd"])
    )
    rounded = converted.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if rounded != declared_rate or Decimal(str(pricing["rounded_usd_per_hour"])) != declared_rate:
        raise ValueError("pricing assumption must round CNY 11/h at 7.2 CNY/USD to USD 1.53/h")

    config["_config_path"] = relative_path(config_path)
    config["_config_hash"] = _canonical_hash(
        {key: value for key, value in config.items() if not key.startswith("_")}
    )
    config["_phase5_config_path"] = relative_path(phase5_path)
    config["_phase5_config_sha256"] = sha256_file(phase5_path)
    config["_phase5_contract_hash"] = _canonical_hash({key: phase5[key] for key in _CONTRACT_KEYS})
    return config


def _verify_artifact(config: Mapping[str, Any]) -> dict[str, Any]:
    source = load_phase4_config(config["workload"]["source_config"])
    for key in (
        "variant",
        "precision",
        "served_name",
        "artifact_path",
        "artifact_sha256",
        "export_manifest",
        "training_time_quantization",
        "deployment_quantization",
    ):
        if source["model"].get(key) != config["model"].get(key):
            raise RuntimeError(f"Phase 4 source and Phase 7.1 model differ at {key}")
    artifact_path = REPO_ROOT / str(config["model"]["artifact_path"])
    actual = sha256_tree(artifact_path)
    expected = str(config["model"]["artifact_sha256"])
    if actual != expected:
        raise RuntimeError(f"A10 artifact tree hash mismatch: {actual} != {expected}")
    manifest_path = REPO_ROOT / str(config["model"]["export_manifest"])
    manifest = json.loads(manifest_path.read_text())
    export = manifest.get("full_precision_export", {})
    if export.get("path") != relative_path(artifact_path) or export.get("sha256") != expected:
        raise RuntimeError("Phase 4 export manifest does not identify the A10 serving artifact")
    receipt = {
        "version": 1,
        "status": "complete",
        "phase": "7.1",
        "git_sha": benchmark_git_sha(),
        "verified_at": _now(),
        "artifact": {
            "path": relative_path(artifact_path),
            "sha256": actual,
            "files": sum(1 for item in artifact_path.rglob("*") if item.is_file()),
            "export_manifest": relative_path(manifest_path),
            "export_manifest_sha256": sha256_file(manifest_path),
        },
    }
    write_json_atomic(ARTIFACT_RECEIPT, receipt)
    return receipt


def _require_artifact(config: Mapping[str, Any]) -> dict[str, Any]:
    if not ARTIFACT_RECEIPT.is_file():
        raise FileNotFoundError("run the Phase 7.1 artifact verification before benchmarking")
    receipt = json.loads(ARTIFACT_RECEIPT.read_text())
    artifact = receipt.get("artifact", {})
    if (
        receipt.get("status") != "complete"
        or receipt.get("git_sha") != benchmark_git_sha()
        or artifact.get("path") != config["model"]["artifact_path"]
        or artifact.get("sha256") != config["model"]["artifact_sha256"]
    ):
        raise RuntimeError("Phase 7.1 artifact verification receipt conflicts with the config")
    return receipt


def _stage_paths(stage: str) -> tuple[Path, Path]:
    return RAW_DIR / f"{stage}.json", REQUESTS_DIR / f"{stage}.requests.jsonl"


async def _run_clean_stage(
    stage: str,
    config: Mapping[str, Any],
    run_once: Callable[[], Awaitable[tuple[dict[str, Any], list[core.Observation]]]],
) -> dict[str, Any]:
    raw_path, request_path = _stage_paths(stage)
    if raw_path.is_file():
        existing = json.loads(raw_path.read_text())
        if existing.get("config_hash") != config["_config_hash"]:
            raise RuntimeError(f"existing {stage} artifact conflicts with the config")
        if existing.get("git_sha") != benchmark_git_sha():
            raise RuntimeError(f"existing {stage} artifact was produced by another commit")
        request = existing.get("request_artifact", {})
        if not request_path.is_file() or request.get("sha256") != sha256_file(request_path):
            raise RuntimeError(f"existing {stage} request artifact hash verification failed")
        print(f"validated existing Phase 7.1 stage: {relative_path(raw_path)}")
        return existing

    attempt = 0
    contaminated_attempts: list[dict[str, Any]] = []
    while True:
        attempt += 1
        await core._wait_host_clean()
        attempt_started_at = _now()
        payload, observations = await run_once()
        if not any(core._cell_contaminated(cell) for cell in payload["cells"]):
            break
        attempt_raw = RAW_DIR / f"{stage}.contaminated-{attempt:02d}.json"
        attempt_requests = REQUESTS_DIR / f"{stage}.contaminated-{attempt:02d}.requests.jsonl"
        write_jsonl_atomic(attempt_requests, [asdict(item) for item in observations])
        attempt_receipt = {
            **payload,
            "version": 1,
            "status": "contaminated-rerun-required",
            "phase": "7.1",
            "stage": stage,
            "config_path": config["_config_path"],
            "config_hash": config["_config_hash"],
            "git_sha": benchmark_git_sha(),
            "reason": "sampled_load1_exceeded_half_logical_core_count",
            "started_at": attempt_started_at,
            "request_artifact": {
                "path": relative_path(attempt_requests),
                "sha256": sha256_file(attempt_requests),
                "rows": len(observations),
            },
        }
        write_json_atomic(attempt_raw, attempt_receipt)
        contaminated_attempts.append(
            {
                "attempt": attempt,
                "raw_artifact": relative_path(attempt_raw),
                "request_artifact": relative_path(attempt_requests),
            }
        )
        print(f"Phase 7.1 {stage} attempt {attempt} contaminated; retained and rerunning")

    write_jsonl_atomic(request_path, [asdict(item) for item in observations])
    receipt = {
        **payload,
        "version": 1,
        "status": "complete",
        "phase": "7.1",
        "stage": stage,
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "git_sha": benchmark_git_sha(),
        "contaminated_attempts": contaminated_attempts,
        "request_artifact": {
            "path": relative_path(request_path),
            "sha256": sha256_file(request_path),
            "rows": len(observations),
        },
        "started_at": attempt_started_at,
        "finished_at": _now(),
        "raw_artifact": relative_path(raw_path),
    }
    write_json_atomic(raw_path, receipt)
    print(f"Phase 7.1 {stage} complete: {relative_path(raw_path)}")
    return receipt


async def _matrix_endpoint_stage(
    config: Mapping[str, Any],
    workload: Sequence[dict[str, Any]],
    *,
    endpoint: str,
    direct_url: str,
    gateway_url: str,
) -> tuple[dict[str, Any], list[core.Observation]]:
    settings = config["direct_gateway"]
    model = str(config["model"]["served_name"])
    cells: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    observations: list[core.Observation] = []
    index = 0
    for profile in config["workload"]["length_profiles"]:
        for concurrency_value in settings["concurrency"]:
            concurrency = int(concurrency_value)
            seed = int(config["workload"]["request_seed"]) + 1000 + index
            rows = core._select_rows(
                workload,
                config,
                profile=profile,
                count=int(config["workload"]["measurement_requests"]),
                seed=seed,
            )
            warmup = core._select_rows(
                workload,
                config,
                profile=profile,
                count=int(config["workload"]["warmup_requests"]),
                seed=seed - 1,
            )
            cell, items = await core._run_cell(
                endpoint=endpoint,
                base_url=direct_url if endpoint == "direct" else gateway_url,
                direct_url=direct_url,
                gateway_url=gateway_url,
                model=model,
                rows=rows,
                warmup_rows=warmup,
                cell_id=f"phase7-1-matrix-{profile}-c{concurrency}-{endpoint}",
                offsets=[0.0] * len(rows),
                concurrency=concurrency,
                deadline_s=float(settings["deadline_s"]),
                offered_qps=None,
                hourly_usd=float(config["hardware"]["hourly_usd"]),
            )
            cells.append(cell)
            observations.extend(items)
            entries.append(
                {
                    "length_profile": profile,
                    "concurrency": concurrency,
                    "endpoint": endpoint,
                    "cell": cell,
                }
            )
            index += 1
    return {
        "started_at": _now(),
        "endpoint": endpoint,
        "cells": cells,
        "entries": entries,
        "execution_contract": "all A10 bare-vLLM cells finish before any gateway cell",
    }, observations


async def _overload_endpoint_stage(
    config: Mapping[str, Any],
    workload: Sequence[dict[str, Any]],
    capacity: Mapping[str, Any],
    *,
    endpoint: str,
    direct_url: str,
    gateway_url: str,
) -> tuple[dict[str, Any], list[core.Observation]]:
    settings = config["overload"]
    capacity_qps = float(capacity["measured_capacity_qps"])
    recovery_threshold = float(capacity["baseline_success_p95_s"]) * float(
        settings["recovery_p95_multiplier"]
    )
    model = str(config["model"]["served_name"])
    cells: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    observations: list[core.Observation] = []
    for index, multiplier_value in enumerate(settings["multipliers"]):
        multiplier = float(multiplier_value)
        offered_qps = capacity_qps * multiplier
        seed = int(config["workload"]["request_seed"]) + 2000 + index
        rows = core._select_rows(
            workload,
            config,
            profile=str(settings["length_profile"]),
            count=int(settings["measurement_requests"]),
            seed=seed,
        )
        warmup = core._select_rows(
            workload,
            config,
            profile=str(settings["length_profile"]),
            count=int(config["workload"]["warmup_requests"]),
            seed=seed - 1,
        )
        offsets = core._poisson_offsets(count=len(rows), qps=offered_qps, seed=seed)
        cell, items = await core._overload_cell(
            endpoint=endpoint,
            base_url=direct_url if endpoint == "direct" else gateway_url,
            direct_url=direct_url,
            gateway_url=gateway_url,
            model=model,
            rows=rows,
            warmup=warmup,
            cell_id=f"phase7-1-overload-{multiplier:g}x-{endpoint}",
            offsets=offsets,
            deadline_s=float(settings["deadline_s"]),
            offered_qps=offered_qps,
            recovery_threshold_s=recovery_threshold,
            recovery_interval_s=float(settings["recovery_probe_interval_s"]),
            recovery_required=int(settings["recovery_consecutive_successes"]),
            hourly_usd=float(config["hardware"]["hourly_usd"]),
        )
        cells.append(cell)
        observations.extend(items)
        entries.append(
            {
                "multiplier": multiplier,
                "offered_qps": offered_qps,
                "endpoint": endpoint,
                "cell": cell,
            }
        )
    return {
        "started_at": _now(),
        "endpoint": endpoint,
        "cells": cells,
        "entries": entries,
        "measured_capacity_qps": capacity_qps,
        "recovery_definition": (
            "seconds after last scheduled arrival until "
            f"{settings['recovery_consecutive_successes']} consecutive probes complete within "
            f"{settings['recovery_p95_multiplier']}x low-load p95"
        ),
    }, observations


def _require_stage(stage: str, config: Mapping[str, Any]) -> dict[str, Any]:
    raw_path, request_path = _stage_paths(stage)
    if not raw_path.is_file() or not request_path.is_file():
        raise FileNotFoundError(f"required Phase 7.1 stage is missing: {stage}")
    receipt = json.loads(raw_path.read_text())
    if (
        receipt.get("status") != "complete"
        or receipt.get("config_hash") != config["_config_hash"]
        or receipt.get("git_sha") != benchmark_git_sha()
        or receipt.get("request_artifact", {}).get("sha256") != sha256_file(request_path)
    ):
        raise RuntimeError(f"required Phase 7.1 stage failed provenance checks: {stage}")
    return receipt


def _pair_matrix(
    direct_stage: Mapping[str, Any], gateway_stage: Mapping[str, Any]
) -> dict[str, Any]:
    direct = {
        (entry["length_profile"], int(entry["concurrency"])): entry["cell"]
        for entry in direct_stage["entries"]
    }
    gateway = {
        (entry["length_profile"], int(entry["concurrency"])): entry["cell"]
        for entry in gateway_stage["entries"]
    }
    if direct.keys() != gateway.keys():
        raise RuntimeError("A10 direct/gateway matrix cell sets differ")
    pairs = []
    for profile, concurrency in direct:
        left = direct[(profile, concurrency)]
        right = gateway[(profile, concurrency)]
        if left["warmup_schedule_sha256"] != right["warmup_schedule_sha256"]:
            raise RuntimeError("A10 direct/gateway warm-up schedules differ")
        pairs.append(
            {
                "length_profile": profile,
                "concurrency": concurrency,
                "execution_order": ["direct_baseline", "gateway_matched"],
                "direct": left,
                "gateway": right,
                "overhead": core._paired_overhead(left, right),
            }
        )
    return {
        "cells": [cell for pair in pairs for cell in (pair["direct"], pair["gateway"])],
        "pairs": pairs,
        "direct_stage": direct_stage["raw_artifact"],
        "gateway_stage": gateway_stage["raw_artifact"],
    }


def _pair_overload(
    direct_stage: Mapping[str, Any], gateway_stage: Mapping[str, Any]
) -> dict[str, Any]:
    direct = {float(entry["multiplier"]): entry for entry in direct_stage["entries"]}
    gateway = {float(entry["multiplier"]): entry for entry in gateway_stage["entries"]}
    if direct.keys() != gateway.keys():
        raise RuntimeError("A10 direct/gateway overload cell sets differ")
    pairs = []
    for multiplier in direct:
        left = direct[multiplier]["cell"]
        right = gateway[multiplier]["cell"]
        if left["request_schedule_sha256"] != right["request_schedule_sha256"]:
            raise RuntimeError("A10 overload direct/gateway schedules differ")
        if left["warmup_schedule_sha256"] != right["warmup_schedule_sha256"]:
            raise RuntimeError("A10 overload direct/gateway warm-up schedules differ")
        pairs.append(
            {
                "multiplier": multiplier,
                "offered_qps": direct[multiplier]["offered_qps"],
                "execution_order": ["direct_baseline", "gateway_overload"],
                "direct": left,
                "gateway": right,
            }
        )
    return {
        "cells": [cell for pair in pairs for cell in (pair["direct"], pair["gateway"])],
        "pairs": pairs,
        "measured_capacity_qps": direct_stage["measured_capacity_qps"],
        "recovery_definition": direct_stage["recovery_definition"],
        "direct_stage": direct_stage["raw_artifact"],
        "gateway_stage": gateway_stage["raw_artifact"],
    }


def _http_5xx_rate(cell: Mapping[str, Any]) -> float:
    statuses = cell.get("http_status_counts", {})
    count = sum(int(value) for status, value in statuses.items() if 500 <= int(status) < 600)
    return count / int(cell["requests"])


def _status_count(cell: Mapping[str, Any], status: int) -> int:
    return int(cell.get("http_status_counts", {}).get(str(status), 0))


def _non_admission_error_rate(cell: Mapping[str, Any], *, gateway: bool) -> float:
    admission_rejects = _status_count(cell, 429) if gateway else 0
    return max(0, int(cell["errors"]) - admission_rejects) / int(cell["requests"])


def _load_request_rows(stage: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = REPO_ROOT / str(stage["request_artifact"]["path"])
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _evaluate_gate(
    config: Mapping[str, Any],
    matrix: Mapping[str, Any],
    overload: Mapping[str, Any],
    gateway_overload_stage: Mapping[str, Any],
) -> dict[str, Any]:
    max_delta = float(config["gate"]["max_upstream_5xx_rate_delta"])
    max_reject_s = float(config["gate"]["max_fast_reject_s"])
    min_rejects = int(config["gate"]["min_429_rejects_per_overload_cell"])
    max_queue = int(config["gateway"]["max_queue_requests"])
    request_rows = _load_request_rows(gateway_overload_stage)

    parity_rows = []
    for family, pairs in (("matrix", matrix["pairs"]), ("overload", overload["pairs"])):
        for pair in pairs:
            direct_rate = _http_5xx_rate(pair["direct"])
            gateway_rate = _http_5xx_rate(pair["gateway"])
            delta = abs(gateway_rate - direct_rate)
            direct_non_admission = _non_admission_error_rate(pair["direct"], gateway=False)
            gateway_non_admission = _non_admission_error_rate(pair["gateway"], gateway=True)
            non_admission_delta = abs(gateway_non_admission - direct_non_admission)
            parity_rows.append(
                {
                    "family": family,
                    "cell": (
                        f"{pair['length_profile']}-c{pair['concurrency']}"
                        if family == "matrix"
                        else f"{pair['multiplier']:g}x"
                    ),
                    "bare_vllm_5xx_rate": direct_rate,
                    "gateway_upstream_5xx_rate": gateway_rate,
                    "absolute_delta": delta,
                    "bare_vllm_non_admission_error_rate": direct_non_admission,
                    "gateway_non_admission_error_rate": gateway_non_admission,
                    "non_admission_absolute_delta": non_admission_delta,
                    "threshold": max_delta,
                    "5xx_pass": delta <= max_delta + 1e-12,
                    "non_admission_error_pass": non_admission_delta <= max_delta + 1e-12,
                    "pass": (
                        delta <= max_delta + 1e-12 and non_admission_delta <= max_delta + 1e-12
                    ),
                }
            )

    overload_rows = []
    for pair in overload["pairs"]:
        cell = pair["gateway"]
        measured = [row for row in request_rows if row.get("cell_id") == cell["cell_id"]]
        rejects = [row for row in measured if row.get("http_status") == 429]
        semantics = all(
            (not config["gate"]["require_retry_after"] or row.get("retry_after"))
            and (
                not config["gate"]["require_overloaded_error_code"]
                or row.get("error_code") == "overloaded"
            )
            and float(row["client_e2e_s"]) <= max_reject_s
            for row in rejects
        )
        sampled_queue = cell["gateway_samples"].get("queue_depth_max")
        process_queue = (cell.get("gateway") or {}).get("queue_high_watermark_process")
        bounded = (
            sampled_queue is not None
            and process_queue is not None
            and float(sampled_queue) <= max_queue
            and float(process_queue) <= max_queue
        )
        decision_count = int(
            (cell.get("gateway") or {}).get("routing_decisions", {}).get("reject_overload", 0)
        )
        overload_rows.append(
            {
                "multiplier": pair["multiplier"],
                "http_429_count": len(rejects),
                "reject_overload_decisions": decision_count,
                "all_429_fast_with_retry_after_and_overloaded_code": semantics,
                "sampled_queue_max": sampled_queue,
                "process_queue_high_watermark": process_queue,
                "configured_queue_bound": max_queue,
                "bounded_queue": bounded,
                "pass": (
                    len(rejects) >= min_rejects
                    and len(rejects) == _status_count(cell, 429)
                    and decision_count >= len(rejects)
                    and semantics
                    and bounded
                ),
            }
        )

    checks = {
        "baseline_completed_before_gateway": True,
        "matched_matrix_receipts": len(matrix["pairs"])
        == len(config["workload"]["length_profiles"])
        * len(config["direct_gateway"]["concurrency"]),
        "overload_receipts": len(overload["pairs"]) == len(config["overload"]["multipliers"]),
        "429_fast_reject_semantics": all(row["pass"] for row in overload_rows),
        "bounded_queue": all(row["bounded_queue"] for row in overload_rows),
        "upstream_5xx_parity": all(row["5xx_pass"] for row in parity_rows),
        "non_admission_error_parity": all(row["non_admission_error_pass"] for row in parity_rows),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "overload_cells": overload_rows,
        "upstream_5xx_parity_cells": parity_rows,
        "thresholds": {
            "max_upstream_5xx_rate_delta": max_delta,
            "max_fast_reject_s": max_reject_s,
            "min_429_rejects_per_overload_cell": min_rejects,
        },
    }


async def _identity(config: Mapping[str, Any], direct_url: str, gateway_url: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        version, models, health = await asyncio.gather(
            client.get(f"{direct_url.rstrip('/')}/version"),
            client.get(f"{direct_url.rstrip('/')}/v1/models"),
            client.get(f"{gateway_url.rstrip('/')}/healthz"),
        )
    version.raise_for_status()
    models.raise_for_status()
    health.raise_for_status()
    version_payload = version.json()
    if str(version_payload.get("version")) != str(config["server"]["vllm_version"]):
        raise RuntimeError("live vLLM version differs from the exact Phase 5 software contract")
    served = {item.get("id") for item in models.json().get("data", [])}
    if config["model"]["served_name"] not in served:
        raise RuntimeError("live vLLM model identity differs from the Phase 7.1 config")
    return {
        "vllm_version_endpoint": version_payload,
        "vllm_models_endpoint": models.json(),
        "gateway_health_status": health.status_code,
    }


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _append_run_record(receipt: Mapping[str, Any]) -> bool:
    record = {
        "phase": "7.1",
        "run_id": receipt["run_id"],
        "status": "complete",
        "config_path": receipt["config_path"],
        "config_hash": receipt["config_hash"],
        "dataset_hash": receipt["workload"]["sha256"],
        "git_sha": receipt["git_sha"],
        "model": receipt["model"],
        "metrics": receipt["metrics"],
        "gate": receipt["gate"],
        "production_blocked": receipt["production_blocked"],
        "cost": receipt["cost"],
        "started_at": receipt["started_at"],
        "finished_at": receipt["finished_at"],
        "raw_artifact": receipt["raw_artifact"],
        "disclosure": receipt["disclosure"],
    }
    return append_jsonl_once(RUNS_PATH, record, key="run_id")


def _write_final_receipt(
    config: Mapping[str, Any],
    workload_receipt: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    artifact: Mapping[str, Any],
    capacity: Mapping[str, Any],
    matrix_direct: Mapping[str, Any],
    matrix_gateway: Mapping[str, Any],
    overload_direct: Mapping[str, Any],
    overload_gateway: Mapping[str, Any],
) -> dict[str, Any]:
    matrix = _pair_matrix(matrix_direct, matrix_gateway)
    overload = _pair_overload(overload_direct, overload_gateway)
    gate = _evaluate_gate(config, matrix, overload, overload_gateway)

    baseline_finished = max(
        _parse_time(stage["finished_at"]) for stage in (capacity, matrix_direct, overload_direct)
    )
    gateway_started = min(
        _parse_time(stage["started_at"]) for stage in (matrix_gateway, overload_gateway)
    )
    if baseline_finished > gateway_started:
        raise RuntimeError(
            "gateway measurement began before the complete A10 bare baseline finished"
        )

    hardware = host_disclosure()
    if "A10" not in str(hardware.get("gpu")):
        raise RuntimeError(f"Phase 7.1 refuses non-A10 hardware: {hardware.get('gpu')!r}")
    started_value = os.environ.get("FORGE_VM_STARTED_AT")
    if not started_value:
        raise RuntimeError("FORGE_VM_STARTED_AT is required for the delegated VM-session ledger")
    started_at = _parse_time(started_value)
    finished_at = datetime.now(UTC)
    session_hours = (finished_at - started_at).total_seconds() / 3600
    hourly_usd = float(config["hardware"]["hourly_usd"])
    receipt = {
        "version": 1,
        "status": "complete",
        "phase": "7.1",
        "run_id": config["run_id"],
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "git_sha": benchmark_git_sha(),
        "model": config["model"],
        "artifact_verification": {
            **artifact["artifact"],
            "receipt": relative_path(ARTIFACT_RECEIPT),
            "receipt_sha256": sha256_file(ARTIFACT_RECEIPT),
        },
        "server": {"identity": identity, "declared": config["server"]},
        "gateway": config["gateway"],
        "hardware": {**hardware, "declared_gpu_type": config["hardware"]["gpu_type"]},
        "workload": workload_receipt,
        "metrics": {
            "capacity": capacity,
            "direct_gateway": matrix,
            "overload": overload,
        },
        "gate": gate,
        "production_blocked": gate["status"] != "pass",
        "disclosure": {
            "hardware_change": (
                "This rerun uses one NVIDIA A10 on Aliyun ECS. Archived Phase 5 numbers use an "
                "RTX 4090 and remain historical only; no A10 latency or throughput value is "
                "compared against the 4090."
            ),
            "paired_scope": config["comparison_scope"],
            "execution_order": (
                "All A10 bare-vLLM capacity, matched-matrix, and overload baseline cells completed "
                "before the gateway process was started."
            ),
            "phase5_contract": {
                "path": config["_phase5_config_path"],
                "sha256": config["_phase5_config_sha256"],
                "contract_hash": config["_phase5_contract_hash"],
            },
            "original_negative_history": {
                "run_id": "phase5_gateway_r1b_bf16_native_mtp",
                "gpu": "RTX4090",
                "receipt": "results/phase5/raw/phase5_gateway_bench.json",
                "finding": "admitted HTTP 502/upstream_error with reject_overload=0",
            },
            "precision": config["model"]["precision"],
            "training_time_quantization": config["model"]["training_time_quantization"],
            "deployment_quantization": config["model"]["deployment_quantization"],
            "fallback": "disabled; one physical R1b MTP vLLM replica",
        },
        "cost": {
            "vm_session_hours": session_hours,
            "hourly_usd": hourly_usd,
            "usd": session_hours * hourly_usd,
            "assumed_cny_per_hour": float(config["pricing"]["assumed_cny_per_hour"]),
            "assumed_cny_per_usd": float(config["pricing"]["assumed_cny_per_usd"]),
            "rate_source": "FORGE_GPU_HOURLY_USD=1.53",
            "scope": "delegated VM boot/session start through final receipt",
            "post_receipt_running_cost_excluded": True,
        },
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "raw_artifact": relative_path(FINAL_RECEIPT),
    }
    write_json_atomic(FINAL_RECEIPT, receipt)
    _append_run_record(receipt)
    ledger = {
        "ledger_id": config["run_id"],
        "phase": "7.1",
        "operation": "a10_bare_vllm_and_gateway_matched_rerun",
        "status": "complete",
        "gate_status": gate["status"],
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "git_sha": benchmark_git_sha(),
        "gpu_type": config["hardware"]["gpu_type"],
        "gpu_hours": session_hours,
        "hourly_usd": hourly_usd,
        "usd": session_hours * hourly_usd,
        "rate_source": "FORGE_GPU_HOURLY_USD=1.53; assumes CNY 11/h at 7.2 CNY/USD",
        "started_at": receipt["started_at"],
        "finished_at": receipt["finished_at"],
        "notes": (
            "Whole delegated VM session through receipt finalization; "
            "later owner stop delay excluded."
        ),
    }
    append_jsonl_once(LEDGER_PATH, ledger, key="ledger_id")
    return receipt


async def _run_remote_stage(
    stage: str,
    config: Mapping[str, Any],
    workload: Sequence[dict[str, Any]],
    workload_receipt: Mapping[str, Any],
    *,
    direct_url: str,
    gateway_url: str,
) -> dict[str, Any]:
    if stage == "baseline":
        capacity = await _run_clean_stage(
            "capacity-a10-bare",
            config,
            lambda: core._capacity_stage(
                config, workload, direct_url=direct_url, gateway_url=gateway_url
            ),
        )
        matrix = await _run_clean_stage(
            "matrix-a10-bare",
            config,
            lambda: _matrix_endpoint_stage(
                config,
                workload,
                endpoint="direct",
                direct_url=direct_url,
                gateway_url=gateway_url,
            ),
        )
        overload = await _run_clean_stage(
            "overload-a10-bare",
            config,
            lambda: _overload_endpoint_stage(
                config,
                workload,
                capacity,
                endpoint="direct",
                direct_url=direct_url,
                gateway_url=gateway_url,
            ),
        )
        return {"status": "complete", "capacity": capacity, "matrix": matrix, "overload": overload}
    if stage == "gateway-matrix":
        _require_stage("capacity-a10-bare", config)
        _require_stage("matrix-a10-bare", config)
        _require_stage("overload-a10-bare", config)
        return await _run_clean_stage(
            "matrix-a10-gateway",
            config,
            lambda: _matrix_endpoint_stage(
                config,
                workload,
                endpoint="gateway",
                direct_url=direct_url,
                gateway_url=gateway_url,
            ),
        )
    if stage == "gateway-overload":
        capacity = _require_stage("capacity-a10-bare", config)
        matrix_direct = _require_stage("matrix-a10-bare", config)
        overload_direct = _require_stage("overload-a10-bare", config)
        matrix_gateway = _require_stage("matrix-a10-gateway", config)
        overload_gateway = await _run_clean_stage(
            "overload-a10-gateway",
            config,
            lambda: _overload_endpoint_stage(
                config,
                workload,
                capacity,
                endpoint="gateway",
                direct_url=direct_url,
                gateway_url=gateway_url,
            ),
        )
        identity = await _identity(config, direct_url, gateway_url)
        return _write_final_receipt(
            config,
            workload_receipt,
            identity=identity,
            artifact=_require_artifact(config),
            capacity=capacity,
            matrix_direct=matrix_direct,
            matrix_gateway=matrix_gateway,
            overload_direct=overload_direct,
            overload_gateway=overload_gateway,
        )
    raise ValueError(f"unsupported Phase 7.1 stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=relative_path(CONFIG_PATH))
    parser.add_argument("--direct-url", default="http://127.0.0.1:8000")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:9000")
    parser.add_argument(
        "--stage",
        choices=(
            "verify-artifact",
            "baseline",
            "gateway-matrix",
            "gateway-overload",
            "import-final",
        ),
        required=True,
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    sha = benchmark_git_sha()
    if _GIT_SHA.fullmatch(sha) is None:
        raise RuntimeError("Phase 7.1 requires a full lowercase Git SHA")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REQUESTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "verify-artifact":
        result = _verify_artifact(config)
    elif args.stage == "import-final":
        if not FINAL_RECEIPT.is_file():
            raise FileNotFoundError(FINAL_RECEIPT)
        result = json.loads(FINAL_RECEIPT.read_text())
        if result.get("config_hash") != config["_config_hash"]:
            raise RuntimeError("synced Phase 7.1 final receipt conflicts with the local config")
        _append_run_record(result)
    else:
        _require_artifact(config)
        workload, workload_receipt = core._phase4_workload(config)
        result = asyncio.run(
            _run_remote_stage(
                args.stage,
                config,
                workload,
                workload_receipt,
                direct_url=args.direct_url,
                gateway_url=args.gateway_url,
            )
        )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "status": result["status"],
                "gate": result.get("gate", {}).get("status"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
