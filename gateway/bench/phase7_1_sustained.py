#!/usr/bin/env python3
"""Run and seal the duration-based Gate 7.1 overload amendment on one A10 VM."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from forge.bench.config import load_phase4_config
from forge.bench.preflight import benchmark_git_sha, host_disclosure
from forge.train.artifacts import append_jsonl_once, sha256_tree, write_json_atomic
from forge.train.config import REPO_ROOT, relative_path, sha256_file
from gateway.bench import phase5_bench as core
from gateway.bench import phase7_1_bench as finite

RESULTS_DIR = REPO_ROOT / "results/phase7_1"
RAW_DIR = RESULTS_DIR / "raw"
REQUESTS_DIR = RESULTS_DIR / "requests"
CONFIG_PATH = REPO_ROOT / "configs/phase7_1/sustained_overload_a10.yaml"
ARTIFACT_RECEIPT = RESULTS_DIR / "artifact_verification-sustained.json"
FINAL_RECEIPT = RAW_DIR / "phase7_1_sustained_gateway_bench.json"
RUNS_PATH = REPO_ROOT / "results/runs.jsonl"
LEDGER_PATH = RESULTS_DIR / "gpu_ledger.jsonl"
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_config(path: str | Path) -> dict[str, Any]:
    amendment_path = Path(path)
    if not amendment_path.is_absolute():
        amendment_path = REPO_ROOT / amendment_path
    amendment = finite._load_yaml(amendment_path)
    if str(amendment.get("phase")) != "7.1":
        raise ValueError('sustained amendment must pin phase: "7.1"')
    if amendment.get("benchmark_contract") != "gate_7_1_sustained_amendment_2026_08_21":
        raise ValueError("unexpected Gate 7.1 sustained benchmark contract")

    source_path = REPO_ROOT / str(amendment.get("source_phase7_1_config", ""))
    if source_path.resolve() != finite.CONFIG_PATH.resolve():
        raise ValueError("source_phase7_1_config must name the archived A10 finite config")
    source = finite._load_config(source_path)

    sustained = amendment.get("sustained_overload", {})
    if [float(value) for value in sustained.get("multipliers", [])] != [
        float(value) for value in source["overload"]["multipliers"]
    ]:
        raise ValueError("sustained multipliers must remain the archived 2x/3x/5x cells")
    if float(sustained.get("minimum_arrival_duration_s", 0)) < 120:
        raise ValueError("Gate 7.1 sustained arrivals must last at least 120 seconds per cell")
    if int(sustained.get("max_client_concurrency", 0)) < 256:
        raise ValueError("sustained client concurrency is too small for the locked overload rates")

    for key in ("hardware", "pricing", "comparison_scope"):
        if amendment.get(key) != source.get(key):
            raise ValueError(f"sustained amendment changed the archived A10 {key} contract")
    for key in (
        "max_upstream_5xx_rate_delta",
        "max_fast_reject_s",
        "min_429_rejects_per_overload_cell",
        "require_retry_after",
        "require_overloaded_error_code",
    ):
        if amendment["gate"].get(key) != source["gate"].get(key):
            raise ValueError(f"sustained amendment changed the archived gate threshold {key}")

    declared_rate = Decimal(str(amendment["hardware"]["hourly_usd"]))
    supplied_rate = os.environ.get("FORGE_GPU_HOURLY_USD")
    if declared_rate != Decimal("1.53") or supplied_rate is None:
        raise RuntimeError("FORGE_GPU_HOURLY_USD=1.53 is required")
    if Decimal(supplied_rate) != declared_rate:
        raise RuntimeError("FORGE_GPU_HOURLY_USD=1.53 is required")
    pricing = amendment["pricing"]
    converted = Decimal(str(pricing["assumed_cny_per_hour"])) / Decimal(
        str(pricing["assumed_cny_per_usd"])
    )
    if converted.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) != declared_rate:
        raise ValueError("pricing must round CNY 11/h at 7.2 CNY/USD to USD 1.53/h")

    config = {
        **source,
        "run_id": amendment["run_id"],
        "benchmark_contract": amendment["benchmark_contract"],
        "sustained_overload": sustained,
        "gate": amendment["gate"],
        "source_finite_receipt": amendment["source_finite_receipt"],
        "_config_path": relative_path(amendment_path),
        "_source_phase7_1_config_path": relative_path(source_path),
        "_source_phase7_1_config_sha256": sha256_file(source_path),
        "_source_phase7_1_config_hash": source["_config_hash"],
    }
    config["_config_hash"] = finite._canonical_hash(
        {
            "amendment": amendment,
            "source_phase7_1_config_hash": source["_config_hash"],
            "phase5_contract_hash": source["_phase5_contract_hash"],
        }
    )
    return config


def _finite_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    path = REPO_ROOT / str(config["source_finite_receipt"])
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "complete" or receipt.get("phase") != "7.1":
        raise RuntimeError("archived finite A10 receipt is incomplete")
    if receipt.get("model", {}).get("artifact_sha256") != config["model"]["artifact_sha256"]:
        raise RuntimeError("sustained run model differs from the archived finite A10 receipt")
    archived_contract = (
        receipt.get("disclosure", {}).get("phase5_contract", {}).get("contract_hash")
    )
    if archived_contract != config["_phase5_contract_hash"]:
        raise RuntimeError("sustained run differs from the archived Phase 5 benchmark contract")
    capacity = receipt.get("metrics", {}).get("capacity", {})
    if float(capacity.get("measured_capacity_qps", 0)) <= 0:
        raise RuntimeError("archived same-box A10 capacity receipt is missing")
    return receipt


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
            raise RuntimeError(f"Phase 4 source and sustained model differ at {key}")
    artifact_path = REPO_ROOT / str(config["model"]["artifact_path"])
    actual = sha256_tree(artifact_path)
    expected = str(config["model"]["artifact_sha256"])
    if actual != expected:
        raise RuntimeError(f"sustained A10 artifact tree hash mismatch: {actual} != {expected}")
    manifest_path = REPO_ROOT / str(config["model"]["export_manifest"])
    manifest = json.loads(manifest_path.read_text())
    export = manifest.get("full_precision_export", {})
    logical_artifact_path = str(config["model"]["artifact_path"])
    if export.get("path") != logical_artifact_path or export.get("sha256") != expected:
        raise RuntimeError("export manifest does not identify the sustained serving artifact")
    receipt = {
        "version": 1,
        "status": "complete",
        "phase": "7.1",
        "scope": "sustained_overload_amendment",
        "git_sha": benchmark_git_sha(),
        "verified_at": _now(),
        "artifact": {
            "path": logical_artifact_path,
            "resolved_path": str(artifact_path.resolve()),
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
        raise FileNotFoundError("run sustained artifact verification before benchmarking")
    receipt = json.loads(ARTIFACT_RECEIPT.read_text())
    if (
        receipt.get("status") != "complete"
        or receipt.get("git_sha") != benchmark_git_sha()
        or receipt.get("artifact", {}).get("sha256") != config["model"]["artifact_sha256"]
    ):
        raise RuntimeError("sustained artifact receipt conflicts with the active benchmark")
    return receipt


def _poisson_offsets_for_duration(*, duration_s: float, qps: float, seed: int) -> list[float]:
    if duration_s <= 0 or qps <= 0:
        raise ValueError("duration and qps must be positive")
    generator = random.Random(seed)
    offsets = [0.0]
    current = 0.0
    while current < duration_s:
        current += generator.expovariate(qps)
        offsets.append(current)
    return offsets


async def _endpoint_stage(
    config: Mapping[str, Any],
    workload: Sequence[dict[str, Any]],
    archived: Mapping[str, Any],
    *,
    endpoint: str,
    direct_url: str,
    gateway_url: str,
) -> tuple[dict[str, Any], list[core.Observation]]:
    source_settings = config["overload"]
    settings = config["sustained_overload"]
    capacity = archived["metrics"]["capacity"]
    capacity_qps = float(capacity["measured_capacity_qps"])
    recovery_threshold = float(capacity["baseline_success_p95_s"]) * float(
        source_settings["recovery_p95_multiplier"]
    )
    model = str(config["model"]["served_name"])
    cells: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    observations: list[core.Observation] = []
    for index, multiplier_value in enumerate(settings["multipliers"]):
        multiplier = float(multiplier_value)
        offered_qps = capacity_qps * multiplier
        seed = int(config["workload"]["request_seed"]) + int(settings["seed_offset"]) + index
        offsets = _poisson_offsets_for_duration(
            duration_s=float(settings["minimum_arrival_duration_s"]),
            qps=offered_qps,
            seed=seed,
        )
        rows = core._select_rows(
            workload,
            config,
            profile=str(source_settings["length_profile"]),
            count=len(offsets),
            seed=seed,
        )
        warmup = core._select_rows(
            workload,
            config,
            profile=str(source_settings["length_profile"]),
            count=int(config["workload"]["warmup_requests"]),
            seed=seed - 1,
        )
        cell, items = await core._overload_cell(
            endpoint=endpoint,
            base_url=direct_url if endpoint == "direct" else gateway_url,
            direct_url=direct_url,
            gateway_url=gateway_url,
            model=model,
            rows=rows,
            warmup=warmup,
            cell_id=f"phase7-1-sustained-{multiplier:g}x-{endpoint}",
            offsets=offsets,
            deadline_s=float(source_settings["deadline_s"]),
            offered_qps=offered_qps,
            recovery_threshold_s=recovery_threshold,
            recovery_interval_s=float(source_settings["recovery_probe_interval_s"]),
            recovery_required=int(source_settings["recovery_consecutive_successes"]),
            hourly_usd=float(config["hardware"]["hourly_usd"]),
            concurrency=int(settings["max_client_concurrency"]),
        )
        cell.update(
            {
                "arrival_duration_s": offsets[-1],
                "minimum_arrival_duration_s": float(settings["minimum_arrival_duration_s"]),
                "scheduled_arrivals": len(offsets),
                "arrival_contract": (
                    "fixed-seed Poisson arrivals through the first offset >= minimum duration"
                ),
            }
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
        "capacity_receipt": capacity["raw_artifact"],
        "minimum_arrival_duration_s": float(settings["minimum_arrival_duration_s"]),
        "arrival_process": "duration_based_poisson_fixed_seed",
        "execution_contract": "all sustained bare-vLLM cells finish before gateway cells",
    }, observations


def _pair_stages(
    direct_stage: Mapping[str, Any], gateway_stage: Mapping[str, Any]
) -> dict[str, Any]:
    direct = {float(entry["multiplier"]): entry for entry in direct_stage["entries"]}
    gateway = {float(entry["multiplier"]): entry for entry in gateway_stage["entries"]}
    if direct.keys() != gateway.keys():
        raise RuntimeError("sustained direct/gateway cell sets differ")
    pairs = []
    for multiplier in direct:
        left = direct[multiplier]["cell"]
        right = gateway[multiplier]["cell"]
        if left["request_schedule_sha256"] != right["request_schedule_sha256"]:
            raise RuntimeError("sustained direct/gateway schedules differ")
        if left["warmup_schedule_sha256"] != right["warmup_schedule_sha256"]:
            raise RuntimeError("sustained direct/gateway warm-up schedules differ")
        pairs.append(
            {
                "multiplier": multiplier,
                "offered_qps": direct[multiplier]["offered_qps"],
                "execution_order": ["sustained_bare_vllm", "sustained_gateway"],
                "direct": left,
                "gateway": right,
            }
        )
    return {
        "cells": [cell for pair in pairs for cell in (pair["direct"], pair["gateway"])],
        "pairs": pairs,
        "schedules_sha256_verified": True,
        "measured_capacity_qps": direct_stage["measured_capacity_qps"],
        "direct_stage": direct_stage["raw_artifact"],
        "gateway_stage": gateway_stage["raw_artifact"],
    }


def _http_5xx_count(cell: Mapping[str, Any]) -> int:
    return sum(
        int(value)
        for status, value in cell.get("http_status_counts", {}).items()
        if 500 <= int(status) < 600
    )


def _status_count(cell: Mapping[str, Any], status: int) -> int:
    return int(cell.get("http_status_counts", {}).get(str(status), 0))


def _upstream_5xx_rate(cell: Mapping[str, Any], *, gateway: bool) -> float:
    admission_rejects = _status_count(cell, 429) if gateway else 0
    upstream_attempts = max(1, int(cell["requests"]) - admission_rejects)
    return _http_5xx_count(cell) / upstream_attempts


def _evaluate_gate(
    config: Mapping[str, Any],
    paired: Mapping[str, Any],
    gateway_stage: Mapping[str, Any],
) -> dict[str, Any]:
    settings = config["sustained_overload"]
    gate_config = config["gate"]
    minimum_duration = float(settings["minimum_arrival_duration_s"])
    max_delta = float(gate_config["max_upstream_5xx_rate_delta"])
    max_fast_reject = float(gate_config["max_fast_reject_s"])
    min_rejects = int(gate_config["min_429_rejects_per_overload_cell"])
    queue_bound = int(config["gateway"]["max_queue_requests"])
    request_rows = finite._load_request_rows(gateway_stage)
    rows = []
    for pair in paired["pairs"]:
        direct = pair["direct"]
        gateway = pair["gateway"]
        measured = [row for row in request_rows if row.get("cell_id") == gateway["cell_id"]]
        rejects = [row for row in measured if row.get("http_status") == 429]
        semantics = bool(rejects) and all(
            (not gate_config["require_retry_after"] or bool(row.get("retry_after")))
            and (
                not gate_config["require_overloaded_error_code"]
                or row.get("error_code") == "overloaded"
            )
            and float(row["client_e2e_s"]) <= max_fast_reject
            for row in rejects
        )
        sampled_queue = gateway["gateway_samples"].get("queue_depth_max")
        process_queue = (gateway.get("gateway") or {}).get("queue_high_watermark_process")
        bounded = (
            sampled_queue is not None
            and process_queue is not None
            and float(sampled_queue) <= queue_bound
            and float(process_queue) <= queue_bound
        )
        saturated = (
            sampled_queue is not None
            and float(sampled_queue) >= queue_bound
            and process_queue is not None
            and float(process_queue) >= queue_bound
        )
        decision_count = int(
            (gateway.get("gateway") or {}).get("routing_decisions", {}).get("reject_overload", 0)
        )
        direct_5xx_rate = _upstream_5xx_rate(direct, gateway=False)
        gateway_5xx_rate = _upstream_5xx_rate(gateway, gateway=True)
        delta = abs(gateway_5xx_rate - direct_5xx_rate)
        durations_pass = (
            float(direct["arrival_duration_s"]) >= minimum_duration
            and float(gateway["arrival_duration_s"]) >= minimum_duration
        )
        rejection_pass = (
            len(rejects) >= min_rejects
            and len(rejects) == _status_count(gateway, 429)
            and decision_count >= len(rejects)
            and semantics
        )
        rows.append(
            {
                "multiplier": pair["multiplier"],
                "offered_qps": pair["offered_qps"],
                "direct_arrival_duration_s": direct["arrival_duration_s"],
                "gateway_arrival_duration_s": gateway["arrival_duration_s"],
                "duration_pass": durations_pass,
                "direct_requests": direct["requests"],
                "gateway_requests": gateway["requests"],
                "http_429_count": len(rejects),
                "reject_overload_decisions": decision_count,
                "all_429_fast_with_retry_after_and_overloaded_code": semantics,
                "sampled_queue_max": sampled_queue,
                "process_queue_high_watermark": process_queue,
                "configured_queue_bound": queue_bound,
                "queue_saturated": saturated,
                "bounded_queue": bounded,
                "bare_vllm_upstream_5xx_rate": direct_5xx_rate,
                "gateway_upstream_5xx_rate": gateway_5xx_rate,
                "upstream_5xx_absolute_delta": delta,
                "upstream_5xx_parity_pass": delta <= max_delta + 1e-12,
                "rejection_semantics_pass": rejection_pass,
                "pass": (
                    durations_pass
                    and rejection_pass
                    and saturated
                    and bounded
                    and delta <= max_delta + 1e-12
                ),
            }
        )

    checks = {
        "duration_at_least_120s_per_cell": all(row["duration_pass"] for row in rows),
        "matched_same_box_schedules": (
            len(rows) == len(settings["multipliers"])
            and bool(paired.get("schedules_sha256_verified"))
        ),
        "queue_saturated_in_every_cell": all(row["queue_saturated"] for row in rows),
        "bounded_queue": all(row["bounded_queue"] for row in rows),
        "429_fast_reject_semantics": all(row["rejection_semantics_pass"] for row in rows),
        "upstream_5xx_parity": all(row["upstream_5xx_parity_pass"] for row in rows),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "sustained_overload_cells": rows,
        "thresholds": {
            "minimum_arrival_duration_s": minimum_duration,
            "max_upstream_5xx_rate_delta": max_delta,
            "max_fast_reject_s": max_fast_reject,
            "min_429_rejects_per_overload_cell": min_rejects,
            "queue_bound_requests": queue_bound,
        },
        "upstream_5xx_denominator": "all requests except gateway admission 429 responses",
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
    archived: Mapping[str, Any],
    artifact: Mapping[str, Any],
    direct_stage: Mapping[str, Any],
    gateway_stage: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    paired = _pair_stages(direct_stage, gateway_stage)
    gate = _evaluate_gate(config, paired, gateway_stage)
    if _parse_time(direct_stage["finished_at"]) > _parse_time(gateway_stage["started_at"]):
        raise RuntimeError("gateway sustained cells began before all bare-vLLM cells finished")
    hardware = host_disclosure()
    if "A10" not in str(hardware.get("gpu")):
        raise RuntimeError(f"sustained Gate 7.1 refuses non-A10 hardware: {hardware.get('gpu')!r}")
    started_value = os.environ.get("FORGE_PHASE7_SESSION_STARTED_AT")
    if not started_value:
        raise RuntimeError("FORGE_PHASE7_SESSION_STARTED_AT is required for the GPU ledger")
    started_at = _parse_time(started_value)
    finished_at = datetime.now(UTC)
    if started_at > finished_at:
        raise RuntimeError("Phase 7 session start is later than sustained receipt finalization")
    session_hours = (finished_at - started_at).total_seconds() / 3600
    hourly_usd = float(config["hardware"]["hourly_usd"])
    finite_path = REPO_ROOT / str(config["source_finite_receipt"])
    receipt = {
        "version": 1,
        "status": "complete",
        "phase": "7.1",
        "run_id": config["run_id"],
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "git_sha": benchmark_git_sha(),
        "model": config["model"],
        "server": {"identity": identity, "declared": config["server"]},
        "gateway": config["gateway"],
        "hardware": {**hardware, "declared_gpu_type": config["hardware"]["gpu_type"]},
        "workload": workload_receipt,
        "artifact_verification": {
            **artifact["artifact"],
            "receipt": relative_path(ARTIFACT_RECEIPT),
            "receipt_sha256": sha256_file(ARTIFACT_RECEIPT),
        },
        "metrics": {"sustained_overload": paired},
        "gate": gate,
        "production_blocked": gate["status"] != "pass",
        "prior_finite_a10_receipt": {
            "run_id": archived["run_id"],
            "git_sha": archived["git_sha"],
            "path": relative_path(finite_path),
            "sha256": sha256_file(finite_path),
            "gate_status_under_superseded_burst_proxy": archived["gate"]["status"],
        },
        "disclosure": {
            "gate_amendment": (
                "The human-approved 2026-08-21 amendment replaces the finite-burst per-cell "
                "429 proxy with duration-based arrivals of at least 120 seconds at 2x/3x/5x."
            ),
            "hardware_change": (
                "This rerun uses one NVIDIA A10 on Aliyun ECS. RTX 4090 Phase 5 results remain "
                "historical only and are not used for A10 latency or throughput comparisons."
            ),
            "scope": (
                "Measured single-node gateway overload contract only; not cloud production, "
                "not multi-GPU scaling, and not a Phase 7.2 Kubernetes result."
            ),
            "execution_order": (
                "All same-box sustained bare-vLLM cells completed before the gateway process "
                "was started and measured with identical schedules."
            ),
            "comparison_scope": config["comparison_scope"],
            "source_phase7_1_config": {
                "path": config["_source_phase7_1_config_path"],
                "sha256": config["_source_phase7_1_config_sha256"],
                "config_hash": config["_source_phase7_1_config_hash"],
            },
            "phase5_contract_hash": config["_phase5_contract_hash"],
            "original_negative_history": archived["disclosure"]["original_negative_history"],
            "precision": config["model"]["precision"],
            "training_time_quantization": config["model"]["training_time_quantization"],
            "deployment_quantization": config["model"]["deployment_quantization"],
            "fallback": "disabled; one physical R1b MTP vLLM replica",
        },
        "cost": {
            "delegated_session_hours_through_gate": session_hours,
            "hourly_usd": hourly_usd,
            "usd": session_hours * hourly_usd,
            "assumed_cny_per_hour": float(config["pricing"]["assumed_cny_per_hour"]),
            "assumed_cny_per_usd": float(config["pricing"]["assumed_cny_per_usd"]),
            "rate_source": "FORGE_GPU_HOURLY_USD=1.53",
            "scope": "current delegated Phase 7 session start through sustained gate receipt",
        },
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "raw_artifact": relative_path(FINAL_RECEIPT),
    }
    write_json_atomic(FINAL_RECEIPT, receipt)
    _append_run_record(receipt)
    append_jsonl_once(
        LEDGER_PATH,
        {
            "ledger_id": config["run_id"],
            "phase": "7.1",
            "operation": "a10_sustained_overload_amendment_2x_3x_5x",
            "status": "complete",
            "gate_status": gate["status"],
            "config_path": config["_config_path"],
            "config_hash": config["_config_hash"],
            "git_sha": benchmark_git_sha(),
            "gpu_type": config["hardware"]["gpu_type"],
            "gpu_hours": session_hours,
            "hourly_usd": hourly_usd,
            "usd": session_hours * hourly_usd,
            "rate_source": "FORGE_GPU_HOURLY_USD=1.53; CNY 11/h at 7.2 CNY/USD",
            "started_at": receipt["started_at"],
            "finished_at": receipt["finished_at"],
            "notes": "Current delegated session through sustained Gate 7.1 finalization.",
        },
        key="ledger_id",
    )
    return receipt


async def _run_stage(
    stage: str,
    config: Mapping[str, Any],
    workload: Sequence[dict[str, Any]],
    workload_receipt: Mapping[str, Any],
    *,
    direct_url: str,
    gateway_url: str,
) -> dict[str, Any]:
    archived = _finite_receipt(config)
    if stage == "bare":
        return await finite._run_clean_stage(
            "sustained-overload-a10-bare",
            config,
            lambda: _endpoint_stage(
                config,
                workload,
                archived,
                endpoint="direct",
                direct_url=direct_url,
                gateway_url=gateway_url,
            ),
        )
    if stage == "gateway":
        direct_stage = finite._require_stage("sustained-overload-a10-bare", config)
        gateway_stage = await finite._run_clean_stage(
            "sustained-overload-a10-gateway",
            config,
            lambda: _endpoint_stage(
                config,
                workload,
                archived,
                endpoint="gateway",
                direct_url=direct_url,
                gateway_url=gateway_url,
            ),
        )
        return _write_final_receipt(
            config,
            workload_receipt,
            archived=archived,
            artifact=_require_artifact(config),
            direct_stage=direct_stage,
            gateway_stage=gateway_stage,
            identity=await finite._identity(config, direct_url, gateway_url),
        )
    raise ValueError(f"unsupported sustained Gate 7.1 stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=relative_path(CONFIG_PATH))
    parser.add_argument("--direct-url", default="http://127.0.0.1:8000")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:9000")
    parser.add_argument(
        "--stage",
        choices=("verify-artifact", "bare", "gateway", "import-final"),
        required=True,
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    sha = benchmark_git_sha()
    if _GIT_SHA.fullmatch(sha) is None:
        raise RuntimeError("sustained Gate 7.1 requires a full lowercase Git SHA")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REQUESTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "verify-artifact":
        _finite_receipt(config)
        result = _verify_artifact(config)
    elif args.stage == "import-final":
        if not FINAL_RECEIPT.is_file():
            raise FileNotFoundError(FINAL_RECEIPT)
        result = json.loads(FINAL_RECEIPT.read_text())
        if result.get("config_hash") != config["_config_hash"]:
            raise RuntimeError("synced sustained Gate 7.1 receipt conflicts with local config")
        _append_run_record(result)
    else:
        _require_artifact(config)
        workload, workload_receipt = core._phase4_workload(config)
        result = asyncio.run(
            _run_stage(
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
