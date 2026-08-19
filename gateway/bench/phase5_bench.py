#!/usr/bin/env python3
"""Run the immutable Phase 5 gateway benchmark stages on the remote GPU host."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import httpx
import yaml

from forge.bench.config import load_phase4_config, phase4_workload_path
from forge.bench.loadgen import VramSampler, normalize_verifier_input
from forge.bench.metrics import (
    parse_prometheus,
    prometheus_delta,
    summarize_latencies,
    summarize_vllm_metrics,
)
from forge.bench.preflight import (
    benchmark_git_sha,
    host_disclosure,
    require_verified_artifact,
)
from forge.bench.system_load import SystemLoadSampler
from forge.bench.workload import build_workload, load_workload
from forge.teacher.filters import breakdown_dict
from forge.train.artifacts import append_jsonl_once, write_json_atomic, write_jsonl_atomic
from forge.train.config import REPO_ROOT, canonical_json, relative_path, sha256_file
from forge.verify.verifier import score

RESULTS_DIR = REPO_ROOT / "results/phase5"
RAW_DIR = RESULTS_DIR / "raw"
REQUESTS_DIR = RESULTS_DIR / "requests"
FINAL_RECEIPT = RAW_DIR / "phase5_gateway_bench.json"
RUNS_PATH = REPO_ROOT / "results/runs.jsonl"
LEDGER_PATH = RESULTS_DIR / "gpu_ledger.jsonl"
PROFILE_EVIDENCE = RESULTS_DIR / "profile/optimization.json"
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass
class Observation:
    endpoint: str
    cell_id: str
    request_id: str
    request_input_sha256: str
    scheduled_offset_s: float
    dispatch_delay_s: float
    client_ttft_s: float | None
    client_e2e_s: float
    client_mean_itl_s: float | None
    output_tokens: int
    stream_events: int
    http_status: int | None
    error_type: str | None
    error_code: str | None
    retry_after: str | None
    forge_route: str | None
    deadline_missed: bool
    output: str
    verifier_input: str
    verifier: dict[str, Any] | None


@dataclass
class PeakConcurrency:
    current: int = 0
    peak: int = 0

    def enter(self) -> None:
        self.current += 1
        self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        self.current -= 1


class PrometheusSampler:
    """Sample gateway gauges during a cell; counters are handled by snapshots."""

    def __init__(self, url: str | None, *, interval_s: float = 0.1) -> None:
        self.url = url
        self.interval_s = interval_s
        self.queue_depth: list[float] = []
        self.active_requests: list[float] = []
        self._stop = asyncio.Event()

    async def run(self) -> None:
        if self.url is None:
            return
        async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
            while not self._stop.is_set():
                try:
                    response = await client.get(self.url)
                    response.raise_for_status()
                    parsed = parse_prometheus(response.text)
                    self.queue_depth.append(_metric_sum(parsed, "forge_gateway_queue_depth"))
                    self.active_requests.append(
                        _metric_sum(parsed, "forge_gateway_active_requests")
                    )
                except httpx.HTTPError:
                    pass
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)

    def stop(self) -> None:
        self._stop.set()

    def summary(self) -> dict[str, Any]:
        return {
            "measurement_side": "gateway_prometheus_poll",
            "samples": len(self.queue_depth),
            "queue_depth_max": max(self.queue_depth) if self.queue_depth else None,
            "queue_depth_mean": fmean(self.queue_depth) if self.queue_depth else None,
            "active_requests_max": (max(self.active_requests) if self.active_requests else None),
        }


def _metric_sum(
    metrics: Mapping[str, Mapping[tuple[tuple[str, str], ...], float]], name: str
) -> float:
    return sum(metrics.get(name, {}).values())


def _labels_value(
    metrics: Mapping[str, Mapping[tuple[tuple[str, str], ...], float]],
    name: str,
    label: str,
    value: str,
) -> float:
    total = 0.0
    for labels, sample in metrics.get(name, {}).items():
        if dict(labels).get(label) == value:
            total += sample
    return total


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or config.get("phase") != 5:
        raise ValueError("Phase 5 benchmark config must be a mapping with phase: 5")
    if float(config["hardware"]["hourly_usd"]) != 0.30:
        raise ValueError("Phase 5 benchmark config must pin hourly_usd to 0.30")
    supplied_rate = os.environ.get("FORGE_GPU_HOURLY_USD")
    if supplied_rate is None or float(supplied_rate) != 0.30:
        raise RuntimeError("FORGE_GPU_HOURLY_USD=0.30 is required")
    config["_config_path"] = relative_path(config_path)
    config["_config_hash"] = _canonical_hash(
        {key: value for key, value in config.items() if not key.startswith("_")}
    )
    return config


def _phase4_workload(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = str(config["workload"]["source_config"])
    source = load_phase4_config(source_path)
    build_workload(source_path, smoke=False)
    path = phase4_workload_path(source, smoke=False)
    rows = load_workload(source, smoke=False)
    if not rows:
        raise RuntimeError("Phase 5 source workload is empty")
    return rows, {
        "source_config": source_path,
        "source_config_hash": source["_config_hash"],
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
    }


def _select_rows(
    workload: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
    *,
    profile: str,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    profile_config = config["workload"]["length_profiles"][profile]
    targets = [int(value) for value in profile_config["input_token_targets"]]
    output_caps = [int(value) for value in profile_config["output_token_caps"]]
    pools: dict[int, list[dict[str, Any]]] = {}
    for target in targets:
        exact = [row for row in workload if int(row["input_token_target"]) == target]
        candidates = exact or sorted(
            workload,
            key=lambda row: abs(int(row["prompt_tokens"]) - target),
        )
        shuffled = list(candidates)
        random.Random(seed + target).shuffle(shuffled)
        pools[target] = shuffled
    positions = Counter()
    selected: list[dict[str, Any]] = []
    for index in range(count):
        target = targets[index % len(targets)]
        pool = pools[target]
        row = dict(pool[positions[target] % len(pool)])
        positions[target] += 1
        row["max_tokens"] = output_caps[index % len(output_caps)]
        row["phase5_request_id"] = hashlib.sha256(
            f"{row['request_id']}:{profile}:{index}:{seed}".encode()
        ).hexdigest()[:24]
        selected.append(row)
    return selected


def _request_payload(model: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": row["messages"],
        "temperature": 0,
        "max_tokens": int(row["max_tokens"]),
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _input_hash(model: str, row: Mapping[str, Any]) -> str:
    return _canonical_hash(_request_payload(model, row))


def _sse_content(chunk: Mapping[str, Any]) -> tuple[str, str | None]:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", None
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return "", None
    delta = choice.get("delta")
    content = delta.get("content") if isinstance(delta, Mapping) else None
    reason = choice.get("finish_reason")
    return (content if isinstance(content, str) else ""), (
        reason if isinstance(reason, str) else None
    )


async def _stream_one(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    base_url: str,
    model: str,
    row: Mapping[str, Any],
    cell_id: str,
    scheduled_at: float,
    origin: float,
    deadline_s: float,
    semaphore: asyncio.Semaphore,
    peak: PeakConcurrency,
) -> Observation:
    delay = max(0.0, scheduled_at - time.perf_counter())
    if delay:
        await asyncio.sleep(delay)
    await semaphore.acquire()
    peak.enter()
    sent_at = time.perf_counter()
    first_event_at: float | None = None
    event_times: list[float] = []
    output_parts: list[str] = []
    usage_tokens: int | None = None
    status: int | None = None
    error_type: str | None = None
    error_code: str | None = None
    retry_after: str | None = None
    forge_route: str | None = None
    payload = _request_payload(model, row)
    headers = {
        "X-Client-ID": "phase5-benchmark",
        "X-Allow-Degrade": "0",
        "X-Request-Timeout-Ms": str(int(deadline_s * 1000)),
    }
    try:
        async with asyncio.timeout(deadline_s):
            async with client.stream(
                "POST",
                f"{base_url.rstrip('/')}/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                status = response.status_code
                retry_after = response.headers.get("retry-after")
                forge_route = response.headers.get("x-forge-route")
                if response.is_error:
                    body = (await response.aread()).decode(errors="replace")[:2000]
                    error_type = f"http_{status}"
                    try:
                        parsed = json.loads(body)
                        error_code = parsed.get("error", {}).get("code")
                    except json.JSONDecodeError:
                        error_code = None
                else:
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            error_type = "invalid_sse_json"
                            break
                        usage = chunk.get("usage")
                        if isinstance(usage, Mapping) and isinstance(
                            usage.get("completion_tokens"), int
                        ):
                            usage_tokens = int(usage["completion_tokens"])
                        content, _ = _sse_content(chunk)
                        if content:
                            now = time.perf_counter()
                            if first_event_at is None:
                                first_event_at = now
                            event_times.append(now)
                            output_parts.append(content)
    except TimeoutError:
        error_type = "deadline_exceeded"
    except httpx.HTTPError as error:
        error_type = f"{type(error).__name__}:{error}"
    finally:
        completed_at = time.perf_counter()
        peak.leave()
        semaphore.release()
    output = "".join(output_parts).strip()
    output_tokens = usage_tokens if usage_tokens is not None else len(event_times)
    e2e = completed_at - sent_at
    ttft = first_event_at - sent_at if first_event_at is not None else None
    itl = max(0.0, e2e - ttft) / (output_tokens - 1) if ttft and output_tokens > 1 else None
    verifier_input = normalize_verifier_input(output)
    scored = None
    if status is not None and 200 <= status < 300 and error_type is None:
        scored = breakdown_dict(score({"label": row["label"]}, verifier_input))
    return Observation(
        endpoint=endpoint,
        cell_id=cell_id,
        request_id=str(row["phase5_request_id"]),
        request_input_sha256=_input_hash(model, row),
        scheduled_offset_s=scheduled_at - origin,
        dispatch_delay_s=max(0.0, sent_at - scheduled_at),
        client_ttft_s=ttft,
        client_e2e_s=e2e,
        client_mean_itl_s=itl,
        output_tokens=output_tokens,
        stream_events=len(event_times),
        http_status=status,
        error_type=error_type,
        error_code=error_code,
        retry_after=retry_after,
        forge_route=forge_route,
        deadline_missed=error_type == "deadline_exceeded",
        output=output,
        verifier_input=verifier_input,
        verifier=scored,
    )


async def _snapshot(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.HTTPError:
        return ""


async def _wait_vllm_idle(direct_url: str, *, timeout_s: float = 180) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        metrics = parse_prometheus(await _snapshot(f"{direct_url.rstrip('/')}/metrics"))
        running = _metric_sum(metrics, "vllm:num_requests_running")
        waiting = _metric_sum(metrics, "vllm:num_requests_waiting")
        if running == 0 and waiting == 0:
            return
        await asyncio.sleep(0.25)
    raise TimeoutError("vLLM did not drain before the next Phase 5 cell")


async def _warmup(
    *,
    endpoint: str,
    base_url: str,
    direct_url: str,
    model: str,
    rows: list[dict[str, Any]],
    concurrency: int,
    deadline_s: float,
) -> dict[str, Any]:
    await _wait_vllm_idle(direct_url)
    sampler = SystemLoadSampler(enabled=True)
    sampler_task = asyncio.create_task(sampler.run())
    try:
        await _run_requests(
            endpoint=endpoint,
            base_url=base_url,
            model=model,
            rows=rows,
            cell_id=f"warmup-{endpoint}",
            offsets=[0.0] * len(rows),
            concurrency=concurrency,
            deadline_s=deadline_s,
        )
    finally:
        sampler.stop()
        await sampler_task
    await _wait_vllm_idle(direct_url)
    return sampler.summary()


async def _run_requests(
    *,
    endpoint: str,
    base_url: str,
    model: str,
    rows: list[dict[str, Any]],
    cell_id: str,
    offsets: list[float],
    concurrency: int,
    deadline_s: float,
) -> tuple[list[Observation], int, float]:
    if len(rows) != len(offsets):
        raise ValueError("rows and offsets must have identical lengths")
    origin = time.perf_counter()
    peak = PeakConcurrency()
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(
        max_connections=concurrency + 8,
        max_keepalive_connections=concurrency + 4,
    )
    async with httpx.AsyncClient(timeout=None, limits=limits, trust_env=False) as client:
        tasks = [
            asyncio.create_task(
                _stream_one(
                    client,
                    endpoint=endpoint,
                    base_url=base_url,
                    model=model,
                    row=row,
                    cell_id=cell_id,
                    scheduled_at=origin + offset,
                    origin=origin,
                    deadline_s=deadline_s,
                    semaphore=semaphore,
                    peak=peak,
                )
            )
            for row, offset in zip(rows, offsets, strict=True)
        ]
        observations = await asyncio.gather(*tasks)
    return observations, peak.peak, max(time.perf_counter() - origin, 1e-9)


def _poisson_offsets(*, count: int, qps: float, seed: int) -> list[float]:
    generator = random.Random(seed)
    offsets: list[float] = []
    current = 0.0
    for index in range(count):
        if index:
            current += generator.expovariate(qps)
        offsets.append(current)
    return offsets


def _gateway_metrics(
    before: Mapping[str, Mapping[tuple[tuple[str, str], ...], float]],
    after: Mapping[str, Mapping[tuple[tuple[str, str], ...], float]],
) -> dict[str, Any]:
    delta = prometheus_delta(before, after)
    decisions = {
        name: int(
            round(
                _labels_value(
                    delta,
                    "forge_gateway_routing_decisions_total",
                    "decision",
                    name,
                )
            )
        )
        for name in (
            "primary",
            "fallback",
            "queued",
            "reject_overload",
            "reject_deadline",
            "reject_rate_limit",
            "reject_unavailable",
            "reject_bad_request",
        )
    }
    return {
        "measurement_side": "gateway_prometheus_delta",
        "routing_decisions": decisions,
        "fallback_share": (
            decisions["fallback"] / (decisions["primary"] + decisions["fallback"])
            if decisions["primary"] + decisions["fallback"]
            else 0.0
        ),
        "queue_high_watermark_process": _metric_sum(after, "forge_gateway_queue_high_watermark"),
        "response_classes": {
            name: int(round(_labels_value(delta, "forge_gateway_responses_total", "class", name)))
            for name in ("2xx", "4xx", "5xx")
        },
    }


def _summarize(
    observations: list[Observation],
    *,
    elapsed_s: float,
    offered_qps: float | None,
    concurrency: int,
    peak_concurrency: int,
    hourly_usd: float,
    vllm_metrics: dict[str, Any],
    gateway_metrics: dict[str, Any] | None,
    gateway_samples: dict[str, Any],
    vram: dict[str, Any],
    co_tenancy: dict[str, Any],
    warmup_co_tenancy: dict[str, Any],
) -> dict[str, Any]:
    successful_http = [
        item
        for item in observations
        if item.http_status is not None
        and 200 <= item.http_status < 300
        and item.error_type is None
    ]
    verifier_successes = [
        item
        for item in successful_http
        if item.verifier is not None and bool(item.verifier.get("task_success"))
    ]
    all_responses = [item for item in observations if item.http_status is not None]
    rejected = [item for item in observations if item.http_status and item.http_status >= 400]
    fast_rejects = [item for item in rejected if item.client_e2e_s <= 1.0]
    status_counts = Counter(
        str(item.http_status) if item.http_status is not None else "transport_error"
        for item in observations
    )
    error_codes = Counter(item.error_code or item.error_type or "none" for item in rejected)
    successful_task_throughput = len(verifier_successes) / elapsed_s
    return {
        "requests": len(observations),
        "http_status_counts": dict(sorted(status_counts.items())),
        "error_semantics": {
            "error_codes": dict(sorted(error_codes.items())),
            "retry_after_present": sum(item.retry_after is not None for item in rejected),
            "fast_rejects_under_1s": len(fast_rejects),
        },
        "errors": len(observations) - len(successful_http),
        "error_rate": 1 - len(successful_http) / len(observations),
        "deadline_misses": sum(item.deadline_missed for item in observations),
        "deadline_miss_rate": sum(item.deadline_missed for item in observations)
        / len(observations),
        "verifier_successes": len(verifier_successes),
        "verifier_task_success_rate": (
            len(verifier_successes) / len(successful_http) if successful_http else 0.0
        ),
        "arrival_rate_qps": offered_qps,
        "arrival_process": (
            "poisson_fixed_seed" if offered_qps is not None else "closed_loop_fixed_concurrency"
        ),
        "concurrency": concurrency,
        "max_observed_in_flight": peak_concurrency,
        "elapsed_s": elapsed_s,
        "response_throughput_per_s": len(all_responses) / elapsed_s,
        "successful_request_throughput_per_s": len(successful_http) / elapsed_s,
        "successful_task_throughput_per_s": successful_task_throughput,
        "achieved_qps_ratio": (
            len(successful_http) / elapsed_s / offered_qps if offered_qps else None
        ),
        "output_tokens_per_s": sum(item.output_tokens for item in successful_http) / elapsed_s,
        "client": {
            "measurement_side": "client_wall_clock_streaming",
            "ttft": summarize_latencies(
                [item.client_ttft_s for item in successful_http if item.client_ttft_s is not None]
            ),
            "itl": summarize_latencies(
                [
                    item.client_mean_itl_s
                    for item in successful_http
                    if item.client_mean_itl_s is not None
                ]
            ),
            "e2e_success": summarize_latencies([item.client_e2e_s for item in successful_http]),
            "e2e_all_responses": summarize_latencies([item.client_e2e_s for item in all_responses]),
            "fast_reject": summarize_latencies([item.client_e2e_s for item in fast_rejects]),
            "dispatch_delay": summarize_latencies([item.dispatch_delay_s for item in observations]),
        },
        "vllm": vllm_metrics,
        "gateway": gateway_metrics,
        "gateway_samples": gateway_samples,
        "vram": vram,
        "co_tenancy": co_tenancy,
        "warmup_co_tenancy": warmup_co_tenancy,
        "cost_per_1k_successful_tasks_usd": (
            hourly_usd * 1000 / (successful_task_throughput * 3600)
            if successful_task_throughput > 0
            else None
        ),
        "cost_formula": "hourly_usd*1000/(verifier_successes_per_second*3600)",
    }


async def _run_cell(
    *,
    endpoint: str,
    base_url: str,
    direct_url: str,
    gateway_url: str,
    model: str,
    rows: list[dict[str, Any]],
    warmup_rows: list[dict[str, Any]],
    cell_id: str,
    offsets: list[float],
    concurrency: int,
    deadline_s: float,
    offered_qps: float | None,
    hourly_usd: float,
) -> tuple[dict[str, Any], list[Observation]]:
    warmup_load = await _warmup(
        endpoint=endpoint,
        base_url=base_url,
        direct_url=direct_url,
        model=model,
        rows=warmup_rows,
        concurrency=min(concurrency, len(warmup_rows)),
        deadline_s=deadline_s,
    )
    vllm_before = parse_prometheus(await _snapshot(f"{direct_url.rstrip('/')}/metrics"))
    gateway_before = (
        parse_prometheus(await _snapshot(f"{gateway_url.rstrip('/')}/metrics"))
        if endpoint == "gateway"
        else {}
    )
    vram = VramSampler(enabled=True)
    load = SystemLoadSampler(enabled=True)
    gateway_sampler = PrometheusSampler(
        f"{gateway_url.rstrip('/')}/metrics" if endpoint == "gateway" else None
    )
    tasks = [
        asyncio.create_task(vram.run()),
        asyncio.create_task(load.run()),
        asyncio.create_task(gateway_sampler.run()),
    ]
    try:
        observations, peak, elapsed = await _run_requests(
            endpoint=endpoint,
            base_url=base_url,
            model=model,
            rows=rows,
            cell_id=cell_id,
            offsets=offsets,
            concurrency=concurrency,
            deadline_s=deadline_s,
        )
    finally:
        vram.stop()
        load.stop()
        gateway_sampler.stop()
        await asyncio.gather(*tasks)
    vllm_after = parse_prometheus(await _snapshot(f"{direct_url.rstrip('/')}/metrics"))
    gateway_after = (
        parse_prometheus(await _snapshot(f"{gateway_url.rstrip('/')}/metrics"))
        if endpoint == "gateway"
        else {}
    )
    await _wait_vllm_idle(direct_url)
    summary = _summarize(
        observations,
        elapsed_s=elapsed,
        offered_qps=offered_qps,
        concurrency=concurrency,
        peak_concurrency=peak,
        hourly_usd=hourly_usd,
        vllm_metrics=summarize_vllm_metrics(prometheus_delta(vllm_before, vllm_after)),
        gateway_metrics=(
            _gateway_metrics(gateway_before, gateway_after) if endpoint == "gateway" else None
        ),
        gateway_samples=gateway_sampler.summary(),
        vram=vram.summary(),
        co_tenancy=load.summary(),
        warmup_co_tenancy=warmup_load,
    )
    summary.update(
        {
            "cell_id": cell_id,
            "endpoint": endpoint,
            "request_schedule_sha256": _canonical_hash(
                [
                    {
                        "request_id": row["phase5_request_id"],
                        "input_sha256": _input_hash(model, row),
                        "offset_s": offset,
                    }
                    for row, offset in zip(rows, offsets, strict=True)
                ]
            ),
            "warmup_schedule_sha256": _canonical_hash(
                [
                    {
                        "request_id": row["phase5_request_id"],
                        "input_sha256": _input_hash(model, row),
                    }
                    for row in warmup_rows
                ]
            ),
        }
    )
    return summary, observations


def _cell_contaminated(cell: Mapping[str, Any]) -> bool:
    return bool(cell["co_tenancy"]["contaminated"]) or bool(
        cell["warmup_co_tenancy"]["contaminated"]
    )


async def _wait_host_clean() -> None:
    capacity = SystemLoadSampler(enabled=False)
    cores = capacity.logical_cpu_count
    threshold = capacity.load_threshold
    while os.getloadavg()[0] > threshold:
        print(
            f"host load1={os.getloadavg()[0]:.2f} exceeds threshold={threshold:.2f}; "
            f"effective_cores={cores:g} source={capacity.core_count_source}; "
            "waiting 30s before the clean rerun",
            flush=True,
        )
        await asyncio.sleep(30)


def _stage_paths(stage: str) -> tuple[Path, Path]:
    return RAW_DIR / f"{stage}.json", REQUESTS_DIR / f"{stage}.requests.jsonl"


async def _run_clean_stage(
    stage: str,
    config: Mapping[str, Any],
    run_once: Callable[[], Awaitable[tuple[dict[str, Any], list[Observation]]]],
) -> dict[str, Any]:
    raw_path, request_path = _stage_paths(stage)
    if raw_path.is_file():
        existing = json.loads(raw_path.read_text())
        if existing.get("config_hash") != config["_config_hash"]:
            raise RuntimeError(f"existing {stage} artifact conflicts with the config")
        if not request_path.is_file() or existing["request_artifact"]["sha256"] != sha256_file(
            request_path
        ):
            raise RuntimeError(f"existing {stage} request artifact hash verification failed")
        print(f"validated existing Phase 5 stage: {relative_path(raw_path)}")
        return existing
    attempt = 0
    contaminated_attempts: list[dict[str, Any]] = []
    while True:
        attempt += 1
        await _wait_host_clean()
        payload, observations = await run_once()
        contaminated = any(_cell_contaminated(cell) for cell in payload["cells"])
        if not contaminated:
            break
        attempt_raw = RAW_DIR / f"{stage}.contaminated-{attempt:02d}.json"
        attempt_requests = REQUESTS_DIR / f"{stage}.contaminated-{attempt:02d}.requests.jsonl"
        write_jsonl_atomic(attempt_requests, [asdict(item) for item in observations])
        attempt_receipt = {
            **payload,
            "status": "contaminated-rerun-required",
            "reason": "sampled_load1_exceeded_half_logical_core_count",
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
        print(f"Phase 5 {stage} attempt {attempt} contaminated; retained and rerunning")
    write_jsonl_atomic(request_path, [asdict(item) for item in observations])
    receipt = {
        **payload,
        "version": 1,
        "status": "complete",
        "phase": 5,
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
        "finished_at": _now(),
        "raw_artifact": relative_path(raw_path),
    }
    write_json_atomic(raw_path, receipt)
    print(f"Phase 5 {stage} complete: {relative_path(raw_path)}")
    return receipt


async def _capacity_stage(
    config: Mapping[str, Any],
    workload: Sequence[dict[str, Any]],
    *,
    direct_url: str,
    gateway_url: str,
) -> tuple[dict[str, Any], list[Observation]]:
    settings = config["capacity"]
    model = str(config["model"]["served_name"])
    rows = _select_rows(
        workload,
        config,
        profile=str(settings["length_profile"]),
        count=int(settings["measurement_requests"]),
        seed=int(config["workload"]["request_seed"]),
    )
    warmup = _select_rows(
        workload,
        config,
        profile=str(settings["length_profile"]),
        count=int(config["workload"]["warmup_requests"]),
        seed=int(config["workload"]["request_seed"]) - 1,
    )
    cells: list[dict[str, Any]] = []
    observations: list[Observation] = []
    baseline_p95: float | None = None
    for index, qps_value in enumerate(settings["arrival_rates_qps"]):
        qps = float(qps_value)
        offsets = _poisson_offsets(
            count=len(rows), qps=qps, seed=int(config["workload"]["request_seed"]) + index
        )
        cell, items = await _run_cell(
            endpoint="direct",
            base_url=direct_url,
            direct_url=direct_url,
            gateway_url=gateway_url,
            model=model,
            rows=rows,
            warmup_rows=warmup,
            cell_id=f"capacity-direct-{qps:g}qps",
            offsets=offsets,
            concurrency=128,
            deadline_s=float(settings["deadline_s"]),
            offered_qps=qps,
            hourly_usd=float(config["hardware"]["hourly_usd"]),
        )
        p95 = cell["client"]["e2e_success"]["p95_s"]
        if baseline_p95 is None and p95 is not None:
            baseline_p95 = float(p95)
        checks = {
            "error_rate": cell["error_rate"] <= float(settings["max_error_rate"]),
            "deadline_miss_rate": cell["deadline_miss_rate"]
            <= float(settings["max_deadline_miss_rate"]),
            "achieved_qps_ratio": cell["achieved_qps_ratio"]
            >= float(settings["min_achieved_qps_ratio"]),
            "p95_inflation": (
                p95 is not None
                and baseline_p95 is not None
                and p95 <= baseline_p95 * float(settings["max_p95_inflation"])
            ),
        }
        cell["stability_checks"] = checks
        cell["stable"] = all(checks.values())
        cells.append(cell)
        observations.extend(items)
    stable = [cell for cell in cells if cell["stable"]]
    if not stable:
        raise RuntimeError("capacity calibration found no stable direct-vLLM point")
    capacity_qps = max(float(cell["arrival_rate_qps"]) for cell in stable)
    max_stable_concurrency = max(int(cell["max_observed_in_flight"]) for cell in stable)
    return {
        "started_at": _now(),
        "cells": cells,
        "measured_capacity_qps": capacity_qps,
        "max_stable_concurrency": max_stable_concurrency,
        "baseline_success_p95_s": baseline_p95,
        "capacity_definition": (
            "highest offered Poisson QPS meeting error, deadline, achieved-QPS, and p95 gates"
        ),
    }, observations


async def _profile_stage(
    config: Mapping[str, Any],
    workload: Sequence[dict[str, Any]],
    *,
    direct_url: str,
    gateway_url: str,
) -> tuple[dict[str, Any], list[Observation]]:
    settings = config["profile"]
    rows = _select_rows(
        workload,
        config,
        profile=str(settings["length_profile"]),
        count=int(settings["measurement_requests"]),
        seed=int(config["workload"]["request_seed"]) + 500,
    )
    warmup = _select_rows(
        workload,
        config,
        profile=str(settings["length_profile"]),
        count=int(config["workload"]["warmup_requests"]),
        seed=int(config["workload"]["request_seed"]) + 499,
    )
    cell, observations = await _run_cell(
        endpoint="gateway",
        base_url=gateway_url,
        direct_url=direct_url,
        gateway_url=gateway_url,
        model=str(config["model"]["served_name"]),
        rows=rows,
        warmup_rows=warmup,
        cell_id="profile-gateway-mixed-c32",
        offsets=[0.0] * len(rows),
        concurrency=int(settings["concurrency"]),
        deadline_s=float(settings["deadline_s"]),
        offered_qps=None,
        hourly_usd=float(config["hardware"]["hourly_usd"]),
    )
    return {
        "started_at": _now(),
        "cells": [cell],
        "profile_contract": {
            "length_profile": settings["length_profile"],
            "concurrency": settings["concurrency"],
            "measurement_requests": settings["measurement_requests"],
            "external_profile_required": True,
        },
    }, observations


def _paired_overhead(direct: Mapping[str, Any], gateway: Mapping[str, Any]) -> dict[str, Any]:
    if direct["request_schedule_sha256"] != gateway["request_schedule_sha256"]:
        raise RuntimeError("direct/gateway request schedules are not identical")

    def delta(path: tuple[str, ...]) -> float | None:
        left: Any = direct
        right: Any = gateway
        for key in path:
            left = left[key]
            right = right[key]
        if left is None or right is None or float(left) == 0:
            return None
        return (float(right) - float(left)) / float(left) * 100

    return {
        "e2e_p50_overhead_pct": delta(("client", "e2e_success", "p50_s")),
        "e2e_p95_overhead_pct": delta(("client", "e2e_success", "p95_s")),
        "ttft_p50_overhead_pct": delta(("client", "ttft", "p50_s")),
        "ttft_p95_overhead_pct": delta(("client", "ttft", "p95_s")),
        "throughput_delta_pct": delta(("successful_request_throughput_per_s",)),
    }


async def _direct_gateway_stage(
    config: Mapping[str, Any],
    workload: Sequence[dict[str, Any]],
    *,
    direct_url: str,
    gateway_url: str,
) -> tuple[dict[str, Any], list[Observation]]:
    settings = config["direct_gateway"]
    model = str(config["model"]["served_name"])
    cells: list[dict[str, Any]] = []
    observations: list[Observation] = []
    pairs: list[dict[str, Any]] = []
    index = 0
    for profile in config["workload"]["length_profiles"]:
        for concurrency_value in settings["concurrency"]:
            concurrency = int(concurrency_value)
            seed = int(config["workload"]["request_seed"]) + 1000 + index
            rows = _select_rows(
                workload,
                config,
                profile=profile,
                count=int(config["workload"]["measurement_requests"]),
                seed=seed,
            )
            warmup = _select_rows(
                workload,
                config,
                profile=profile,
                count=int(config["workload"]["warmup_requests"]),
                seed=seed - 1,
            )
            order = ("direct", "gateway") if index % 2 == 0 else ("gateway", "direct")
            pair: dict[str, Any] = {
                "length_profile": profile,
                "concurrency": concurrency,
                "execution_order": list(order),
            }
            for endpoint in order:
                cell, items = await _run_cell(
                    endpoint=endpoint,
                    base_url=direct_url if endpoint == "direct" else gateway_url,
                    direct_url=direct_url,
                    gateway_url=gateway_url,
                    model=model,
                    rows=rows,
                    warmup_rows=warmup,
                    cell_id=f"direct-gateway-{profile}-c{concurrency}-{endpoint}",
                    offsets=[0.0] * len(rows),
                    concurrency=concurrency,
                    deadline_s=float(settings["deadline_s"]),
                    offered_qps=None,
                    hourly_usd=float(config["hardware"]["hourly_usd"]),
                )
                cells.append(cell)
                observations.extend(items)
                pair[endpoint] = cell
            pair["overhead"] = _paired_overhead(pair["direct"], pair["gateway"])
            pairs.append(pair)
            index += 1
    return {"started_at": _now(), "cells": cells, "pairs": pairs}, observations


async def _recovery_probes(
    *,
    endpoint: str,
    base_url: str,
    model: str,
    row: dict[str, Any],
    cell_id: str,
    burst_end_at: float,
    deadline_s: float,
    threshold_s: float,
    interval_s: float,
    required: int,
) -> tuple[float | None, list[Observation]]:
    await asyncio.sleep(max(0.0, burst_end_at - time.perf_counter()))
    started = time.perf_counter()
    consecutive = 0
    probes: list[Observation] = []
    while time.perf_counter() - started < 180:
        observations, _, _ = await _run_requests(
            endpoint=endpoint,
            base_url=base_url,
            model=model,
            rows=[row],
            cell_id=f"{cell_id}-recovery",
            offsets=[0.0],
            concurrency=1,
            deadline_s=deadline_s,
        )
        probe = observations[0]
        probes.append(probe)
        healthy = (
            probe.http_status is not None
            and 200 <= probe.http_status < 300
            and probe.error_type is None
            and probe.client_e2e_s <= threshold_s
        )
        consecutive = consecutive + 1 if healthy else 0
        if consecutive >= required:
            return time.perf_counter() - burst_end_at, probes
        await asyncio.sleep(interval_s)
    return None, probes


async def _overload_cell(
    *,
    endpoint: str,
    base_url: str,
    direct_url: str,
    gateway_url: str,
    model: str,
    rows: list[dict[str, Any]],
    warmup: list[dict[str, Any]],
    cell_id: str,
    offsets: list[float],
    deadline_s: float,
    offered_qps: float,
    recovery_threshold_s: float,
    recovery_interval_s: float,
    recovery_required: int,
    hourly_usd: float,
) -> tuple[dict[str, Any], list[Observation]]:
    warmup_load = await _warmup(
        endpoint=endpoint,
        base_url=base_url,
        direct_url=direct_url,
        model=model,
        rows=warmup,
        concurrency=len(warmup),
        deadline_s=deadline_s,
    )
    vllm_before = parse_prometheus(await _snapshot(f"{direct_url.rstrip('/')}/metrics"))
    gateway_before = (
        parse_prometheus(await _snapshot(f"{gateway_url.rstrip('/')}/metrics"))
        if endpoint == "gateway"
        else {}
    )
    vram = VramSampler(enabled=True)
    load = SystemLoadSampler(enabled=True)
    gateway_sampler = PrometheusSampler(
        f"{gateway_url.rstrip('/')}/metrics" if endpoint == "gateway" else None
    )
    monitor_tasks = [
        asyncio.create_task(vram.run()),
        asyncio.create_task(load.run()),
        asyncio.create_task(gateway_sampler.run()),
    ]
    origin = time.perf_counter()
    burst_end_at = origin + offsets[-1]
    try:
        burst_task = asyncio.create_task(
            _run_requests(
                endpoint=endpoint,
                base_url=base_url,
                model=model,
                rows=rows,
                cell_id=cell_id,
                offsets=offsets,
                concurrency=256,
                deadline_s=deadline_s,
            )
        )
        recovery_task = asyncio.create_task(
            _recovery_probes(
                endpoint=endpoint,
                base_url=base_url,
                model=model,
                row=warmup[0],
                cell_id=cell_id,
                burst_end_at=burst_end_at,
                deadline_s=deadline_s,
                threshold_s=recovery_threshold_s,
                interval_s=recovery_interval_s,
                required=recovery_required,
            )
        )
        (observations, peak, elapsed), (recovery_s, probes) = await asyncio.gather(
            burst_task, recovery_task
        )
    finally:
        vram.stop()
        load.stop()
        gateway_sampler.stop()
        await asyncio.gather(*monitor_tasks)
    vllm_after = parse_prometheus(await _snapshot(f"{direct_url.rstrip('/')}/metrics"))
    gateway_after = (
        parse_prometheus(await _snapshot(f"{gateway_url.rstrip('/')}/metrics"))
        if endpoint == "gateway"
        else {}
    )
    await _wait_vllm_idle(direct_url)
    cell = _summarize(
        observations,
        elapsed_s=elapsed,
        offered_qps=offered_qps,
        concurrency=256,
        peak_concurrency=peak,
        hourly_usd=hourly_usd,
        vllm_metrics=summarize_vllm_metrics(prometheus_delta(vllm_before, vllm_after)),
        gateway_metrics=(
            _gateway_metrics(gateway_before, gateway_after) if endpoint == "gateway" else None
        ),
        gateway_samples=gateway_sampler.summary(),
        vram=vram.summary(),
        co_tenancy=load.summary(),
        warmup_co_tenancy=warmup_load,
    )
    cell.update(
        {
            "cell_id": cell_id,
            "endpoint": endpoint,
            "request_schedule_sha256": _canonical_hash(
                [
                    {
                        "request_id": row["phase5_request_id"],
                        "input_sha256": _input_hash(model, row),
                        "offset_s": offset,
                    }
                    for row, offset in zip(rows, offsets, strict=True)
                ]
            ),
            "warmup_schedule_sha256": _canonical_hash([row["phase5_request_id"] for row in warmup]),
            "recovery_time_s": recovery_s,
            "recovery_threshold_s": recovery_threshold_s,
            "recovery_probe_count": len(probes),
            "recovery_probes": [asdict(item) for item in probes],
        }
    )
    return cell, [*observations, *probes]


async def _overload_stage(
    config: Mapping[str, Any],
    workload: Sequence[dict[str, Any]],
    capacity: Mapping[str, Any],
    *,
    direct_url: str,
    gateway_url: str,
) -> tuple[dict[str, Any], list[Observation]]:
    settings = config["overload"]
    capacity_qps = float(capacity["measured_capacity_qps"])
    baseline_p95 = float(capacity["baseline_success_p95_s"])
    recovery_threshold = baseline_p95 * float(settings["recovery_p95_multiplier"])
    model = str(config["model"]["served_name"])
    cells: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    observations: list[Observation] = []
    for index, multiplier_value in enumerate(settings["multipliers"]):
        multiplier = float(multiplier_value)
        qps = capacity_qps * multiplier
        seed = int(config["workload"]["request_seed"]) + 2000 + index
        rows = _select_rows(
            workload,
            config,
            profile=str(settings["length_profile"]),
            count=int(settings["measurement_requests"]),
            seed=seed,
        )
        warmup = _select_rows(
            workload,
            config,
            profile=str(settings["length_profile"]),
            count=int(config["workload"]["warmup_requests"]),
            seed=seed - 1,
        )
        offsets = _poisson_offsets(count=len(rows), qps=qps, seed=seed)
        order = ("direct", "gateway") if index % 2 == 0 else ("gateway", "direct")
        pair: dict[str, Any] = {
            "multiplier": multiplier,
            "offered_qps": qps,
            "execution_order": list(order),
        }
        for endpoint in order:
            cell, items = await _overload_cell(
                endpoint=endpoint,
                base_url=direct_url if endpoint == "direct" else gateway_url,
                direct_url=direct_url,
                gateway_url=gateway_url,
                model=model,
                rows=rows,
                warmup=warmup,
                cell_id=f"overload-{multiplier:g}x-{endpoint}",
                offsets=offsets,
                deadline_s=float(settings["deadline_s"]),
                offered_qps=qps,
                recovery_threshold_s=recovery_threshold,
                recovery_interval_s=float(settings["recovery_probe_interval_s"]),
                recovery_required=int(settings["recovery_consecutive_successes"]),
                hourly_usd=float(config["hardware"]["hourly_usd"]),
            )
            cells.append(cell)
            observations.extend(items)
            pair[endpoint] = cell
        if pair["direct"]["request_schedule_sha256"] != pair["gateway"]["request_schedule_sha256"]:
            raise RuntimeError("overload direct/gateway schedules are not identical")
        pairs.append(pair)
    return {
        "started_at": _now(),
        "cells": cells,
        "pairs": pairs,
        "measured_capacity_qps": capacity_qps,
        "recovery_definition": (
            "seconds after last scheduled arrival until "
            f"{settings['recovery_consecutive_successes']} consecutive probes complete within "
            f"{settings['recovery_p95_multiplier']}x low-load p95"
        ),
    }, observations


async def _identity(direct_url: str, gateway_url: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        version, models, gateway_health = await asyncio.gather(
            client.get(f"{direct_url.rstrip('/')}/version"),
            client.get(f"{direct_url.rstrip('/')}/v1/models"),
            client.get(f"{gateway_url.rstrip('/')}/healthz"),
        )
        version.raise_for_status()
        models.raise_for_status()
        gateway_health.raise_for_status()
        return {
            "vllm_version_endpoint": version.json(),
            "vllm_models_endpoint": models.json(),
            "gateway_health_status": gateway_health.status_code,
        }


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC)


def _optimization_comparison(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    left = before["cells"][0]
    right = after["cells"][0]
    if left["request_schedule_sha256"] != right["request_schedule_sha256"]:
        raise RuntimeError("profile before/after schedules differ")

    def pct(before_value: float | None, after_value: float | None) -> float | None:
        if before_value is None or after_value is None or before_value == 0:
            return None
        return (after_value - before_value) / before_value * 100

    return {
        "request_schedule_sha256": left["request_schedule_sha256"],
        "before_git_sha": before["git_sha"],
        "after_git_sha": after["git_sha"],
        "e2e_p50_before_s": left["client"]["e2e_success"]["p50_s"],
        "e2e_p50_after_s": right["client"]["e2e_success"]["p50_s"],
        "e2e_p50_delta_pct": pct(
            left["client"]["e2e_success"]["p50_s"],
            right["client"]["e2e_success"]["p50_s"],
        ),
        "e2e_p95_before_s": left["client"]["e2e_success"]["p95_s"],
        "e2e_p95_after_s": right["client"]["e2e_success"]["p95_s"],
        "e2e_p95_delta_pct": pct(
            left["client"]["e2e_success"]["p95_s"],
            right["client"]["e2e_success"]["p95_s"],
        ),
        "throughput_before_per_s": left["successful_request_throughput_per_s"],
        "throughput_after_per_s": right["successful_request_throughput_per_s"],
        "throughput_delta_pct": pct(
            left["successful_request_throughput_per_s"],
            right["successful_request_throughput_per_s"],
        ),
    }


def _write_final_receipt(
    config: Mapping[str, Any],
    workload_receipt: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    capacity: Mapping[str, Any],
    profile_before: Mapping[str, Any],
    profile_after: Mapping[str, Any],
    direct_gateway: Mapping[str, Any],
    overload: Mapping[str, Any],
) -> dict[str, Any]:
    started_value = os.environ.get("FORGE_POD_STARTED_AT")
    if not started_value:
        raise RuntimeError("FORGE_POD_STARTED_AT is required for the full GPU ledger")
    started_at = _parse_time(started_value)
    finished_at = datetime.now(UTC)
    gpu_hours = (finished_at - started_at).total_seconds() / 3600
    git_sha = benchmark_git_sha()
    baseline_git_sha = profile_before["git_sha"]
    optimized_git_sha = profile_after["git_sha"]
    if optimized_git_sha != git_sha:
        raise RuntimeError("profile-after SHA must match the final benchmark SHA")
    if not PROFILE_EVIDENCE.is_file():
        raise FileNotFoundError(
            f"profile-driven optimization evidence is missing: {PROFILE_EVIDENCE}"
        )
    optimization_evidence = json.loads(PROFILE_EVIDENCE.read_text())
    if optimization_evidence.get("status") != "complete":
        raise RuntimeError("profile-driven optimization evidence is incomplete")
    if optimization_evidence.get("baseline_git_sha") != baseline_git_sha:
        raise RuntimeError("optimization evidence baseline SHA conflicts with profile-before")
    if optimization_evidence.get("optimized_git_sha") != optimized_git_sha:
        raise RuntimeError("optimization evidence optimized SHA conflicts with profile-after")
    artifact = require_verified_artifact(load_phase4_config(config["workload"]["source_config"]))
    receipt = {
        "version": 1,
        "status": "complete",
        "phase": 5,
        "run_id": config["run_id"],
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "git_sha": git_sha,
        "baseline_git_sha": baseline_git_sha,
        "model": config["model"],
        "artifact_verification": artifact,
        "server": {
            "identity": identity,
            "declared": config["server"],
        },
        "gateway": config["gateway"],
        "hardware": {
            **host_disclosure(),
            "hourly_usd": float(config["hardware"]["hourly_usd"]),
            "rate_source": "FORGE_GPU_HOURLY_USD",
        },
        "workload": workload_receipt,
        "disclosure": {
            "precision": config["model"]["precision"],
            "training_time_quantization": config["model"]["training_time_quantization"],
            "deployment_quantization": config["model"]["deployment_quantization"],
            "load": {
                "direct_gateway": "closed-loop fixed concurrency",
                "capacity_and_overload": "Poisson arrivals with fixed seeds",
            },
            "arrival_rates": {
                "capacity_qps": config["capacity"]["arrival_rates_qps"],
                "overload_multipliers": config["overload"]["multipliers"],
            },
            "warmup": {
                "requests_per_cell": config["workload"]["warmup_requests"],
                "excluded_from_measurement": True,
            },
            "measurement_side": {
                "latency": "client monotonic wall clock around streamed OpenAI HTTP requests",
                "vllm": "server Prometheus counter/histogram deltas",
                "gateway": "gateway Prometheus counter deltas and gauge polling",
                "vram": "device-total nvidia-smi samples",
                "cpu_co_tenancy": "server-host getloadavg and /proc/stat CPU utilization",
            },
            "co_tenancy": (
                "The pod was shared with an unrelated CPU-only task. Every warm-up and measured "
                "cell sampled load average and CPU utilization; any load1 sample above half the "
                "logical core count contaminated and reran the entire stage."
            ),
            "fallback": (
                "Disabled because the contract colocates one R1b MTP vLLM replica; fallback share "
                "is reported as zero rather than pretending the same physical replica is "
                "independent."
            ),
            "verifier": "forge.verify.verifier.score after locked </think> normalization",
        },
        "metrics": {
            "capacity": capacity,
            "direct_gateway": direct_gateway,
            "overload": overload,
            "optimization": {
                **_optimization_comparison(profile_before, profile_after),
                "profile_before_artifact": profile_before["raw_artifact"],
                "profile_after_artifact": profile_after["raw_artifact"],
                "profile_evidence": {
                    **optimization_evidence,
                    "path": relative_path(PROFILE_EVIDENCE),
                    "sha256": sha256_file(PROFILE_EVIDENCE),
                },
            },
        },
        "cost": {
            "gpu_hours": gpu_hours,
            "hourly_usd": float(config["hardware"]["hourly_usd"]),
            "usd": gpu_hours * float(config["hardware"]["hourly_usd"]),
            "api_usd": 0.0,
            "scope": "authorized pod uptime from successful start through final receipt",
            "post_task_running_cost_excluded": True,
        },
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "raw_artifact": relative_path(FINAL_RECEIPT),
    }
    write_json_atomic(FINAL_RECEIPT, receipt)
    run_record = {
        "phase": 5,
        "run_id": config["run_id"],
        "status": "complete",
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "dataset_hash": workload_receipt["sha256"],
        "git_sha": git_sha,
        "baseline_git_sha": baseline_git_sha,
        "model": config["model"],
        "metrics": receipt["metrics"],
        "cost": receipt["cost"],
        "started_at": receipt["started_at"],
        "finished_at": receipt["finished_at"],
        "raw_artifact": receipt["raw_artifact"],
        "disclosure": receipt["disclosure"],
    }
    append_jsonl_once(RUNS_PATH, run_record, key="run_id")
    ledger_record = {
        "ledger_id": config["run_id"],
        "phase": 5,
        "operation": "remote_gateway_benchmark",
        "status": "complete",
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "git_sha": git_sha,
        "gpu_type": config["hardware"]["gpu_type"],
        "gpu_hours": gpu_hours,
        "hourly_usd": float(config["hardware"]["hourly_usd"]),
        "usd": gpu_hours * float(config["hardware"]["hourly_usd"]),
        "rate_source": "FORGE_GPU_HOURLY_USD supplied by the authorized remote launcher",
        "started_at": receipt["started_at"],
        "finished_at": receipt["finished_at"],
        "notes": (
            "Whole authorized pod uptime through receipt finalization; ongoing owner-requested "
            "post-task uptime is explicitly excluded."
        ),
    }
    append_jsonl_once(LEDGER_PATH, ledger_record, key="ledger_id")
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
    if stage == "capacity":
        return await _run_clean_stage(
            "capacity",
            config,
            lambda: _capacity_stage(
                config, workload, direct_url=direct_url, gateway_url=gateway_url
            ),
        )
    if stage in {"profile-before", "profile-after"}:
        return await _run_clean_stage(
            stage,
            config,
            lambda: _profile_stage(
                config, workload, direct_url=direct_url, gateway_url=gateway_url
            ),
        )
    if stage == "final":
        capacity_path, _ = _stage_paths("capacity")
        before_path, _ = _stage_paths("profile-before")
        after_path, _ = _stage_paths("profile-after")
        for path in (capacity_path, before_path, after_path):
            if not path.is_file():
                raise FileNotFoundError(f"required Phase 5 stage is missing: {path}")
        capacity = json.loads(capacity_path.read_text())
        profile_before = json.loads(before_path.read_text())
        profile_after = json.loads(after_path.read_text())
        direct_gateway = await _run_clean_stage(
            "direct-gateway",
            config,
            lambda: _direct_gateway_stage(
                config, workload, direct_url=direct_url, gateway_url=gateway_url
            ),
        )
        overload = await _run_clean_stage(
            "overload",
            config,
            lambda: _overload_stage(
                config,
                workload,
                capacity,
                direct_url=direct_url,
                gateway_url=gateway_url,
            ),
        )
        identity = await _identity(direct_url, gateway_url)
        return _write_final_receipt(
            config,
            workload_receipt,
            identity=identity,
            capacity=capacity,
            profile_before=profile_before,
            profile_after=profile_after,
            direct_gateway=direct_gateway,
            overload=overload,
        )
    raise ValueError(f"unsupported Phase 5 stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--direct-url", required=True)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument(
        "--stage",
        choices=("capacity", "profile-before", "profile-after", "final"),
        default="final",
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    sha = benchmark_git_sha()
    if _GIT_SHA.fullmatch(sha) is None:
        raise RuntimeError("Phase 5 requires a full lowercase Git SHA")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    workload, workload_receipt = _phase4_workload(config)
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
    print(json.dumps({"stage": args.stage, "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
