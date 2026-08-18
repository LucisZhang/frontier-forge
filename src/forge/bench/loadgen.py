"""Workload-controlled OpenAI-compatible streaming load generator for Phase 4."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from forge.teacher.filters import breakdown_dict
from forge.verify.verifier import score

from .config import load_phase4_config
from .metrics import (
    parse_prometheus,
    prometheus_delta,
    summarize_latencies,
    summarize_vllm_metrics,
)
from .workload import load_workload


@dataclass
class RequestObservation:
    request_id: str
    complaint_id: int
    offered_qps: float
    concurrency_cap: int
    scheduled_offset_s: float
    dispatch_delay_s: float
    client_ttft_s: float | None
    client_e2e_s: float
    client_mean_itl_s: float | None
    output_tokens: int
    stream_events: int
    finish_reason: str | None
    http_status: int | None
    error: str | None
    deadline_missed: bool
    output: str
    verifier: dict[str, Any] | None
    response_headers: dict[str, str]


class PeakConcurrency:
    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self._lock = asyncio.Lock()

    async def enter(self) -> None:
        async with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    async def leave(self) -> None:
        async with self._lock:
            self.current -= 1


class VramSampler:
    """Sample device-wide VRAM on the benchmark host via read-only nvidia-smi."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.values: list[float] = []
        self._stop = asyncio.Event()

    async def run(self) -> None:
        if not self.enabled:
            return
        while not self._stop.is_set():
            try:
                process = await asyncio.create_subprocess_exec(
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await process.communicate()
                if process.returncode == 0:
                    self.values.extend(
                        float(line.strip()) for line in stdout.decode().splitlines() if line.strip()
                    )
            except (FileNotFoundError, ValueError):
                self.enabled = False
                return
            with suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=0.2)

    def stop(self) -> None:
        self._stop.set()

    def summary(self) -> dict[str, Any]:
        return {
            "measurement_side": "host_nvidia_smi_device_total",
            "samples": len(self.values),
            "peak_mib": max(self.values) if self.values else None,
            "mean_mib": sum(self.values) / len(self.values) if self.values else None,
        }


def _sse_content(chunk: Mapping[str, Any]) -> tuple[str, str | None]:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", None
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return "", None
    delta = choice.get("delta")
    content = ""
    if isinstance(delta, Mapping):
        value = delta.get("content")
        if isinstance(value, str):
            content = value
    reason = choice.get("finish_reason")
    return content, reason if isinstance(reason, str) else None


async def _stream_one(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    row: Mapping[str, Any],
    offered_qps: float,
    concurrency_cap: int,
    scheduled_at: float,
    schedule_origin: float,
    deadline_s: float,
    semaphore: asyncio.Semaphore,
    peak: PeakConcurrency,
    extra_payload: Mapping[str, Any] | None,
) -> RequestObservation:
    target_delay = max(0.0, scheduled_at - time.perf_counter())
    if target_delay:
        await asyncio.sleep(target_delay)
    await semaphore.acquire()
    await peak.enter()
    sent_at = time.perf_counter()
    first_event_at: float | None = None
    event_times: list[float] = []
    output_parts: list[str] = []
    usage_tokens: int | None = None
    finish_reason: str | None = None
    status: int | None = None
    error: str | None = None
    headers: dict[str, str] = {}
    payload: dict[str, Any] = {
        "model": model,
        "messages": row["messages"],
        "temperature": 0,
        "max_tokens": int(row["max_tokens"]),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra_payload:
        payload.update(extra_payload)
    try:
        async with asyncio.timeout(deadline_s):
            async with client.stream(
                "POST", f"{base_url.rstrip('/')}/v1/chat/completions", json=payload
            ) as response:
                status = response.status_code
                headers = {
                    key.lower(): value
                    for key, value in response.headers.items()
                    if key.lower().startswith(("x-", "server-timing"))
                }
                if response.is_error:
                    body = (await response.aread()).decode(errors="replace")[:1000]
                    raise RuntimeError(f"HTTP {status}: {body}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"invalid SSE JSON: {exc}") from exc
                    usage = chunk.get("usage")
                    if isinstance(usage, Mapping) and isinstance(
                        usage.get("completion_tokens"), int
                    ):
                        usage_tokens = int(usage["completion_tokens"])
                    content, reason = _sse_content(chunk)
                    if reason:
                        finish_reason = reason
                    if content:
                        now = time.perf_counter()
                        if first_event_at is None:
                            first_event_at = now
                        event_times.append(now)
                        output_parts.append(content)
    except TimeoutError:
        error = f"deadline_exceeded:{deadline_s}s"
    except (httpx.HTTPError, RuntimeError) as exc:
        error = f"{type(exc).__name__}:{exc}"
    finally:
        completed_at = time.perf_counter()
        await peak.leave()
        semaphore.release()

    output = "".join(output_parts).strip()
    output_tokens = usage_tokens if usage_tokens is not None else len(event_times)
    e2e = completed_at - sent_at
    ttft = first_event_at - sent_at if first_event_at is not None else None
    itl = None
    if ttft is not None and output_tokens > 1:
        itl = max(0.0, e2e - ttft) / (output_tokens - 1)
    scored = breakdown_dict(score({"label": row["label"]}, output)) if error is None else None
    return RequestObservation(
        request_id=str(row["request_id"]),
        complaint_id=int(row["complaint_id"]),
        offered_qps=offered_qps,
        concurrency_cap=concurrency_cap,
        scheduled_offset_s=scheduled_at - schedule_origin,
        dispatch_delay_s=max(0.0, sent_at - scheduled_at),
        client_ttft_s=ttft,
        client_e2e_s=e2e,
        client_mean_itl_s=itl,
        output_tokens=output_tokens,
        stream_events=len(event_times),
        finish_reason=finish_reason,
        http_status=status,
        error=error,
        deadline_missed=error is not None and error.startswith("deadline_exceeded"),
        output=output,
        verifier=scored,
        response_headers=headers,
    )


def poisson_offsets(*, count: int, qps: float, seed: int) -> list[float]:
    generator = random.Random(seed)
    offsets: list[float] = []
    current = 0.0
    for index in range(count):
        if index:
            current += generator.expovariate(qps)
        offsets.append(current)
    return offsets


def _scheduled_rows(
    workload: Sequence[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    if not workload:
        raise ValueError("workload is empty")
    order = list(workload)
    random.Random(seed).shuffle(order)
    return [order[index % len(order)] for index in range(count)]


async def _metrics_snapshot(client: httpx.AsyncClient, base_url: str) -> str:
    try:
        response = await client.get(f"{base_url.rstrip('/')}/metrics", timeout=10)
        response.raise_for_status()
        return response.text
    except httpx.HTTPError:
        return ""


async def _run_arrivals(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    rows: list[dict[str, Any]],
    qps: float,
    max_concurrency: int,
    deadline_s: float,
    seed: int,
    extra_payload: Mapping[str, Any] | None,
) -> tuple[list[RequestObservation], int, float]:
    offsets = poisson_offsets(count=len(rows), qps=qps, seed=seed)
    origin = time.perf_counter()
    semaphore = asyncio.Semaphore(max_concurrency)
    peak = PeakConcurrency()
    tasks = [
        asyncio.create_task(
            _stream_one(
                client,
                base_url=base_url,
                model=model,
                row=row,
                offered_qps=qps,
                concurrency_cap=max_concurrency,
                scheduled_at=origin + offset,
                schedule_origin=origin,
                deadline_s=deadline_s,
                semaphore=semaphore,
                peak=peak,
                extra_payload=extra_payload,
            )
        )
        for row, offset in zip(rows, offsets, strict=True)
    ]
    observations = await asyncio.gather(*tasks)
    elapsed = max(time.perf_counter() - origin, 1e-9)
    return observations, peak.peak, elapsed


def summarize_point(
    observations: list[RequestObservation],
    *,
    qps: float,
    elapsed_s: float,
    peak_concurrency: int,
    hourly_usd: float,
    server_metrics: dict[str, Any],
    vram: dict[str, Any],
) -> dict[str, Any]:
    completed = [item for item in observations if item.error is None]
    successes = [
        item
        for item in completed
        if item.verifier is not None and bool(item.verifier.get("task_success"))
    ]
    request_throughput = len(completed) / elapsed_s
    successful_throughput = len(successes) / elapsed_s
    output_tokens = sum(item.output_tokens for item in completed)
    client = {
        "measurement_side": "client_wall_clock_streaming",
        "ttft": summarize_latencies(
            [item.client_ttft_s for item in completed if item.client_ttft_s is not None]
        ),
        "itl": summarize_latencies(
            [item.client_mean_itl_s for item in completed if item.client_mean_itl_s is not None]
        ),
        "e2e": summarize_latencies([item.client_e2e_s for item in completed]),
        "dispatch_delay": summarize_latencies([item.dispatch_delay_s for item in observations]),
        "elapsed_s": elapsed_s,
        "request_throughput_per_s": request_throughput,
        "successful_task_throughput_per_s": successful_throughput,
        "output_tokens_per_s": output_tokens / elapsed_s,
    }
    return {
        "arrival_rate_qps": qps,
        "arrival_process": "poisson_fixed_seed",
        "concurrency_cap": observations[0].concurrency_cap if observations else 0,
        "max_observed_in_flight": peak_concurrency,
        "requests": len(observations),
        "completed": len(completed),
        "errors": len(observations) - len(completed),
        "deadline_misses": sum(item.deadline_missed for item in observations),
        "verifier_successes": len(successes),
        "verifier_task_success_rate": len(successes) / len(completed) if completed else 0.0,
        "error_rate": (len(observations) - len(completed)) / len(observations),
        "deadline_miss_rate": sum(item.deadline_missed for item in observations)
        / len(observations),
        "achieved_qps_ratio": request_throughput / qps,
        "client": client,
        "server": server_metrics,
        "vram": vram,
        "cost_per_1k_successful_tasks_usd": (
            hourly_usd * 1000 / (successful_throughput * 3600)
            if successful_throughput > 0
            else None
        ),
        "cost_formula": "hourly_usd*1000/(verifier_successes_per_second*3600)",
        "stable": None,
    }


def classify_stability(points: list[dict[str, Any]], stability: Mapping[str, Any]) -> None:
    baseline_p95 = next(
        (
            point["client"]["e2e"]["p95_s"]
            for point in points
            if point["client"]["e2e"]["p95_s"] is not None
        ),
        None,
    )
    for point in points:
        p95 = point["client"]["e2e"]["p95_s"]
        p95_ok = (
            baseline_p95 is not None
            and p95 is not None
            and p95 <= baseline_p95 * float(stability["max_p95_inflation"])
        )
        checks = {
            "error_rate": point["error_rate"] <= float(stability["max_error_rate"]),
            "deadline_miss_rate": point["deadline_miss_rate"]
            <= float(stability["max_deadline_miss_rate"]),
            "achieved_qps_ratio": point["achieved_qps_ratio"]
            >= float(stability["min_achieved_qps_ratio"]),
            "p95_inflation": p95_ok,
        }
        point["stability_checks"] = checks
        point["stable"] = all(checks.values())


async def run_load_benchmark(
    config: Mapping[str, Any],
    *,
    base_url: str,
    smoke: bool,
    extra_payload: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workload = load_workload(config, smoke=smoke)
    settings = config["workload"]
    measurement_count = (
        min(4, int(settings["measurement_requests"]))
        if smoke
        else int(settings["measurement_requests"])
    )
    warmup_count = (
        min(2, int(settings["warmup_requests"])) if smoke else int(settings["warmup_requests"])
    )
    rates = [50.0] if smoke else [float(item) for item in settings["arrival_rates_qps"]]
    timeout = httpx.Timeout(None)
    limits = httpx.Limits(
        max_connections=int(settings["max_concurrency"]) + 4,
        max_keepalive_connections=int(settings["max_concurrency"]),
    )
    async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
        health = await client.get(f"{base_url.rstrip('/')}/health", timeout=10)
        health.raise_for_status()
        warmup_rows = _scheduled_rows(
            workload, count=warmup_count, seed=int(settings["request_seed"]) - 1
        )
        await _run_arrivals(
            client,
            base_url=base_url,
            model=str(config["model"]["served_name"]),
            rows=warmup_rows,
            qps=max(rates),
            max_concurrency=int(settings["max_concurrency"]),
            deadline_s=float(settings["deadline_s"]),
            seed=int(settings["request_seed"]) - 1,
            extra_payload=extra_payload,
        )
        points: list[dict[str, Any]] = []
        request_records: list[dict[str, Any]] = []
        for index, qps in enumerate(rates):
            rows = _scheduled_rows(
                workload,
                count=measurement_count,
                seed=int(settings["request_seed"]) + index,
            )
            before = parse_prometheus(await _metrics_snapshot(client, base_url))
            vram = VramSampler(enabled=not smoke)
            sampler_task = asyncio.create_task(vram.run())
            observations, peak_concurrency, elapsed = await _run_arrivals(
                client,
                base_url=base_url,
                model=str(config["model"]["served_name"]),
                rows=rows,
                qps=qps,
                max_concurrency=int(settings["max_concurrency"]),
                deadline_s=float(settings["deadline_s"]),
                seed=int(settings["request_seed"]) + index,
                extra_payload=extra_payload,
            )
            vram.stop()
            await sampler_task
            after = parse_prometheus(await _metrics_snapshot(client, base_url))
            server = summarize_vllm_metrics(prometheus_delta(before, after))
            point = summarize_point(
                observations,
                qps=qps,
                elapsed_s=elapsed,
                peak_concurrency=peak_concurrency,
                hourly_usd=float(config["hardware"]["hourly_usd"]),
                server_metrics=server,
                vram=vram.summary(),
            )
            point["point_index"] = index
            points.append(point)
            request_records.extend(
                {"point_index": index, **asdict(observation)} for observation in observations
            )
        classify_stability(points, config["stability"])
        return points, request_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_phase4_config(args.config)
    if config["experiment"] not in {"serve", "spec_decode"}:
        raise ValueError("loadgen supports serve and spec_decode configs")
    points, requests = asyncio.run(
        run_load_benchmark(config, base_url=args.base_url, smoke=args.smoke)
    )
    print(json.dumps({"points": points, "requests": len(requests)}, sort_keys=True))


if __name__ == "__main__":
    main()
