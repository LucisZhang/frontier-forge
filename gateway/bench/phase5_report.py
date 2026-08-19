#!/usr/bin/env python3
# ruff: noqa: E501 -- report templates keep Markdown rows and prose readable as literals.
"""Render the traceable Phase 5 gateway report and README result block."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import median
from typing import Any

import yaml

from forge.train.artifacts import write_json_atomic
from forge.train.config import REPO_ROOT, relative_path, sha256_file

RESULTS_DIR = REPO_ROOT / "results/phase5"
RAW_RECEIPT = RESULTS_DIR / "raw/phase5_gateway_bench.json"
REPORT_PATH = REPO_ROOT / "results/phase5_gateway_report.md"
MANIFEST_PATH = REPO_ROOT / "results/phase5_report_manifest.json"
README_PATH = REPO_ROOT / "gateway/README.md"
VERIFICATION_PATH = RESULTS_DIR / "verification.json"
START_MARKER = "<!-- PHASE5_BENCH_RESULTS_START -->"
END_MARKER = "<!-- PHASE5_BENCH_RESULTS_END -->"


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}{suffix}"


def _pct(value: Any) -> str:
    return _fmt(value, 1, "%")


def _ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 1000:.1f}"


def _median(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return median(present) if present else None


def _all_cells(receipt: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    metrics = receipt["metrics"]
    stages = [
        metrics["capacity"],
        metrics["direct_gateway"],
        metrics["overload"],
    ]
    cells: list[Mapping[str, Any]] = []
    for stage in stages:
        cells.extend(stage["cells"])
    return cells


def _co_tenancy_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    cells = _all_cells(receipt)
    samples = [cell["co_tenancy"] for cell in cells]
    warmups = [cell["warmup_co_tenancy"] for cell in cells]
    all_samples = [*samples, *warmups]
    return {
        "load1_max": max(float(item["load1_max"] or 0) for item in all_samples),
        "cpu_utilization_max": max(float(item["cpu_utilization_max"] or 0) for item in all_samples),
        "logical_cpu_count": all_samples[0]["logical_cpu_count"],
        "host_logical_cpu_count": all_samples[0]["host_logical_cpu_count"],
        "core_count_source": all_samples[0]["core_count_source"],
        "threshold": all_samples[0]["load1_contamination_threshold"],
        "contaminated_stage_attempts": sum(
            len(stage["contaminated_attempts"])
            for stage in (
                receipt["metrics"]["capacity"],
                receipt["metrics"]["direct_gateway"],
                receipt["metrics"]["overload"],
            )
        ),
    }


def _overhead_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    pairs = receipt["metrics"]["direct_gateway"]["pairs"]
    stable_pairs = [
        pair
        for pair in pairs
        if pair["direct"]["error_rate"] <= 0.05 and pair["gateway"]["error_rate"] <= 0.05
    ]
    return {
        "stable_pairs": len(stable_pairs),
        "p50_median_pct": _median(
            pair["overhead"]["e2e_p50_overhead_pct"] for pair in stable_pairs
        ),
        "p95_median_pct": _median(
            pair["overhead"]["e2e_p95_overhead_pct"] for pair in stable_pairs
        ),
        "ttft_p50_median_pct": _median(
            pair["overhead"]["ttft_p50_overhead_pct"] for pair in stable_pairs
        ),
        "throughput_median_pct": _median(
            pair["overhead"]["throughput_delta_pct"] for pair in stable_pairs
        ),
    }


def _error_semantics(cell: Mapping[str, Any]) -> str:
    statuses = ", ".join(f"{key}:{value}" for key, value in cell["http_status_counts"].items())
    codes = cell["error_semantics"]["error_codes"]
    rendered_codes = ", ".join(f"{key}:{value}" for key, value in codes.items()) or "none"
    return f"status[{statuses}] code[{rendered_codes}]"


def _resume_claim(receipt: Mapping[str, Any], overhead: Mapping[str, Any]) -> str:
    overload_pair = receipt["metrics"]["overload"]["pairs"][-1]
    gateway = overload_pair["gateway"]
    direct = overload_pair["direct"]
    multiplier = overload_pair["multiplier"]
    queue_max = gateway["gateway_samples"]["queue_depth_max"]
    reject_p50 = gateway["client"]["fast_reject"]["p50_s"]
    return (
        "在单卡 RTX 4090 上为 R1b BF16 + 原生 MTP vLLM 实现 C++20 token-aware "
        f"admission gateway：稳定单元格端到端 p50 中位开销 {_pct(overhead['p50_median_pct'])}，"
        f"{_fmt(multiplier, 0)}× 过载时队列峰值 {_fmt(queue_max, 0)}、503 快速拒绝 p50 "
        f"{_ms(reject_p50)} ms，恢复 {_fmt(gateway['recovery_time_s'])} s（裸 vLLM "
        f"{_fmt(direct['recovery_time_s'])} s）。"
    )


def _direct_gateway_table(receipt: Mapping[str, Any]) -> str:
    lines = [
        "| 长度分布 | 并发 | 端点 | TTFT p50/p95 ms | ITL p50/p95 ms | E2E p50/p95 s | req/s | tok/s | 成功任务成本 $/1k | 错误率 | VRAM peak MiB |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in receipt["metrics"]["direct_gateway"]["pairs"]:
        for endpoint in ("direct", "gateway"):
            cell = pair[endpoint]
            client = cell["client"]
            lines.append(
                "| {profile} | {concurrency} | {endpoint} | {ttft50}/{ttft95} | "
                "{itl50}/{itl95} | {e2e50}/{e2e95} | {rps} | {tps} | {cost} | "
                "{errors} | {vram} |".format(
                    profile=pair["length_profile"],
                    concurrency=pair["concurrency"],
                    endpoint=endpoint,
                    ttft50=_ms(client["ttft"]["p50_s"]),
                    ttft95=_ms(client["ttft"]["p95_s"]),
                    itl50=_ms(client["itl"]["p50_s"]),
                    itl95=_ms(client["itl"]["p95_s"]),
                    e2e50=_fmt(client["e2e_success"]["p50_s"]),
                    e2e95=_fmt(client["e2e_success"]["p95_s"]),
                    rps=_fmt(cell["successful_request_throughput_per_s"]),
                    tps=_fmt(cell["output_tokens_per_s"], 1),
                    cost=_fmt(cell["cost_per_1k_successful_tasks_usd"], 4),
                    errors=_pct(cell["error_rate"] * 100),
                    vram=_fmt(cell["vram"]["peak_mib"], 0),
                )
            )
        lines.append(
            "|  |  | paired overhead |  |  | p50 {p50}, p95 {p95} | throughput {throughput} |  |  |  |  |".format(
                p50=_pct(pair["overhead"]["e2e_p50_overhead_pct"]),
                p95=_pct(pair["overhead"]["e2e_p95_overhead_pct"]),
                throughput=_pct(pair["overhead"]["throughput_delta_pct"]),
            )
        )
    return "\n".join(lines)


def _capacity_table(receipt: Mapping[str, Any]) -> str:
    lines = [
        "| offered QPS | achieved ratio | p95 E2E s | peak concurrency | error | deadline miss | stable | load1 max | CPU max |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for cell in receipt["metrics"]["capacity"]["cells"]:
        lines.append(
            "| {qps} | {ratio} | {p95} | {concurrency} | {error} | {deadline} | {stable} | {load} | {cpu} |".format(
                qps=_fmt(cell["arrival_rate_qps"]),
                ratio=_fmt(cell["achieved_qps_ratio"]),
                p95=_fmt(cell["client"]["e2e_success"]["p95_s"]),
                concurrency=cell["max_observed_in_flight"],
                error=_pct(cell["error_rate"] * 100),
                deadline=_pct(cell["deadline_miss_rate"] * 100),
                stable="yes" if cell["stable"] else "no",
                load=_fmt(cell["co_tenancy"]["load1_max"], 2),
                cpu=_pct(cell["co_tenancy"]["cpu_utilization_max"] * 100),
            )
        )
    return "\n".join(lines)


def _overload_table(receipt: Mapping[str, Any]) -> str:
    lines = [
        "| 倍数 | 端点 | offered QPS | E2E all p95 s | 成功 p95 s | error | fast-reject p50 ms | queue max | fallback | recovery s | 错误语义 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for pair in receipt["metrics"]["overload"]["pairs"]:
        for endpoint in ("direct", "gateway"):
            cell = pair[endpoint]
            gateway = cell.get("gateway") or {}
            lines.append(
                "| {multiplier}× | {endpoint} | {qps} | {all_p95} | {success_p95} | "
                "{error} | {reject} | {queue} | {fallback} | {recovery} | {semantics} |".format(
                    multiplier=_fmt(pair["multiplier"], 0),
                    endpoint=endpoint,
                    qps=_fmt(pair["offered_qps"]),
                    all_p95=_fmt(cell["client"]["e2e_all_responses"]["p95_s"]),
                    success_p95=_fmt(cell["client"]["e2e_success"]["p95_s"]),
                    error=_pct(cell["error_rate"] * 100),
                    reject=_ms(cell["client"]["fast_reject"]["p50_s"]),
                    queue=_fmt(cell["gateway_samples"]["queue_depth_max"], 0),
                    fallback=_pct(gateway.get("fallback_share", 0) * 100),
                    recovery=_fmt(cell["recovery_time_s"]),
                    semantics=_error_semantics(cell),
                )
            )
    return "\n".join(lines)


def _gate_lines(receipt: Mapping[str, Any], verification: Mapping[str, Any]) -> list[str]:
    overload_cells = [pair["gateway"] for pair in receipt["metrics"]["overload"]["pairs"]]
    bounded = all(
        cell["gateway_samples"]["queue_depth_max"] is not None
        and cell["gateway_samples"]["queue_depth_max"] <= receipt["gateway"]["max_queue_requests"]
        for cell in overload_cells
    )
    fast_failure = any(
        cell["error_semantics"]["fast_rejects_under_1s"] > 0 for cell in overload_cells
    )
    return [
        f"- [{'x' if verification.get('asan_ubsan') == 'green' else ' '}] ASan/UBSan green",
        f"- [{'x' if verification.get('tsan') == 'green' else ' '}] TSan green",
        f"- [{'x' if verification.get('failure_injection') == 'green' else ' '}] failure-injection suite green on mock upstream",
        f"- [{'x' if bounded and fast_failure else ' '}] overload = bounded queue + fast failure, never unbounded growth",
        "- [x] direct-vs-gateway overhead quantified",
        "- [x] one profile-driven optimization documented with matched before/after requests",
        "- [x] resume-claim sentence drafted from measured numbers",
    ]


def _report(
    receipt: Mapping[str, Any],
    config: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> tuple[str, str]:
    overhead = _overhead_summary(receipt)
    co_tenancy = _co_tenancy_summary(receipt)
    optimization = receipt["metrics"]["optimization"]
    evidence = optimization["profile_evidence"]
    claim = _resume_claim(receipt, overhead)
    gates = "\n".join(_gate_lines(receipt, verification))
    report = f"""# Phase 5 gateway remote benchmark report

Status: **complete**. Run `{receipt["run_id"]}` on git `{receipt["git_sha"]}`; baseline gateway git `{receipt["baseline_git_sha"]}`. Raw receipt: `{receipt["raw_artifact"]}`.

## Result

- Bare-vLLM measured capacity: **{_fmt(receipt["metrics"]["capacity"]["measured_capacity_qps"])} QPS**; max stable observed concurrency **{receipt["metrics"]["capacity"]["max_stable_concurrency"]}**.
- Across {overhead["stable_pairs"]} stable direct/gateway cells, median gateway E2E overhead was **p50 {_pct(overhead["p50_median_pct"])} / p95 {_pct(overhead["p95_median_pct"])}**; median throughput delta **{_pct(overhead["throughput_median_pct"])}**.
- Profile-driven optimization: `{evidence["optimization"]["summary"]}`. Matched profile cell E2E p50 changed **{_fmt(optimization["e2e_p50_before_s"])} → {_fmt(optimization["e2e_p50_after_s"])} s ({_pct(optimization["e2e_p50_delta_pct"])})**; throughput **{_fmt(optimization["throughput_before_per_s"])} → {_fmt(optimization["throughput_after_per_s"])} req/s ({_pct(optimization["throughput_delta_pct"])})**.

## Resume claim draft

> {claim}

## Capacity calibration

{_capacity_table(receipt)}

Capacity is the highest offered Poisson QPS that passes the pinned error, deadline, achieved-QPS, and p95-inflation gates.

## Direct vs gateway: concurrency × length distribution

{_direct_gateway_table(receipt)}

Every direct/gateway pair used identical serialized request hashes, offsets, warm-up count, model artifact, precision, and hardware. Pair execution order alternated to reduce ordering bias.

## Overload: 2× / 3× / 5× capacity

{_overload_table(receipt)}

Fallback was deliberately disabled: this run had one physical R1b MTP vLLM replica, so routing the same backend through a second logical pool would fabricate independent fallback capacity. Fallback share is therefore honestly reported as zero.

## Profile-driven optimization

- Profiler: `{evidence["profiler"]}`; baseline artifact `{evidence["baseline_profile"]["path"]}` (`{evidence["baseline_profile"]["sha256"]}`).
- Largest measured gateway-side cost: `{evidence["largest_gateway_cost"]["summary"]}`.
- Change: `{evidence["optimization"]["summary"]}`.
- Matched request schedule: `{optimization["request_schedule_sha256"]}`.
- E2E p95: {_fmt(optimization["e2e_p95_before_s"])} → {_fmt(optimization["e2e_p95_after_s"])} s ({_pct(optimization["e2e_p95_delta_pct"])}).

## Disclosure

- Hardware: `{receipt["hardware"]["gpu"]}`, {receipt["hardware"]["gpu_memory_total_mib"]} MiB VRAM, {receipt["hardware"]["logical_cpu_count"]} logical CPUs, driver `{receipt["hardware"]["driver_version"]}`.
- Model: R1b MTP-preserving export `{receipt["model"]["artifact_sha256"]}`; BF16 deployment, no deployment quantization. Training used QLoRA NF4; these are separate facts.
- Server: vLLM `{config["server"]["vllm_version"]}`, native MTP speculative decoding with {config["server"]["num_speculative_tokens"]} speculative token, max model length {config["server"]["max_model_len"]}, max sequences {config["server"]["max_num_seqs"]}.
- Load: closed-loop fixed concurrency for overhead cells; fixed-seed Poisson arrivals for capacity and overload. Warm-up: {config["workload"]["warmup_requests"]} requests per cell, excluded.
- Measurement side: client monotonic streaming latency; vLLM and gateway Prometheus deltas; device-wide `nvidia-smi` VRAM; server-host load average and `/proc/stat` CPU utilization.
- Co-tenancy: {receipt["disclosure"]["co_tenancy"]} Maximum sampled load1 **{_fmt(co_tenancy["load1_max"], 2)}** vs contamination threshold **{_fmt(co_tenancy["threshold"], 1)}** ({co_tenancy["logical_cpu_count"]} effective CPUs from `{co_tenancy["core_count_source"]}`; host exposes {co_tenancy["host_logical_cpu_count"]} logical CPUs); max host CPU utilization **{_pct(co_tenancy["cpu_utilization_max"] * 100)}**. Contaminated stage attempts retained/rerun: **{co_tenancy["contaminated_stage_attempts"]}**.
- Cost: `${receipt["cost"]["hourly_usd"]:.2f}/GPU-hour`; accounted task uptime {_fmt(receipt["cost"]["gpu_hours"], 4)} h = `${receipt["cost"]["usd"]:.4f}`. Owner-requested post-task running time is not yet in this closed receipt.
- Successful-task cost uses verifier successes, never raw token counts.

## Reproduction

```sh
FORGE_GPU_HOURLY_USD=0.30 FORGE_BENCH_GIT_SHA=<sha> \\
  DIRECT_URL=http://127.0.0.1:8000 GATEWAY_URL=http://127.0.0.1:9000 \\
  make gateway-bench STAGE=<capacity|profile-before|profile-after|final>
make gateway-bench-report
```

## Phase 5 gate

{gates}
"""
    readme_block = f"""{START_MARKER}
## Remote benchmark result

- R1b BF16 + native-MTP vLLM capacity: **{_fmt(receipt["metrics"]["capacity"]["measured_capacity_qps"])} QPS**.
- Stable-cell gateway E2E overhead: median **p50 {_pct(overhead["p50_median_pct"])}, p95 {_pct(overhead["p95_median_pct"])}**.
- Profiled optimization: E2E p50 **{_fmt(optimization["e2e_p50_before_s"])} → {_fmt(optimization["e2e_p50_after_s"])} s**; throughput **{_fmt(optimization["throughput_before_per_s"])} → {_fmt(optimization["throughput_after_per_s"])} req/s**.
- Full methodology, overload semantics, disclosure, raw-artifact pointers, and gate checklist: [`results/phase5_gateway_report.md`](../results/phase5_gateway_report.md).

Resume claim draft:

> {claim}
{END_MARKER}"""
    return report, readme_block


def _replace_readme_block(readme: str, block: str) -> str:
    if readme.count(START_MARKER) != 1 or readme.count(END_MARKER) != 1:
        raise RuntimeError("gateway README must contain exactly one Phase 5 result marker pair")
    start = readme.index(START_MARKER)
    end = readme.index(END_MARKER) + len(END_MARKER)
    return f"{readme[:start]}{block}{readme[end:]}"


def run(config_path: str | Path) -> dict[str, Any]:
    if not RAW_RECEIPT.is_file():
        raise FileNotFoundError(f"Phase 5 final receipt is missing: {RAW_RECEIPT}")
    receipt = json.loads(RAW_RECEIPT.read_text())
    config = yaml.safe_load((REPO_ROOT / config_path).read_text())
    if receipt.get("status") != "complete" or receipt.get("phase") != 5:
        raise RuntimeError("Phase 5 final receipt is incomplete")
    verification = json.loads(VERIFICATION_PATH.read_text()) if VERIFICATION_PATH.is_file() else {}
    report, readme_block = _report(receipt, config, verification)
    REPORT_PATH.write_text(report)
    README_PATH.write_text(_replace_readme_block(README_PATH.read_text(), readme_block))
    manifest = {
        "version": 1,
        "status": "complete",
        "phase": 5,
        "run_id": receipt["run_id"],
        "git_sha": receipt["git_sha"],
        "generated_at": receipt["finished_at"],
        "inputs": {
            "raw_receipt": {
                "path": relative_path(RAW_RECEIPT),
                "sha256": sha256_file(RAW_RECEIPT),
            },
            "verification": (
                {
                    "path": relative_path(VERIFICATION_PATH),
                    "sha256": sha256_file(VERIFICATION_PATH),
                }
                if VERIFICATION_PATH.is_file()
                else None
            ),
        },
        "outputs": {
            "report": {
                "path": relative_path(REPORT_PATH),
                "sha256": sha256_file(REPORT_PATH),
            },
            "gateway_readme": {
                "path": relative_path(README_PATH),
                "sha256": sha256_file(README_PATH),
            },
        },
    }
    write_json_atomic(MANIFEST_PATH, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    manifest = run(args.config)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
