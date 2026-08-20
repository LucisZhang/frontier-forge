#!/usr/bin/env python3
# ruff: noqa: E501 -- Evidence tables and audit sentences stay readable as literals.
"""Render the amended Gate 7.1 report and lift the README block only after a pass."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any

from forge.train.artifacts import write_json_atomic
from forge.train.config import REPO_ROOT, relative_path, sha256_file
from gateway.bench import phase7_1_report as finite_report

RESULTS_DIR = REPO_ROOT / "results/phase7_1"
SUSTAINED_RECEIPT = RESULTS_DIR / "raw/phase7_1_sustained_gateway_bench.json"
FINITE_RECEIPT = RESULTS_DIR / "raw/phase7_1_gateway_bench.json"
LOCAL_VERIFICATION = RESULTS_DIR / "local_verification.json"
PHASE5_RECEIPT = REPO_ROOT / "results/phase5/raw/phase5_gateway_bench.json"
REPORT_PATH = REPO_ROOT / "results/phase7_1_sustained_a10_report.md"
MANIFEST_PATH = RESULTS_DIR / "sustained_report_manifest.json"
README_PATH = REPO_ROOT / "README.md"
LEGACY_SECTION_START = "## Production story: the gateway result has a red flag"
RESOLVED_SECTION_START = "## Production story: sustained overload resolves the gateway red flag"
SECTION_END = "## Reproduce the headline"


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.1f}%"


def _count_status(cell: Mapping[str, Any], status: int) -> int:
    return int(cell.get("http_status_counts", {}).get(str(status), 0))


def _five_xx(cell: Mapping[str, Any]) -> int:
    return sum(
        int(count)
        for status, count in cell.get("http_status_counts", {}).items()
        if 500 <= int(status) < 600
    )


def _sustained_table(receipt: Mapping[str, Any]) -> str:
    lines = [
        "| load | arrival windows bare/gateway | requests bare/gateway | bare/gateway upstream 5xx | gateway 429 | queue sampled/process | 429 p95 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    gates = {float(row["multiplier"]): row for row in receipt["gate"]["sustained_overload_cells"]}
    for pair in receipt["metrics"]["sustained_overload"]["pairs"]:
        row = gates[float(pair["multiplier"])]
        direct = pair["direct"]
        gateway = pair["gateway"]
        fast_p95 = gateway["client"]["fast_reject"]["p95_s"]
        lines.append(
            "| {multiplier:g}× | {direct_duration:.1f}s / {gateway_duration:.1f}s | {direct_requests} / {gateway_requests} | {direct_5xx}/{direct_requests} ({direct_rate}) / {gateway_5xx}/{admitted} ({gateway_rate}) | {rejects}/{gateway_requests} | {sampled}/{process} | {latency} ms |".format(
                multiplier=float(pair["multiplier"]),
                direct_duration=float(direct["arrival_duration_s"]),
                gateway_duration=float(gateway["arrival_duration_s"]),
                direct_requests=direct["requests"],
                gateway_requests=gateway["requests"],
                direct_5xx=_five_xx(direct),
                gateway_5xx=_five_xx(gateway),
                admitted=int(gateway["requests"]) - _count_status(gateway, 429),
                direct_rate=_pct(row["bare_vllm_upstream_5xx_rate"]),
                gateway_rate=_pct(row["gateway_upstream_5xx_rate"]),
                rejects=row["http_429_count"],
                sampled=_fmt(row["sampled_queue_max"], 0),
                process=_fmt(row["process_queue_high_watermark"], 0),
                latency=_fmt(float(fast_p95) * 1000 if fast_p95 is not None else None, 1),
            )
        )
    return "\n".join(lines)


def _gate_checklist(
    sustained: Mapping[str, Any], finite: Mapping[str, Any], local: Mapping[str, Any]
) -> str:
    local_root = local.get("status") == "local_complete_remote_pending"
    regression = (
        local.get("regression", {}).get("pre_fix", {}).get("failed") == 2
        and local.get("regression", {}).get("post_change", {}).get("failed") == 0
    )
    finite_receipts = (
        len(finite["metrics"]["direct_gateway"]["pairs"]) == 9
        and len(finite["metrics"]["overload"]["pairs"]) == 3
    )
    gate = sustained["gate"]
    return "\n".join(
        [
            f"- [{'x' if local_root else ' '}] root cause documented with evidence",
            f"- [{'x' if regression else ' '}] regression tests old-fail/new-pass",
            f"- [{'x' if finite_receipts and gate['checks']['matched_same_box_schedules'] else ' '}] matched matrix + finite and sustained overload receipts",
            f"- [{'x' if gate['checks']['duration_at_least_120s_per_cell'] else ' '}] duration-based arrivals ≥120 s at 2×/3×/5× for bare vLLM and gateway",
            f"- [{'x' if gate['checks']['queue_saturated_in_every_cell'] and gate['checks']['bounded_queue'] else ' '}] queue saturated at the configured bound without exceeding it",
            f"- [{'x' if gate['checks']['429_fast_reject_semantics'] else ' '}] excess surfaced as fast HTTP 429/overloaded with Retry-After",
            f"- [{'x' if gate['checks']['upstream_5xx_parity'] else ' '}] admitted gateway upstream 5xx stayed within ±{gate['thresholds']['max_upstream_5xx_rate_delta'] * 100:.1f} pp of paired bare vLLM",
            f"- [x] production-blocked disposition recorded: {'lifted for the measured single-node gateway contract' if not sustained['production_blocked'] else 'retained because the sustained gate failed'}",
        ]
    )


def _report(
    sustained: Mapping[str, Any],
    finite: Mapping[str, Any],
    phase5: Mapping[str, Any],
    local: Mapping[str, Any],
) -> str:
    status = sustained["gate"]["status"].upper()
    decision = (
        "The Phase 5 production block is lifted for the measured single-node gateway overload contract only. This does not claim cloud production, multi-GPU scaling, or Kubernetes readiness."
        if status == "PASS"
        else "Production remains blocked because one or more duration-based overload checks failed. Phase 7.2 must not proceed."
    )
    return f"""# Phase 7.1 sustained-overload amendment report

Status: **Gate 7.1 {status}**. Run `{sustained["run_id"]}` at git `{sustained["git_sha"]}`. Raw receipt: `{sustained["raw_artifact"]}`.

## Decision

{decision}

The amendment used one same-box **{sustained["hardware"]["gpu"]}** ({sustained["hardware"]["gpu_memory_total_mib"]} MiB) on Aliyun ECS. Each 2×/3×/5× cell used fixed-seed Poisson arrivals through at least 120 seconds. All bare-vLLM cells completed before the gateway process started; direct and gateway schedules match by SHA-256. The A10 is a hardware substitution for the archived RTX 4090, so no cross-GPU latency or throughput comparison is made.

## Before receipt: archived RTX 4090 defect

{finite_report._history_table(phase5)}

Those 29 Phase 5 overload errors were admitted HTTP 502/`upstream_error` responses with `reject_overload=0`; success-only latency in error-bearing cells remains survivor-biased history and is not relabeled.

## Intermediate receipt: finite A10 burst

{finite_report._a10_overload_table(finite)}

The finite 60-request A10 burst had queue high-watermarks 8, 20, and 24 at 2×, 3×, and 5×. Only 5× reached the bound and correctly shed 14 requests. The original per-cell “at least one 429” proxy therefore failed for 2×/3× even though those finite bursts fit in the configured queue. This receipt remains valid; only that proxy was superseded by the human-approved duration amendment.

## After receipt: duration-based A10 overload

{_sustained_table(sustained)}

The gate was declared before this rerun: every direct and gateway arrival window must be at least {sustained["gate"]["thresholds"]["minimum_arrival_duration_s"]:.0f} seconds; every gateway cell must visibly sample the queue at its {sustained["gate"]["thresholds"]["queue_bound_requests"]}-request bound; all excess admission rejects must be HTTP 429/`overloaded`, carry `Retry-After`, and complete within {sustained["gate"]["thresholds"]["max_fast_reject_s"]:.1f} second; admitted gateway 5xx rate must stay within ±{sustained["gate"]["thresholds"]["max_upstream_5xx_rate_delta"] * 100:.1f} percentage points of paired bare vLLM.

## Cost

- Rate: `FORGE_GPU_HOURLY_USD=1.53`, derived from ¥11/hour at 7.2 CNY/USD.
- Delegated session through this receipt: {_fmt(sustained["cost"]["delegated_session_hours_through_gate"], 4)} h = **${_fmt(sustained["cost"]["usd"], 4)}**.

## Gate 7.1

{_gate_checklist(sustained, finite, local)}

## Reproduction

```sh
export FORGE_GPU_HOURLY_USD=1.53
export FORGE_BENCH_GIT_SHA={sustained["git_sha"]}
export FORGE_PHASE7_SESSION_STARTED_AT={sustained["started_at"]}
python -m gateway.bench.phase7_1_sustained --stage verify-artifact
python -m gateway.bench.phase7_1_sustained --stage bare
# start the pinned gateway only after the bare stage completes
python -m gateway.bench.phase7_1_sustained --stage gateway
python -m gateway.bench.phase7_1_sustained_report --update-readme
```
"""


def _readme_section(
    sustained: Mapping[str, Any], finite: Mapping[str, Any], phase5: Mapping[str, Any]
) -> str:
    return f"""{RESOLVED_SECTION_START}

The original Phase 5 RTX 4090 result remains an important negative finding: admitted requests returned HTTP 502/`upstream_error`, `reject_overload=0`, and non-stable gateway cells reached 10–85% errors while paired bare vLLM had 0%. Lower success-only latency in those cells was survivor-biased. The original [Phase 5 receipt](results/phase5/raw/phase5_gateway_bench.json) and [report](results/phase5_gateway_report.md) remain unchanged.

The first same-box A10 rerun fixed the connection-reuse failure: all nine matched matrix cells and all finite overload cells had 0 upstream 5xx. Its 60-request bursts reached queue high-watermarks 8/20/24 at 2×/3×/5×; 2× and 3× fit inside the 24-request queue, while 5× shed 14 requests as HTTP 429. The earlier “one 429 in every finite cell” check was therefore a miscalibrated proxy, not evidence that bounded admission failed. That [finite A10 receipt]({finite["raw_artifact"]}) remains preserved.

The human-approved amendment replaced that proxy with fixed-seed Poisson arrivals lasting at least 120 seconds per 2×/3×/5× cell, gateway versus same-box bare vLLM:

{_sustained_table(sustained)}

Every sustained gateway cell sampled the queue at its configured bound, and excess requests surfaced as fast HTTP 429/`overloaded` responses with `Retry-After`; admitted upstream 5xx remained within the predeclared ±{sustained["gate"]["thresholds"]["max_upstream_5xx_rate_delta"] * 100:.1f} pp band of paired bare vLLM. The Phase 5 production block is therefore lifted for this measured **single-node gateway overload contract only**. This is not evidence of cloud production, multi-GPU scaling, or Phase 7.2 Kubernetes readiness. See the [amended Gate 7.1 report](results/phase7_1_sustained_a10_report.md) and [raw sustained receipt]({sustained["raw_artifact"]}).

"""


def _replace_section(readme: str, replacement: str) -> str:
    starts = [
        marker for marker in (LEGACY_SECTION_START, RESOLVED_SECTION_START) if marker in readme
    ]
    if len(starts) != 1 or readme.count(SECTION_END) != 1:
        raise RuntimeError("README production section markers are missing or ambiguous")
    start = readme.index(starts[0])
    end = readme.index(SECTION_END)
    return f"{readme[:start]}{replacement}{readme[end:]}"


def run(*, update_readme: bool = False) -> dict[str, Any]:
    sustained = json.loads(SUSTAINED_RECEIPT.read_text())
    finite = json.loads(FINITE_RECEIPT.read_text())
    phase5 = json.loads(PHASE5_RECEIPT.read_text())
    local = json.loads(LOCAL_VERIFICATION.read_text())
    if sustained.get("status") != "complete" or sustained.get("phase") != "7.1":
        raise RuntimeError("sustained Gate 7.1 receipt is incomplete")
    REPORT_PATH.write_text(_report(sustained, finite, phase5, local))

    readme_updated = False
    if update_readme:
        if sustained["gate"]["status"] != "pass" or sustained.get("production_blocked"):
            raise RuntimeError("README red flag can only be lifted after sustained Gate 7.1 passes")
        replacement = _readme_section(sustained, finite, phase5)
        README_PATH.write_text(_replace_section(README_PATH.read_text(), replacement))
        readme_updated = True

    manifest = {
        "version": 1,
        "status": "complete",
        "phase": "7.1",
        "run_id": sustained["run_id"],
        "git_sha": sustained["git_sha"],
        "gate_status": sustained["gate"]["status"],
        "readme_updated": readme_updated,
        "generated_at": sustained["finished_at"],
        "inputs": {
            "sustained_receipt": {
                "path": relative_path(SUSTAINED_RECEIPT),
                "sha256": sha256_file(SUSTAINED_RECEIPT),
            },
            "finite_a10_receipt": {
                "path": relative_path(FINITE_RECEIPT),
                "sha256": sha256_file(FINITE_RECEIPT),
            },
            "phase5_history": {
                "path": relative_path(PHASE5_RECEIPT),
                "sha256": sha256_file(PHASE5_RECEIPT),
            },
            "local_verification": {
                "path": relative_path(LOCAL_VERIFICATION),
                "sha256": sha256_file(LOCAL_VERIFICATION),
            },
        },
        "outputs": {
            "report": {"path": relative_path(REPORT_PATH), "sha256": sha256_file(REPORT_PATH)},
            "readme": (
                {"path": relative_path(README_PATH), "sha256": sha256_file(README_PATH)}
                if readme_updated
                else None
            ),
        },
    }
    write_json_atomic(MANIFEST_PATH, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(update_readme=args.update_readme), sort_keys=True))


if __name__ == "__main__":
    main()
