#!/usr/bin/env python3
# ruff: noqa: E501 -- Markdown tables and evidence sentences stay readable as literals.
"""Render the Phase 7.1 A10 rerun report and conditionally lift the README red flag."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from statistics import median
from typing import Any

from forge.train.artifacts import write_json_atomic
from forge.train.config import REPO_ROOT, relative_path, sha256_file

RESULTS_DIR = REPO_ROOT / "results/phase7_1"
RAW_RECEIPT = RESULTS_DIR / "raw/phase7_1_gateway_bench.json"
LOCAL_VERIFICATION = RESULTS_DIR / "local_verification.json"
PHASE5_RECEIPT = REPO_ROOT / "results/phase5/raw/phase5_gateway_bench.json"
REPORT_PATH = REPO_ROOT / "results/phase7_1_gateway_a10_report.md"
MANIFEST_PATH = RESULTS_DIR / "report_manifest.json"
README_PATH = REPO_ROOT / "README.md"
SECTION_START = "## Production story: the gateway result has a red flag"
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


def _median_present(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return median(present) if present else None


def _stable_overhead(receipt: Mapping[str, Any]) -> dict[str, Any]:
    stable = [
        pair
        for pair in receipt["metrics"]["direct_gateway"]["pairs"]
        if float(pair["direct"]["error_rate"]) <= 0.05
        and float(pair["gateway"]["error_rate"]) <= 0.05
    ]
    return {
        "cells": len(stable),
        "p50": _median_present(pair["overhead"]["e2e_p50_overhead_pct"] for pair in stable),
        "p95": _median_present(pair["overhead"]["e2e_p95_overhead_pct"] for pair in stable),
        "throughput": _median_present(pair["overhead"]["throughput_delta_pct"] for pair in stable),
    }


def _a10_overload_table(receipt: Mapping[str, Any]) -> str:
    lines = [
        "| 负载 | bare A10 5xx | gateway A10 5xx | gateway 429 | reject_overload | queue sampled/process | 429 p50 ms |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    gate_rows = {float(row["multiplier"]): row for row in receipt["gate"]["overload_cells"]}
    for pair in receipt["metrics"]["overload"]["pairs"]:
        gateway = pair["gateway"]
        gate = gate_rows[float(pair["multiplier"])]
        lines.append(
            "| {multiplier:g}× | {direct_5xx}/{requests} | {gateway_5xx}/{requests} | {rejects}/{requests} | {decisions} | {sampled}/{process} | {latency} |".format(
                multiplier=float(pair["multiplier"]),
                direct_5xx=_five_xx(pair["direct"]),
                gateway_5xx=_five_xx(gateway),
                rejects=_count_status(gateway, 429),
                decisions=gate["reject_overload_decisions"],
                sampled=_fmt(gate["sampled_queue_max"], 0),
                process=_fmt(gate["process_queue_high_watermark"], 0),
                latency=(
                    _fmt(gateway["client"]["fast_reject"]["p50_s"] * 1000, 1)
                    if gateway["client"]["fast_reject"]["p50_s"] is not None
                    else "n/a"
                ),
                requests=gateway["requests"],
            )
        )
    return "\n".join(lines)


def _matrix_table(receipt: Mapping[str, Any]) -> str:
    lines = [
        "| 长度 | 并发 | bare/gateway errors | E2E p50 overhead | E2E p95 overhead | throughput delta | interpretation |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for pair in receipt["metrics"]["direct_gateway"]["pairs"]:
        direct_error = float(pair["direct"]["error_rate"])
        gateway_error = float(pair["gateway"]["error_rate"])
        stable = direct_error <= 0.05 and gateway_error <= 0.05
        lines.append(
            "| {profile} | {concurrency} | {direct}/{gateway} | {p50}% | {p95}% | {throughput}% | {meaning} |".format(
                profile=pair["length_profile"],
                concurrency=pair["concurrency"],
                direct=_pct(direct_error),
                gateway=_pct(gateway_error),
                p50=_fmt(pair["overhead"]["e2e_p50_overhead_pct"], 1),
                p95=_fmt(pair["overhead"]["e2e_p95_overhead_pct"], 1),
                throughput=_fmt(pair["overhead"]["throughput_delta_pct"], 1),
                meaning=(
                    "paired A10 stable cell"
                    if stable
                    else "not a latency win; success-only survivor set"
                ),
            )
        )
    return "\n".join(lines)


def _gate_checklist(receipt: Mapping[str, Any], local: Mapping[str, Any]) -> str:
    gate = receipt["gate"]
    local_root = local.get("status") == "local_complete_remote_pending"
    regression = (
        local.get("regression", {}).get("pre_fix", {}).get("failed") == 2
        and local.get("regression", {}).get("post_change", {}).get("failed") == 0
    )
    production_disposition = (
        "lifted for the measured single-node gateway contract"
        if not receipt["production_blocked"]
        else "retained because one or more remote acceptance checks failed"
    )
    return "\n".join(
        [
            f"- [{'x' if local_root else ' '}] root cause documented with evidence",
            f"- [{'x' if regression else ' '}] regression tests old-fail/new-pass",
            f"- [{'x' if gate['checks']['matched_matrix_receipts'] and gate['checks']['overload_receipts'] else ' '}] matched matrix + overload rerun receipts",
            f"- [{'x' if gate['checks']['429_fast_reject_semantics'] and gate['checks']['bounded_queue'] else ' '}] 429 semantics and bounded queue verified",
            f"- [{'x' if gate['checks']['upstream_5xx_parity'] else ' '}] upstream 5xx rate within the pinned ±{gate['thresholds']['max_upstream_5xx_rate_delta'] * 100:.1f} pp of paired bare vLLM",
            f"- [{'x' if gate['checks']['non_admission_error_parity'] else ' '}] all non-admission errors, including transport failures without an HTTP status, stayed within the same parity band",
            f"- [x] production-blocked disposition recorded: {production_disposition}",
        ]
    )


def _history_table(phase5: Mapping[str, Any]) -> str:
    lines = [
        "| RTX 4090 Phase 5 load | bare errors | gateway errors | observed semantics |",
        "|---:|---:|---:|---|",
    ]
    for pair in phase5["metrics"]["overload"]["pairs"]:
        gateway = pair["gateway"]
        codes = gateway["error_semantics"]["error_codes"]
        rendered = ", ".join(f"{key}:{value}" for key, value in sorted(codes.items())) or "none"
        lines.append(
            f"| {float(pair['multiplier']):g}× | {pair['direct']['errors']}/{pair['direct']['requests']} | {gateway['errors']}/{gateway['requests']} | HTTP statuses {gateway['http_status_counts']}; codes {rendered} |"
        )
    return "\n".join(lines)


def _report(receipt: Mapping[str, Any], phase5: Mapping[str, Any], local: Mapping[str, Any]) -> str:
    overhead = _stable_overhead(receipt)
    status = "PASS" if receipt["gate"]["status"] == "pass" else "FAIL"
    disposition = (
        "The Phase 5 production-blocked flag is lifted only for this measured single-node gateway overload contract. This is not a cloud-production or multi-GPU claim."
        if receipt["gate"]["status"] == "pass"
        else "Production remains blocked. The README historical red-flag section must remain unchanged."
    )
    return f"""# Phase 7.1 A10 gateway matched rerun report

Status: **Gate 7.1 {status}**. Run `{receipt["run_id"]}` at git `{receipt["git_sha"]}`. Raw receipt: `{receipt["raw_artifact"]}`.

## Decision

{disposition}

The rerun used one **{receipt["hardware"]["gpu"]}** ({receipt["hardware"]["gpu_memory_total_mib"]} MiB) on the Aliyun ECS VM. Phase 5 used an RTX 4090. The 4090 evidence below is retained only as defect history; **no A10 latency or throughput number is compared with a 4090 number**.

All A10 bare-vLLM capacity, nine-cell concurrency × length matrix, and overload baseline cells completed before the gateway process started. The same A10 model artifact, BF16 precision, vLLM 0.17.0, request rows, seeds, warm-ups, schedules, measurement counts, and deadlines were then used for the gateway cells.

## Archived Phase 5 finding (RTX 4090; history retained)

{_history_table(phase5)}

Every archived error above passed admission and returned HTTP 502/`upstream_error`; `reject_overload=0`. Those cells are not relabeled as 429 and their success-only latency remains survivor-biased.

## Phase 7.1 A10 overload rerun

{_a10_overload_table(receipt)}

The gate defines upstream-5xx parity before looking at results: each gateway cell must be within **±{receipt["gate"]["thresholds"]["max_upstream_5xx_rate_delta"] * 100:.1f} percentage points** of its paired A10 bare-vLLM cell. The same band also applies to all non-admission errors, so transport failures without an HTTP status cannot disappear from the gate. Every overload cell must contain at least {receipt["gate"]["thresholds"]["min_429_rejects_per_overload_cell"]} HTTP 429 response, with `Retry-After`, error code `overloaded`, client latency ≤{receipt["gate"]["thresholds"]["max_fast_reject_s"]:.1f} s, and both sampled and process queue high-watermarks ≤{receipt["gateway"]["max_queue_requests"]}.

## Same-box A10 matched matrix

{_matrix_table(receipt)}

Across {overhead["cells"]} stable A10 pairs, median same-box gateway overhead was p50 **{_fmt(overhead["p50"], 1)}%**, p95 **{_fmt(overhead["p95"], 1)}%**, and throughput delta **{_fmt(overhead["throughput"], 1)}%**. These claims are A10-to-A10 only.

## Cost and pricing assumption

- `FORGE_GPU_HOURLY_USD=1.53`.
- The rate assumes **¥11/hour at 7.2 CNY/USD**: 11 / 7.2 = 1.5278, rounded to $1.53/hour.
- Accounted delegated VM session: {_fmt(receipt["cost"]["vm_session_hours"], 4)} h = **${_fmt(receipt["cost"]["usd"], 4)}**. Time after final receipt while the owner stops the VM is excluded and disclosed.

## Gate 7.1

{_gate_checklist(receipt, local)}

## Reproduction

```sh
export FORGE_GPU_HOURLY_USD=1.53
export FORGE_BENCH_GIT_SHA={receipt["git_sha"]}
export FORGE_VM_STARTED_AT={receipt["started_at"]}
python -m gateway.bench.phase7_1_bench --stage verify-artifact
python -m gateway.bench.phase7_1_bench --stage baseline
python -m gateway.bench.phase7_1_bench --stage gateway-matrix
python -m gateway.bench.phase7_1_bench --stage gateway-overload
python -m gateway.bench.phase7_1_report
```
"""


def _readme_section(receipt: Mapping[str, Any], phase5: Mapping[str, Any]) -> str:
    overhead = _stable_overhead(receipt)
    return f"""## Production story: Phase 7.1 repaired overload semantics on A10

The original Phase 5 RTX 4090 result remains an important negative finding: admitted requests returned HTTP 502/`upstream_error`, `reject_overload=0`, and non-stable gateway cells reached 10–85% errors while paired bare vLLM had 0%. Lower success-only latency in those cells was survivor-biased. The original [Phase 5 receipt](results/phase5/raw/phase5_gateway_bench.json) and [report](results/phase5_gateway_report.md) remain unchanged.

Phase 7.1 first completed a fresh bare-vLLM baseline on one Aliyun ECS **NVIDIA A10**, then ran the exact matched gateway matrix and overload schedules on that same VM. This is a hardware substitution, not a 4090-to-A10 performance comparison; all current overhead claims are paired A10-to-A10 only.

{_a10_overload_table(receipt)}

Every A10 overload cell produced bounded-queue HTTP 429/`overloaded` fast rejects with `Retry-After`; upstream 5xx stayed within the predeclared ±{receipt["gate"]["thresholds"]["max_upstream_5xx_rate_delta"] * 100:.1f} pp parity band of paired bare vLLM. Across {overhead["cells"]} stable A10 matrix pairs, median E2E overhead was p50 **{_fmt(overhead["p50"], 1)}% / p95 {_fmt(overhead["p95"], 1)}%**, with throughput delta **{_fmt(overhead["throughput"], 1)}%**.

The Phase 5 production block is therefore lifted for this measured **single-node gateway overload contract only**. It is not evidence of cloud production, multi-GPU scaling, or Phase 7.2 Kubernetes readiness. Full A10 methodology, pricing assumption, raw pointers, and Gate 7.1 checklist: [Phase 7.1 report](results/phase7_1_gateway_a10_report.md).

"""


def _replace_section(readme: str, replacement: str) -> str:
    if readme.count(SECTION_START) != 1 or readme.count(SECTION_END) != 1:
        raise RuntimeError("README production section markers are missing or ambiguous")
    start = readme.index(SECTION_START)
    end = readme.index(SECTION_END)
    return f"{readme[:start]}{replacement}{readme[end:]}"


def run(*, update_readme: bool = False) -> dict[str, Any]:
    receipt = json.loads(RAW_RECEIPT.read_text())
    phase5 = json.loads(PHASE5_RECEIPT.read_text())
    local = json.loads(LOCAL_VERIFICATION.read_text())
    if receipt.get("status") != "complete" or str(receipt.get("phase")) != "7.1":
        raise RuntimeError("Phase 7.1 final receipt is incomplete")
    report = _report(receipt, phase5, local)
    REPORT_PATH.write_text(report)

    readme_updated = False
    if update_readme:
        if receipt["gate"]["status"] != "pass" or receipt.get("production_blocked"):
            raise RuntimeError("README red flag can only be rewritten after Gate 7.1 passes")
        replacement = _readme_section(receipt, phase5)
        README_PATH.write_text(_replace_section(README_PATH.read_text(), replacement))
        readme_updated = True

    manifest = {
        "version": 1,
        "status": "complete",
        "phase": "7.1",
        "run_id": receipt["run_id"],
        "git_sha": receipt["git_sha"],
        "gate_status": receipt["gate"]["status"],
        "readme_updated": readme_updated,
        "generated_at": receipt["finished_at"],
        "inputs": {
            "raw_receipt": {"path": relative_path(RAW_RECEIPT), "sha256": sha256_file(RAW_RECEIPT)},
            "local_verification": {
                "path": relative_path(LOCAL_VERIFICATION),
                "sha256": sha256_file(LOCAL_VERIFICATION),
            },
            "phase5_history": {
                "path": relative_path(PHASE5_RECEIPT),
                "sha256": sha256_file(PHASE5_RECEIPT),
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
