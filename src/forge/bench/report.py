"""Regenerate the Phase 4 serving report and figures from raw artifacts only."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from forge.train.artifacts import write_json_atomic, write_text_atomic
from forge.train.config import REPO_ROOT, relative_path, sha256_file

from .config import load_phase4_config, phase4_config_paths, phase4_raw_path


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _number(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _load_receipts(*, smoke: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    configs = [load_phase4_config(path) for path in phase4_config_paths()]
    receipts = []
    missing = []
    for config in configs:
        path = phase4_raw_path(config, smoke=smoke)
        if not path.is_file():
            missing.append(relative_path(path))
            continue
        receipt = json.loads(path.read_text())
        if (
            receipt.get("status") != "complete"
            or receipt.get("config_hash") != config["_config_hash"]
            or receipt.get("run_id") != config["run_id"]
        ):
            raise RuntimeError(f"raw artifact/config mismatch: {path}")
        requests = REPO_ROOT / receipt["request_artifact"]["path"]
        if sha256_file(requests) != receipt["request_artifact"]["sha256"]:
            raise RuntimeError(f"request artifact hash mismatch: {requests}")
        evidence = receipt.get("speculative_method_evidence")
        if not smoke and receipt["model"].get("speculative_enabled"):
            if not isinstance(evidence, dict):
                raise RuntimeError(f"speculative method evidence is missing: {path}")
            evidence_path = REPO_ROOT / evidence["path"]
            if sha256_file(evidence_path) != evidence["sha256"]:
                raise RuntimeError(f"speculative method evidence hash mismatch: {evidence_path}")
            base_audit = evidence.get("base_index_audit")
            if isinstance(base_audit, dict):
                base_audit_path = REPO_ROOT / base_audit["path"]
                if sha256_file(base_audit_path) != base_audit["sha256"]:
                    raise RuntimeError(f"base index audit hash mismatch: {base_audit_path}")
            for failure in evidence["prior_failures"]:
                failure_path = REPO_ROOT / failure["path"]
                if sha256_file(failure_path) != failure["sha256"]:
                    raise RuntimeError(f"prior failure evidence hash mismatch: {failure_path}")
        receipts.append(receipt)
    if missing:
        raise RuntimeError(f"Phase 4 raw artifacts are incomplete: {', '.join(missing)}")
    return configs, receipts


def _representative_point(receipt: dict[str, Any]) -> dict[str, Any]:
    points = receipt["metrics"]["points"]
    stable = [point for point in points if point["stable"]]
    return (stable or points)[-1]


def _serving_tables(receipts: list[dict[str, Any]]) -> list[str]:
    serve = [item for item in receipts if item["experiment"] == "serve"]
    lines = [
        "## Serving sweep",
        "",
        "Client timings are wall-clock observations around the streamed HTTP request. "
        "ITL is the per-request mean `(E2E - TTFT) / (completion_tokens - 1)`. "
        "The server-side table below is independently derived from vLLM Prometheus histograms.",
        "",
        "| Variant | Precision | QPS | TTFT p50 s | ITL p50 s | E2E p50 s | E2E p95 s | tok/s | req/s | Success | VRAM peak MiB | Cost / 1k success | Stable |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for receipt in serve:
        for point in receipt["metrics"]["points"]:
            client = point["client"]
            lines.append(
                "| {variant} | {precision} | {qps:.3f} | {ttft} | {itl} | {p50} | "
                "{p95} | {tps} | {rps} | {success} | {vram} | {cost} | {stable} |".format(
                    variant=receipt["model"]["variant"],
                    precision=receipt["model"]["precision"],
                    qps=point["arrival_rate_qps"],
                    ttft=_seconds(client["ttft"]["p50_s"]),
                    itl=_seconds(client["itl"]["p50_s"]),
                    p50=_seconds(client["e2e"]["p50_s"]),
                    p95=_seconds(client["e2e"]["p95_s"]),
                    tps=_number(client["output_tokens_per_s"]),
                    rps=_number(client["request_throughput_per_s"]),
                    success=_pct(point["verifier_task_success_rate"]),
                    vram=_number(point["vram"]["peak_mib"], 0),
                    cost=_money(point["cost_per_1k_successful_tasks_usd"]),
                    stable="yes" if point["stable"] else "no",
                )
            )
    lines.extend(
        [
            "",
            "| Variant | Precision | Max stable observed concurrency | Max stable offered QPS |",
            "|---|---|---:|---:|",
        ]
    )
    for receipt in serve:
        stable = [point for point in receipt["metrics"]["points"] if point["stable"]]
        lines.append(
            "| {variant} | {precision} | {concurrency} | {qps} |".format(
                variant=receipt["model"]["variant"],
                precision=receipt["model"]["precision"],
                concurrency=(
                    max(point["max_observed_in_flight"] for point in stable) if stable else "none"
                ),
                qps=_number(max((point["arrival_rate_qps"] for point in stable), default=None), 3),
            )
        )
    lines.extend(
        [
            "",
            "### Independent server-side timing",
            "",
            "The representative point is the highest stable QPS, or the highest tested QPS "
            "when no point passes the declared stability criteria.",
            "",
            "| Variant | Precision | QPS | Server TTFT p50/p95 s | Server ITL p50/p95 s | Server E2E p50/p95 s |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for receipt in serve:
        point = _representative_point(receipt)
        server = point["server"]
        lines.append(
            "| {variant} | {precision} | {qps:.3f} | {ttft50}/{ttft95} | "
            "{itl50}/{itl95} | {e2e50}/{e2e95} |".format(
                variant=receipt["model"]["variant"],
                precision=receipt["model"]["precision"],
                qps=point["arrival_rate_qps"],
                ttft50=_seconds(server["ttft"]["p50_s"]),
                ttft95=_seconds(server["ttft"]["p95_s"]),
                itl50=_seconds(server["itl"]["p50_s"]),
                itl95=_seconds(server["itl"]["p95_s"]),
                e2e50=_seconds(server["e2e"]["p50_s"]),
                e2e95=_seconds(server["e2e"]["p95_s"]),
            )
        )
    return lines


def _spec_rows(receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spec = [item for item in receipts if item["experiment"] == "spec_decode"]
    baseline = next(item for item in spec if not item["model"].get("speculative_enabled"))
    enabled = next(item for item in spec if item["model"].get("speculative_enabled"))
    baseline_by_qps = {point["arrival_rate_qps"]: point for point in baseline["metrics"]["points"]}
    rows = []
    for point in enabled["metrics"]["points"]:
        base = baseline_by_qps[point["arrival_rate_qps"]]
        base_p95 = base["client"]["e2e"]["p95_s"]
        spec_p95 = point["client"]["e2e"]["p95_s"]
        base_success = base["client"]["successful_task_throughput_per_s"]
        spec_success = point["client"]["successful_task_throughput_per_s"]
        win = (
            base_p95 is not None
            and spec_p95 is not None
            and spec_p95 <= base_p95
            and spec_success >= base_success
        )
        rows.append(
            {
                "qps": point["arrival_rate_qps"],
                "baseline_p95_s": base_p95,
                "spec_p95_s": spec_p95,
                "p95_delta_s": (
                    spec_p95 - base_p95 if base_p95 is not None and spec_p95 is not None else None
                ),
                "baseline_successful_req_s": base_success,
                "spec_successful_req_s": spec_success,
                "acceptance_rate": point["server"]["speculative"]["acceptance_rate"],
                "mean_acceptance_length": point["server"]["speculative"]["mean_acceptance_length"],
                "load1_max": point.get("co_tenancy", {}).get("load1_max"),
                "cpu_utilization_mean": point.get("co_tenancy", {}).get("cpu_utilization_mean"),
                "contaminated": point.get("co_tenancy", {}).get("contaminated"),
                "verdict": "win" if win else "lose",
            }
        )
    return rows


def _spec_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 720, 360
    left, right, top, bottom = 70, 30, 35, 65
    qps_values = [float(row["qps"]) for row in rows]
    deltas = [float(row["p95_delta_s"] or 0.0) for row in rows]
    x_min, x_max = min(qps_values), max(qps_values)
    y_extent = max(max(abs(value) for value in deltas), 0.001)

    def x(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-9) * (width - left - right)

    def y(value: float) -> float:
        return top + (y_extent - value) / (2 * y_extent) * (height - top - bottom)

    points = " ".join(
        f"{x(qps):.1f},{y(delta):.1f}" for qps, delta in zip(qps_values, deltas, strict=True)
    )
    circles = "\n".join(
        f'<circle cx="{x(float(row["qps"])):.1f}" cy="{y(float(row["p95_delta_s"] or 0)):.1f}" '
        f'r="6" fill="{"#1b9e77" if row["verdict"] == "win" else "#d95f02"}" />'
        for row in rows
    )
    labels = "\n".join(
        f'<text x="{x(value):.1f}" y="{height - 34}" text-anchor="middle" font-size="12">{value:g}</text>'
        for value in qps_values
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="20" text-anchor="middle" font-family="sans-serif" font-size="16">Speculative decoding win/lose boundary</text>
<line x1="{left}" x2="{width - right}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="#555" stroke-dasharray="5 4"/>
<polyline points="{points}" fill="none" stroke="#377eb8" stroke-width="2"/>
{circles}
{labels}
<text x="{width / 2}" y="{height - 8}" text-anchor="middle" font-family="sans-serif" font-size="13">Offered QPS</text>
<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">Spec - baseline p95 E2E (s)</text>
<text x="{left}" y="{height - 48}" font-family="sans-serif" font-size="11" fill="#1b9e77">green = win</text>
<text x="{left + 95}" y="{height - 48}" font-family="sans-serif" font-size="11" fill="#d95f02">orange = lose</text>
</svg>
"""


def _spec_section(receipts: list[dict[str, Any]], *, figure_path: Path) -> list[str]:
    rows = _spec_rows(receipts)
    enabled = next(
        item
        for item in receipts
        if item["experiment"] == "spec_decode" and item["model"].get("speculative_enabled")
    )
    speculative = enabled.get("speculative") or {}
    evidence = enabled.get("speculative_method_evidence") or {}
    write_text_atomic(figure_path, _spec_svg(rows))
    wins = [row["qps"] for row in rows if row["verdict"] == "win"]
    boundary = f"highest tested winning QPS = {max(wins):g}" if wins else "no win in tested range"
    lines = [
        "## Speculative decoding boundary",
        "",
        "A point is a win only when speculative decoding has no worse client p95 E2E "
        "and no lower verifier-successful request throughput than the matched baseline. "
        f"Boundary: **{boundary}**.",
        "",
        "Method run: **{method}** (`{model}`, {tokens} draft token). D1.3 selected "
        "the model-native path because `{reason}`; the fixed-revision base index contains "
        "{mtp_count} `mtp.*` weights and the R1b adapter contains none. The sibling R1b "
        "export restores those exact base tensor bytes. vLLM stays at `{version}` with "
        "no M-RoPE patch and no version change.".format(
            method=speculative.get("method", "unrecorded"),
            model=speculative.get("draft_model", "model-native MTP"),
            tokens=speculative.get("num_speculative_tokens", "n/a"),
            reason=evidence.get("reason", "smoke_config_only"),
            mtp_count=(evidence.get("base_index_audit") or {}).get(
                "mtp_weight_key_count", "not evaluated in smoke"
            ),
            version=(evidence.get("vllm_contract") or {}).get("version", "0.17.0"),
        ),
        "",
        "Prior failed attempts remain archived: "
        + ", ".join(f"`{item['path']}`" for item in evidence.get("prior_failures", []))
        + ("." if evidence.get("prior_failures") else "not loaded in SMOKE."),
        "",
        f"![Speculative decoding boundary]({figure_path.name})",
        "",
        "| QPS | Baseline p95 s | Spec p95 s | Delta s | Baseline success req/s | Spec success req/s | Acceptance | Mean acceptance length | Load1 max | CPU mean | Clean | Verdict |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {qps:.3f} | {base} | {spec} | {delta} | {base_req} | {spec_req} | "
            "{acceptance} | {length} | {load1} | {cpu} | {clean} | {verdict} |".format(
                qps=row["qps"],
                base=_seconds(row["baseline_p95_s"]),
                spec=_seconds(row["spec_p95_s"]),
                delta=_seconds(row["p95_delta_s"]),
                base_req=_number(row["baseline_successful_req_s"]),
                spec_req=_number(row["spec_successful_req_s"]),
                acceptance=_pct(row["acceptance_rate"]),
                length=_number(row["mean_acceptance_length"]),
                load1=_number(row["load1_max"]),
                cpu=_pct(row["cpu_utilization_mean"]),
                clean="no" if row["contaminated"] else "yes",
                verdict=row["verdict"],
            )
        )
    return lines


def _structured_section(receipts: list[dict[str, Any]]) -> list[str]:
    structured = [item for item in receipts if item["experiment"] == "structured"]
    lines = [
        "## Structured-output deep dive",
        "",
        "### Backend overhead and cold compile",
        "",
        "Cold compile overhead is first request latency minus the median repeated latency "
        "for the same previously unseen schema. Steady constraint overhead is repeated "
        "constrained median minus an unconstrained control with the same prompt.",
        "",
        "| Backend | Required fields | Cold latency s | Steady p50 s | Cold compile overhead s | Steady constraint overhead s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for receipt in structured:
        for item in receipt["metrics"]["schema_compile"]:
            steady = [row["latency_s"] for row in item["steady"] if row["error"] is None]
            steady.sort()
            steady_p50 = steady[len(steady) // 2] if steady else None
            lines.append(
                "| {backend} | {fields} | {cold} | {steady} | {compile} | {overhead} |".format(
                    backend=receipt["metrics"]["backend"],
                    fields=item["schema_required_fields"],
                    cold=_seconds(item["cold"]["latency_s"]),
                    steady=_seconds(steady_p50),
                    compile=_seconds(item["cold_compile_overhead_s"]),
                    overhead=_seconds(item["steady_constraint_overhead_s"]),
                )
            )
    lines.extend(
        [
            "",
            "### Constraint tax and two-pass mitigation",
            "",
            "The one-pass condition sends `tool_choice=required` and a simultaneous response "
            "JSON schema. Two-pass first obtains a model-selected tool, then constrains the "
            "complete ticket with that selected tool fixed. Missing first-pass choices are not "
            "filled from gold labels and count as failures.",
            "",
            "| Backend | Tools-only call rate | Simultaneous call rate | Constraint-tax delta | One-pass task success | Two-pass task success | Mitigation delta | One-pass p50 s | Two-pass p50 s | Latency delta s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for receipt in structured:
        item = receipt["metrics"]["constraint_tax_and_mitigation"]
        lines.append(
            "| {backend} | {tools} | {sim} | {tax} | {before} | {after} | {delta} | "
            "{before_lat} | {after_lat} | {lat_delta} |".format(
                backend=receipt["metrics"]["backend"],
                tools=_pct(item["unconstrained_tool_call_rate"]),
                sim=_pct(item["simultaneous_schema_tool_call_rate"]),
                tax=_pct(item["constraint_tax_tool_call_rate_delta"]),
                before=_pct(item["simultaneous_task_success"]),
                after=_pct(item["two_pass_task_success"]),
                delta=_pct(item["mitigation_task_success_delta"]),
                before_lat=_seconds(item["simultaneous_latency_p50_s"]),
                after_lat=_seconds(item["two_pass_latency_p50_s"]),
                lat_delta=_seconds(item["latency_delta_p50_s"]),
            )
        )
    lines.extend(
        [
            "",
            "### CPU co-tenancy measurements",
            "",
            "| Backend | Logical cores | Load1 threshold | Load1 mean/max | CPU mean/max | Samples | Clean | Contaminated attempts retained |",
            "|---|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for receipt in structured:
        load = receipt["metrics"].get("co_tenancy", {})
        lines.append(
            "| {backend} | {cores} | {threshold} | {load_mean}/{load_max} | "
            "{cpu_mean}/{cpu_max} | {samples} | {clean} | {attempts} |".format(
                backend=receipt["metrics"]["backend"],
                cores=load.get("logical_cpu_count", "n/a"),
                threshold=_number(load.get("load1_contamination_threshold")),
                load_mean=_number(load.get("load1_mean")),
                load_max=_number(load.get("load1_max")),
                cpu_mean=_pct(load.get("cpu_utilization_mean")),
                cpu_max=_pct(load.get("cpu_utilization_max")),
                samples=load.get("samples", 0),
                clean="no" if load.get("contaminated") else "yes",
                attempts=len(receipt.get("contaminated_sweeps", [])),
            )
        )
    return lines


def _disclosure(receipts: list[dict[str, Any]], *, smoke: bool) -> list[str]:
    first = receipts[0]
    workload = first["workload"]
    hardware = first["hardware"]
    serve_models = [
        f"{item['model']['variant']} / {item['model']['precision']}"
        for item in receipts
        if item["experiment"] == "serve"
    ]
    lines = [
        "## Disclosure",
        "",
        f"- Mode: {'SMOKE (non-headline)' if smoke else 'FULL GPU'}.",
        f"- Hardware: {hardware.get('gpu') or 'mock CPU'}; driver {hardware.get('driver_version') or 'n/a'}; device memory {hardware.get('gpu_memory_total_mib') or 'n/a'} MiB.",
        f"- CPU co-tenancy: the pod was shared with an unrelated CPU-only task; host has {hardware.get('logical_cpu_count') or 'n/a'} logical cores. Every new spec-decode and structured sweep sampled load average and CPU utilization; any load1 sample above half the core count contaminated and reran the entire sweep. Contaminated attempts remain linked from the raw receipt.",
        "- Historical serving sweeps completed before the CPU co-tenancy sampling requirement and are identified as such; matched spec baseline/native-MTP and both structured backends use the new clean-sweep gate.",
        f"- Serving variants and precision: {', '.join(serve_models)}.",
        f"- vLLM version: {first['server']['packages'].get('vllm') or first['server']['declared_vllm_version']}.",
        f"- Workload source: frozen `{workload['split']}` evaluation rows; workload SHA-256 `{workload['sha256']}`.",
        f"- Input lengths: targets {workload['input_token_targets']} with weights {workload['input_token_weights']}; measurement = {workload['input_length_measurement']}.",
        f"- Output controls: max-token caps {workload['output_token_caps']} with weights {workload['output_token_weights']}.",
        f"- Arrival process: Poisson, fixed seed {workload['request_seed']}; offered sweep {workload['arrival_rates_qps']} QPS; concurrency cap {workload['max_concurrency']}.",
        f"- Warm-up: {workload['warmup_requests']} requests excluded; measurement: {workload['measurement_requests']} requests per QPS point.",
        f"- Timing sides: client = {first['timing_disclosure']['client']}; server = {first['timing_disclosure']['server']}.",
        f"- Verifier: `{first['verifier_disclosure']['implementation']}`; input normalization = {first['verifier_disclosure']['input_normalization']}; request artifacts preserve both raw and normalized outputs.",
        "- Stable means all declared error-rate, deadline-miss, achieved-QPS, and p95-inflation checks pass; max stable concurrency is the largest observed in-flight count among such points.",
        "- Cost per 1k successful tasks uses verifier-passing requests in the denominator, never token count.",
    ]
    for receipt in receipts:
        supersedes = receipt.get("supersedes")
        if supersedes:
            lines.append(
                f"- Superseded measurement retained: `{supersedes['run_id']}` at "
                f"`{supersedes['raw_artifact']}`; reason: {supersedes['reason']}"
            )
    return lines


def build_report(*, smoke: bool) -> dict[str, Any]:
    configs, receipts = _load_receipts(smoke=smoke)
    report_path = REPO_ROOT / (
        "data/smoke/phase4/phase4_serving_report.md"
        if smoke
        else "results/phase4_serving_report.md"
    )
    figure_path = REPO_ROOT / (
        "data/smoke/phase4/phase4_spec_decode_boundary.svg"
        if smoke
        else "results/phase4_spec_decode_boundary.svg"
    )
    title = "# Phase 4 serving and inference engineering report"
    lines = [title, "", *_disclosure(receipts, smoke=smoke), ""]
    lines.extend(_serving_tables(receipts))
    lines.extend([""])
    lines.extend(_spec_section(receipts, figure_path=figure_path))
    lines.extend([""])
    lines.extend(_structured_section(receipts))
    serve = [item for item in receipts if item["experiment"] == "serve"]
    costs_complete = all(
        point["cost_per_1k_successful_tasks_usd"] is not None
        for receipt in serve
        for point in receipt["metrics"]["points"]
    )
    latency_receipts = [
        item for item in receipts if item["experiment"] in {"spec_decode", "structured"}
    ]
    co_tenancy_clean = all(
        (
            all(
                point.get("co_tenancy", {}).get("samples", 0) > 0
                and not point.get("co_tenancy", {}).get("contaminated", True)
                for point in receipt["metrics"]["points"]
            )
            if receipt["experiment"] == "spec_decode"
            else receipt["metrics"].get("co_tenancy", {}).get("samples", 0) > 0
            and not receipt["metrics"].get("co_tenancy", {}).get("contaminated", True)
        )
        for receipt in latency_receipts
    )
    spec_enabled = next(
        item
        for item in receipts
        if item["experiment"] == "spec_decode" and item["model"].get("speculative_enabled")
    )
    acceptance_complete = all(
        point["server"]["speculative"].get("acceptance_rate") is not None
        for point in spec_enabled["metrics"]["points"]
    )
    lines.extend(
        [
            "",
            "## Phase 4 gate",
            "",
            "- [x] Disclosure block includes hardware, precision, load distribution, arrival rate, warm-up, and timing side.",
            f"- [{'x' if co_tenancy_clean else ' '}] New latency-sensitive sweeps record CPU/load co-tenancy and final headline attempts are below the contamination threshold.",
            f"- [{'x' if costs_complete else ' '}] Cost per 1k successful tasks is computed against verifier-passing requests.",
            f"- [{'x' if acceptance_complete else ' '}] Speculative-decoding acceptance rate is recorded at every QPS point.",
            "- [x] Speculative-decoding win/lose boundary is tabulated and plotted.",
            "- [x] Constraint-tax tool-call rates and two-pass task-success/latency deltas are reported.",
            "- [x] Report and SVG are deterministic functions of hash-checked raw artifacts.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "make bench-report",
            "```",
            "",
            "Raw provenance:",
            "",
        ]
    )
    for receipt in receipts:
        lines.append(
            f"- `{receipt['run_id']}`: `{receipt['raw_artifact']}`; config `{receipt['config_path']}`; Git `{receipt['git_sha']}`."
        )
    content = "\n".join(lines) + "\n"
    write_text_atomic(report_path, content)
    manifest_path = REPO_ROOT / (
        "data/smoke/phase4/report_manifest.json" if smoke else "results/phase4_report_manifest.json"
    )
    manifest = {
        "version": 1,
        "status": "complete",
        "phase": 4,
        "mode": "smoke" if smoke else "full",
        "report_path": relative_path(report_path),
        "report_sha256": sha256_file(report_path),
        "figure_path": relative_path(figure_path),
        "figure_sha256": sha256_file(figure_path),
        "source_raw_artifacts": {
            receipt["run_id"]: {
                "path": receipt["raw_artifact"],
                "sha256": sha256_file(REPO_ROOT / receipt["raw_artifact"]),
            }
            for receipt in receipts
        },
        "config_hashes": {config["run_id"]: config["_config_hash"] for config in configs},
        "supplemental_evidence": {
            receipt["run_id"]: receipt["speculative_method_evidence"]
            for receipt in receipts
            if receipt.get("speculative_method_evidence") is not None
        },
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = build_report(smoke=args.smoke)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
