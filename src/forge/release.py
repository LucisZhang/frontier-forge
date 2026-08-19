"""Phase 6 offline release, demo, and headline-reproduction gates."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from statistics import median
from typing import Any

from forge.train.artifacts import write_json_atomic
from forge.train.config import REPO_ROOT, relative_path, sha256_file

RESULTS = REPO_ROOT / "results"
SOURCE_MANIFEST = RESULTS / "phase6/source_manifest.json"
HEADLINE_PATH = RESULTS / "phase6/headline.json"
HANDOFF_PATH = RESULTS / "phase6/cascade_handoff.json"
RELEASE_MANIFEST = RESULTS / "phase6/release_manifest.json"
DEMO_DATA_PATH = REPO_ROOT / "demo/data/release.json"
DEMO_DATA_JS = REPO_ROOT / "demo/data/release.js"
DEMO_DIST = REPO_ROOT / "demo/dist"

SOURCE_PATHS = tuple(
    REPO_ROOT / path
    for path in (
        "data/ingest/manifest.json",
        "results/runs.jsonl",
        "results/phase3_paired_deltas.json",
        "results/phase3_backend_agreement.json",
        "results/phase3_export_manifest_r1b_trl_s0.json",
        "results/phase3_export_selection.json",
        "results/phase4/raw/phase4_serve_r1b_bf16_v2.json",
        "results/phase4/raw/phase4_serve_r1b_gptq_int4.json",
        "results/phase4/raw/phase4_spec_decode_r1b_bf16_baseline_v2.json",
        "results/phase4/raw/phase4_spec_decode_r1b_bf16_native_mtp.json",
        "results/phase4/raw/phase4_structured_r1b_bf16_xgrammar.json",
        "results/phase4/raw/phase4_structured_r1b_bf16_outlines.json",
        "results/phase4/r1b_mtp_reexport_manifest.json",
        "results/phase5/raw/phase5_gateway_bench.json",
        "results/phase5/verification.json",
    )
)

RELEASE_FILES = tuple(
    REPO_ROOT / path
    for path in (
        "README.md",
        "MODEL_CARD.md",
        "gateway/README.md",
        "results/phase5_gateway_report.md",
        "results/phase6/source_manifest.json",
        "results/phase6/headline.json",
        "results/phase6/cascade_handoff.json",
        "results/phase6/hf_archive_receipt.json",
        "results/phase6/remote_disk_audit.json",
        "results/phase6/hf_archive_logs.tar.gz",
        "demo/index.html",
        "demo/assets/styles.css",
        "demo/assets/app.js",
        "demo/data/release.json",
        "demo/data/release.js",
    )
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _javascript_bytes(value: object) -> bytes:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return f"window.FORGE_RELEASE = {payload};\n".encode()


def _records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in (RESULTS / "runs.jsonl").read_text().splitlines():
        if not line:
            continue
        record = json.loads(line)
        run_id = record.get("run_id")
        if run_id in records:
            raise RuntimeError(f"duplicate run_id in append-only ledger: {run_id}")
        records[str(run_id)] = record
    return records


def build_source_manifest() -> dict[str, Any]:
    missing = [relative_path(path) for path in SOURCE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 6 source files missing: {missing}")
    return {
        "version": 1,
        "phase": 6,
        "files": [
            {"path": relative_path(path), "sha256": sha256_file(path)} for path in SOURCE_PATHS
        ],
    }


def verify_source_manifest() -> dict[str, Any]:
    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(f"missing source manifest: {SOURCE_MANIFEST}")
    committed = _json(SOURCE_MANIFEST)
    current = build_source_manifest()
    if committed != current:
        expected = {item["path"]: item["sha256"] for item in committed.get("files", [])}
        actual = {item["path"]: item["sha256"] for item in current["files"]}
        changed = sorted(
            path
            for path in expected.keys() | actual.keys()
            if expected.get(path) != actual.get(path)
        )
        raise RuntimeError(f"Phase 6 source hash gate failed: {changed}")
    return committed


def _phase3_payload(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    paired = _json(RESULTS / "phase3_paired_deltas.json")
    ladder_ids = (
        ("R0 base", "r0_base_s0", "complete"),
        ("R1 rule SFT (1,450)", "r1_sft_rule_s0", "complete"),
        ("R1b rule SFT (20,000)", "r1b_sft_rule_20k_s0", "release-selected"),
        ("R2 distilled SFT", "r2_sft_distilled_s0", "complete-negative"),
        ("R3 DPO", "r3_dpo_s0", "complete"),
        ("R4 v2 GRPO seed 0", "r4_grpo_phase3_2_fresh_pool_s0", "partial-only"),
        ("R4 v2 GRPO seed 1", "r4_grpo_phase3_2_fresh_pool_s1", "partial-only"),
    )
    ladder = []
    for label, run_id, status in ladder_ids:
        record = records[run_id]
        ladder.append(
            {
                "label": label,
                "run_id": run_id,
                "status": status,
                "task_success": record["metrics"]["task_success"],
                "ci95": record["metrics"]["ci95"],
                "schema_valid": record["metrics"]["schema_valid"],
                "tool_accuracy": record["metrics"]["tool_acc"],
                "gpu_hours": record["cost"]["gpu_hours"],
                "usd": record["cost"]["usd"],
            }
        )
    headline_run = records["r1b_sft_rule_20k_s0"]
    delta = next(
        item
        for item in paired["optional_ablation_pairs"]
        if item["from"] == "r1" and item["to"] == "r1b"
    )
    return {
        "headline": {
            "run_id": headline_run["run_id"],
            "task_success": headline_run["metrics"]["task_success"],
            "ci95": headline_run["metrics"]["ci95"],
            "paired_delta_vs_r1": delta,
            "gpu_hours": headline_run["cost"]["gpu_hours"],
            "usd": headline_run["cost"]["usd"],
            "statement": (
                "Scaling free rule labels from 1,450 to 20,000 raised frozen-eval task "
                "success from 66.35% to 99.05%: +32.70 percentage points, paired 95% "
                "CI [30.60, 34.50], using 15.236 measured RTX 4090 GPU-hours ($4.571)."
            ),
        },
        "ladder": ladder,
        "r4_v2": paired["r4_v2_aggregate"],
        "r4_seed_deltas": paired["r4_v2_seed_deltas"],
        "backend_agreement": _json(RESULTS / "phase3_backend_agreement.json"),
    }


def _point(receipt: dict[str, Any], qps: float) -> dict[str, Any]:
    matches = [
        item
        for item in receipt["metrics"]["points"]
        if abs(float(item["arrival_rate_qps"]) - qps) < 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {qps:g} QPS point in {receipt.get('run_id')}")
    return matches[0]


def _serving_point(receipt: dict[str, Any], *, label: str) -> dict[str, Any]:
    point = _point(receipt, 4.0)
    return {
        "label": label,
        "run_id": receipt["run_id"],
        "artifact_sha256": receipt["model"]["artifact_sha256"],
        "precision": receipt["model"]["precision"],
        "arrival_rate_qps": point["arrival_rate_qps"],
        "requests": point["requests"],
        "task_success": point["verifier_task_success_rate"],
        "stable": point["stable"],
        "ttft_p50_s": point["client"]["ttft"]["p50_s"],
        "e2e_p50_s": point["client"]["e2e"]["p50_s"],
        "e2e_p95_s": point["client"]["e2e"]["p95_s"],
        "output_tokens_per_s": point["client"]["output_tokens_per_s"],
        "cost_per_1k_successful_tasks_usd": point["cost_per_1k_successful_tasks_usd"],
        "vram_peak_mib": point["vram"]["peak_mib"],
    }


def _phase4_payload() -> dict[str, Any]:
    bf16 = _json(RESULTS / "phase4/raw/phase4_serve_r1b_bf16_v2.json")
    gptq = _json(RESULTS / "phase4/raw/phase4_serve_r1b_gptq_int4.json")
    baseline = _json(RESULTS / "phase4/raw/phase4_spec_decode_r1b_bf16_baseline_v2.json")
    mtp = _json(RESULTS / "phase4/raw/phase4_spec_decode_r1b_bf16_native_mtp.json")
    speculative = []
    for base_point in baseline["metrics"]["points"]:
        spec_point = _point(mtp, float(base_point["arrival_rate_qps"]))
        base_p95 = base_point["client"]["e2e"]["p95_s"]
        spec_p95 = spec_point["client"]["e2e"]["p95_s"]
        base_rate = base_point["client"]["successful_task_throughput_per_s"]
        spec_rate = spec_point["client"]["successful_task_throughput_per_s"]
        speculative.append(
            {
                "qps": base_point["arrival_rate_qps"],
                "baseline_p95_s": base_p95,
                "native_mtp_p95_s": spec_p95,
                "p95_delta_s": spec_p95 - base_p95,
                "baseline_successful_tasks_per_s": base_rate,
                "native_mtp_successful_tasks_per_s": spec_rate,
                "acceptance_rate": spec_point["server"]["speculative"]["acceptance_rate"],
                "verdict": ("win" if spec_p95 <= base_p95 and spec_rate >= base_rate else "lose"),
            }
        )
    structured = []
    for backend in ("xgrammar", "outlines"):
        receipt = _json(RESULTS / f"phase4/raw/phase4_structured_r1b_bf16_{backend}.json")
        item = receipt["metrics"]["constraint_tax_and_mitigation"]
        structured.append({"backend": backend, "run_id": receipt["run_id"], **item})
    return {
        "serving_at_4_qps": [
            _serving_point(bf16, label="R1b BF16"),
            _serving_point(gptq, label="R1b GPTQ-int4"),
            _serving_point(mtp, label="R1b BF16 + native MTP"),
        ],
        "speculative_boundary": {
            "method": "model-native MTP",
            "transition": "0.25 QPS lose; 0.50-4.00 QPS win",
            "points": speculative,
        },
        "structured_output": structured,
    }


def _phase5_payload() -> dict[str, Any]:
    receipt = _json(RESULTS / "phase5/raw/phase5_gateway_bench.json")
    pairs = receipt["metrics"]["direct_gateway"]["pairs"]
    stable = [
        pair
        for pair in pairs
        if float(pair["direct"]["error_rate"]) <= 0.05 + 1e-12
        and float(pair["gateway"]["error_rate"]) <= 0.05 + 1e-12
    ]
    nonstable = [pair for pair in pairs if pair not in stable]
    overload = []
    for pair in receipt["metrics"]["overload"]["pairs"]:
        direct = pair["direct"]
        gateway = pair["gateway"]
        overload.append(
            {
                "multiplier": pair["multiplier"],
                "offered_qps": pair["offered_qps"],
                "direct_error_rate": direct["error_rate"],
                "gateway_error_rate": gateway["error_rate"],
                "direct_all_response_p95_s": direct["client"]["e2e_all_responses"]["p95_s"],
                "gateway_all_response_p95_s": gateway["client"]["e2e_all_responses"]["p95_s"],
                "gateway_success_p95_s": gateway["client"]["e2e_success"]["p95_s"],
                "subsecond_error_response_p50_s": gateway["client"]["fast_reject"]["p50_s"],
                "queue_depth_max": gateway["gateway_samples"]["queue_depth_max"],
                "http_status_counts": gateway["http_status_counts"],
                "error_codes": gateway["error_semantics"]["error_codes"],
                "routing_decisions": gateway["gateway"]["routing_decisions"],
                "direct_recovery_s": direct["recovery_time_s"],
                "gateway_recovery_s": gateway["recovery_time_s"],
            }
        )
    nonstable_errors = [float(pair["gateway"]["error_rate"]) for pair in nonstable]
    return {
        "run_id": receipt["run_id"],
        "capacity_qps": receipt["metrics"]["capacity"]["measured_capacity_qps"],
        "max_stable_concurrency": receipt["metrics"]["capacity"]["max_stable_concurrency"],
        "stable_pair_count": len(stable),
        "stable_median_e2e_p50_overhead_pct": median(
            pair["overhead"]["e2e_p50_overhead_pct"] for pair in stable
        ),
        "stable_median_e2e_p95_overhead_pct": median(
            pair["overhead"]["e2e_p95_overhead_pct"] for pair in stable
        ),
        "stable_median_throughput_delta_pct": median(
            pair["overhead"]["throughput_delta_pct"] for pair in stable
        ),
        "nonstable_cells": [
            {
                "length_profile": pair["length_profile"],
                "concurrency": pair["concurrency"],
                "direct_error_rate": pair["direct"]["error_rate"],
                "gateway_error_rate": pair["gateway"]["error_rate"],
                "success_only_p95_overhead_pct": pair["overhead"]["e2e_p95_overhead_pct"],
                "interpretation": "survivor-biased; not a latency win",
            }
            for pair in nonstable
        ],
        "nonstable_gateway_error_rate_range": [min(nonstable_errors), max(nonstable_errors)],
        "paired_bare_vllm_error_rate_range": [
            min(float(pair["direct"]["error_rate"]) for pair in nonstable),
            max(float(pair["direct"]["error_rate"]) for pair in nonstable),
        ],
        "overload": overload,
        "known_limitation": (
            "Measured overload errors were admitted primary requests returning HTTP "
            "502/upstream_error, not designed 429 admission fast rejects; every overload "
            "cell recorded reject_overload=0. The connection-handling defect remains "
            "uncorrected in this measured build. Lower p95 in error-bearing cells is "
            "conditional on failed work and is not an unconditional win."
        ),
    }


def _exports_payload() -> dict[str, Any]:
    original = _json(RESULTS / "phase3_export_manifest_r1b_trl_s0.json")
    mtp = _json(RESULTS / "phase4/r1b_mtp_reexport_manifest.json")
    return {
        "bf16": original["full_precision_export"],
        "gptq_int4": original["deployment_int4_export"],
        "bf16_mtp_preserved": mtp["full_precision_export"],
        "source_adapter": mtp["source_adapter"],
        "preserved_mtp": mtp["preserved_mtp"],
    }


def build_payload(source_manifest_sha256: str) -> dict[str, Any]:
    records = _records()
    return {
        "schema_version": "frontier-forge-release-v1",
        "phase": 6,
        "provenance": {
            "source_manifest": relative_path(SOURCE_MANIFEST),
            "source_manifest_sha256": source_manifest_sha256,
            "dataset_hash": records["r1b_sft_rule_20k_s0"]["dataset_hash"],
            "bootstrap_resamples": 1000,
        },
        "training": _phase3_payload(records),
        "serving": _phase4_payload(),
        "gateway": _phase5_payload(),
        "exports": _exports_payload(),
    }


def build_handoff(payload: dict[str, Any]) -> dict[str, Any]:
    ingest = _json(REPO_ROOT / "data/ingest/manifest.json")
    cal = next(item for item in ingest["sources"] if item["source_split"] == "cal")
    native_mtp = next(
        item
        for item in payload["serving"]["serving_at_4_qps"]
        if item["label"] == "R1b BF16 + native MTP"
    )
    return {
        "schema_version": "frontier-forge-cascade-handoff-v1",
        "source_repository": "https://github.com/LucisZhang/frontier-forge",
        "source_manifest_sha256": payload["provenance"]["source_manifest_sha256"],
        "shared_frozen_cal": {
            "rows": cal["rows"],
            "source_split_sha256": cal["sha256"],
            "membership": "exact nlp-eval-lab CAL membership",
        },
        "quality": payload["training"]["headline"],
        "service_profile": native_mtp,
        "exports": payload["exports"],
        "integration_contract": {
            "status": "scenario-only-without-joint-cal-predictions",
            "terminal_failure_rate_for_cal_scenario": 1.0 - native_mtp["task_success"],
            "terminal_cost_per_request_usd_conservative": (
                native_mtp["cost_per_1k_successful_tasks_usd"] / 1000.0
            ),
            "threshold_input": "committed nlp-eval-lab CAL risk-coverage table",
            "warning": (
                "R1b consumes source_product/source_issue/source_company under input "
                "contract v2 and solves structured ticket/action policy, whereas the "
                "upstream cascade predicts the product class from narrative text. This "
                "handoff does not establish R1b as a leakage-free replacement classifier "
                "and contains no joint per-row CAL predictions. Any re-optimized threshold "
                "is an explicitly labeled scenario, not a certified cascade result."
            ),
        },
    }


def _write_payloads() -> None:
    source = build_source_manifest()
    write_json_atomic(SOURCE_MANIFEST, source)
    source_sha = sha256_file(SOURCE_MANIFEST)
    payload = build_payload(source_sha)
    handoff = build_handoff(payload)
    write_json_atomic(HEADLINE_PATH, payload)
    write_json_atomic(DEMO_DATA_PATH, payload)
    DEMO_DATA_JS.parent.mkdir(parents=True, exist_ok=True)
    DEMO_DATA_JS.write_bytes(_javascript_bytes(payload))
    write_json_atomic(HANDOFF_PATH, handoff)


def _build_release_manifest() -> dict[str, Any]:
    missing = [relative_path(path) for path in RELEASE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"release files missing: {missing}")
    return {
        "version": 1,
        "phase": 6,
        "files": [
            {"path": relative_path(path), "sha256": sha256_file(path)} for path in RELEASE_FILES
        ],
    }


def write_release() -> dict[str, Any]:
    _write_payloads()
    manifest = _build_release_manifest()
    write_json_atomic(RELEASE_MANIFEST, manifest)
    return manifest


def verify_release() -> dict[str, Any]:
    source = verify_source_manifest()
    source_sha = sha256_file(SOURCE_MANIFEST)
    expected_payload = build_payload(source_sha)
    expected_handoff = build_handoff(expected_payload)
    expected = {
        HEADLINE_PATH: _json_bytes(expected_payload),
        DEMO_DATA_PATH: _json_bytes(expected_payload),
        DEMO_DATA_JS: _javascript_bytes(expected_payload),
        HANDOFF_PATH: _json_bytes(expected_handoff),
    }
    changed = [relative_path(path) for path, body in expected.items() if path.read_bytes() != body]
    if changed:
        raise RuntimeError(f"derived Phase 6 outputs are stale: {changed}")
    if not RELEASE_MANIFEST.is_file():
        raise FileNotFoundError(f"missing release manifest: {RELEASE_MANIFEST}")
    committed = _json(RELEASE_MANIFEST)
    current = _build_release_manifest()
    if committed != current:
        expected_hashes = {item["path"]: item["sha256"] for item in committed["files"]}
        current_hashes = {item["path"]: item["sha256"] for item in current["files"]}
        drift = sorted(
            path
            for path in expected_hashes.keys() | current_hashes.keys()
            if expected_hashes.get(path) != current_hashes.get(path)
        )
        raise RuntimeError(f"release file hash gate failed: {drift}")
    return {
        "source_files_verified": len(source["files"]),
        "derived_outputs_verified": len(expected),
        "release_files_verified": len(committed["files"]),
        "source_manifest_sha256": source_sha,
        "headline_sha256": sha256_file(HEADLINE_PATH),
    }


def build_demo() -> dict[str, Any]:
    verification = verify_release()
    sources = (
        REPO_ROOT / "demo/index.html",
        REPO_ROOT / "demo/assets/styles.css",
        REPO_ROOT / "demo/assets/app.js",
        DEMO_DATA_PATH,
        DEMO_DATA_JS,
    )
    for path in sources[:3]:
        text = path.read_text().lower()
        if "http://" in text or "https://" in text:
            raise RuntimeError(f"offline demo contains a network URL: {relative_path(path)}")
    temporary = REPO_ROOT / "demo/.dist-phase6.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "assets").mkdir(parents=True)
    (temporary / "data").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "demo/index.html", temporary / "index.html")
    shutil.copy2(REPO_ROOT / "demo/assets/styles.css", temporary / "assets/styles.css")
    shutil.copy2(REPO_ROOT / "demo/assets/app.js", temporary / "assets/app.js")
    shutil.copy2(DEMO_DATA_PATH, temporary / "data/release.json")
    shutil.copy2(DEMO_DATA_JS, temporary / "data/release.js")
    files = sorted(path for path in temporary.rglob("*") if path.is_file())
    build_manifest = {
        "version": 1,
        "offline": True,
        "files": [
            {
                "path": path.relative_to(temporary).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in files
        ],
        "release_verification": verification,
    }
    write_json_atomic(temporary / "build-manifest.json", build_manifest)
    if DEMO_DIST.exists():
        shutil.rmtree(DEMO_DIST)
    temporary.replace(DEMO_DIST)
    return build_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="regenerate committed release files")
    mode.add_argument("--demo-build", action="store_true", help="build the offline demo")
    args = parser.parse_args(argv)
    if args.write:
        result = write_release()
    elif args.demo_build:
        result = build_demo()
    else:
        result = verify_release()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
