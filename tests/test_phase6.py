from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from forge import release
from forge.train.config import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_release_headline_is_derived_from_ledger_and_paired_receipt() -> None:
    payload = release.build_payload("f" * 64)
    headline = payload["training"]["headline"]

    assert headline["run_id"] == "r1b_sft_rule_20k_s0"
    assert headline["training_seeds"] == [0]
    assert headline["task_success"] == pytest.approx(0.9905)
    assert headline["ci95"] == [0.986, 0.9945]
    assert headline["paired_delta_vs_r1"]["mean_task_success_delta"] == pytest.approx(0.327)
    assert headline["paired_delta_vs_r1"]["ci95"] == [0.306, 0.345]
    assert payload["training"]["r4_v2"]["status"] == "aborted-zero-reward-variance"


def test_project_spend_includes_negative_attempts_and_all_teacher_api_receipts() -> None:
    spend = release.build_payload("f" * 64)["project_spend"]

    assert spend["gpu_components"]["phase3"]["receipt_rows"] == 21
    assert spend["gpu_components"]["phase4"]["receipt_rows"] == 13
    assert spend["gpu_components"]["phase5"]["receipt_rows"] == 1
    assert spend["teacher_api_receipt_rows"] == 5
    assert spend["gpu_hours"] == pytest.approx(37.58140120110837)
    assert spend["gpu_usd"] == pytest.approx(11.274420360332511)
    assert spend["teacher_api_usd"] == pytest.approx(13.038138)
    assert spend["total_usd"] == pytest.approx(24.31255836033251)


def test_n20_serving_points_disclose_wilson_interval_fragility() -> None:
    points = release.build_payload("f" * 64)["serving"]["serving_at_4_qps"]

    assert len(points) == 3
    assert all(point["requests"] == 20 for point in points)
    assert all(point["verifier_successes"] == 19 for point in points)
    assert all(
        point["task_success_wilson95"] == pytest.approx([0.763868806553258, 0.9911185511992047])
        for point in points
    )


def test_release_copy_preserves_reviewed_license_and_disclosure_language() -> None:
    model_card = (ROOT / "MODEL_CARD.md").read_text()
    readme = (ROOT / "README.md").read_text()
    license_text = (ROOT / "LICENSE").read_text()

    assert "license: apache-2.0" in model_card
    assert "7/50 wrong" in model_card
    assert "deliberately enriched 50-row strong-action" in model_card
    assert "not comparable to the earlier v2 4% figure" in model_card
    assert "200-row stratified human audit" not in model_card
    assert "CFPB Consumer Complaint Database" in model_card
    assert license_text.startswith("                                 Apache License\n")
    assert "Version 2.0, January 2004" in license_text

    assert "single training seed (seed 0)" in readme
    assert "τ=0.8484" in readme
    assert "τ=0.8483569229" not in readme
    assert "95% Wilson interval is **[76.4%,\n99.1%]**" in readme
    assert "100% at $0 model-inference cost" in readme
    assert readme.count("$24.313") == 1


def test_gateway_payload_preserves_the_overload_failure_semantics() -> None:
    gateway = release.build_payload("f" * 64)["gateway"]

    assert gateway["nonstable_gateway_error_rate_range"] == pytest.approx([0.1, 0.85])
    assert gateway["paired_bare_vllm_error_rate_range"] == [0.0, 0.0]
    assert {tuple(row["error_codes"]) for row in gateway["overload"]} == {("upstream_error",)}
    assert all(row["http_status_counts"].get("502", 0) > 0 for row in gateway["overload"])
    assert all(row["routing_decisions"]["reject_overload"] == 0 for row in gateway["overload"])
    assert "not designed 429 admission fast rejects" in gateway["known_limitation"]
    assert all(
        row["interpretation"] == "survivor-biased; not a latency win"
        for row in gateway["nonstable_cells"]
    )


def test_phase7_1_payload_copies_the_sustained_gate_and_highest_load_statuses() -> None:
    receipt = json.loads(
        (ROOT / "results/phase7_1/raw/phase7_1_sustained_gateway_bench.json").read_text()
    )
    phase7_1 = release.build_payload("f" * 64)["phase7_1"]
    highest_gate_cell = max(
        receipt["gate"]["sustained_overload_cells"], key=lambda cell: cell["offered_qps"]
    )
    highest_load_cells = [
        cell
        for cell in receipt["metrics"]["sustained_overload"]["cells"]
        if cell["arrival_rate_qps"] == highest_gate_cell["offered_qps"]
    ]
    by_endpoint = {cell["endpoint"]: cell for cell in highest_load_cells}

    assert phase7_1["run_id"] == receipt["run_id"]
    assert phase7_1["status"] == receipt["status"]
    assert phase7_1["production_blocked"] == receipt["production_blocked"]
    assert phase7_1["gate"] == receipt["gate"]
    assert phase7_1["highest_load"]["multiplier"] == highest_gate_cell["multiplier"]
    assert phase7_1["highest_load"]["offered_qps"] == highest_gate_cell["offered_qps"]
    assert (
        phase7_1["highest_load"]["bare_vllm_http_status_counts"]
        == by_endpoint["direct"]["http_status_counts"]
    )
    assert (
        phase7_1["highest_load"]["gateway_http_status_counts"]
        == by_endpoint["gateway"]["http_status_counts"]
    )


def test_phase7_2_payload_copies_scaling_cold_start_canary_and_alert_receipts() -> None:
    raw = ROOT / "results/phase7_2/raw"
    scaling = json.loads((raw / "gateway_keda_scale.json").read_text())
    cold_start = json.loads((raw / "gpu_cold_start.json").read_text())
    canary = json.loads((raw / "canary_release.json").read_text())
    alert_receipts = [
        json.loads((raw / name).read_text())
        for name in (
            "alert_ForgeAvailabilityBurnRate.json",
            "alert_ForgeLatencyBurnRate.json",
        )
    ]
    phase7_2 = release.build_payload("f" * 64)["phase7_2"]

    assert phase7_2["gateway_scaling"] == {
        key: scaling[key]
        for key in (
            "status",
            "before",
            "after",
            "max_ready_replicas",
            "max_queue_depth",
            "max_queue_high_watermark",
            "k6_counts",
            "scaled_down",
            "checks",
        )
    }
    assert phase7_2["gpu_cold_start"] == {
        key: cold_start[key]
        for key in ("status", "iterations_completed", "iterations_required", "distribution_s")
    }
    assert phase7_2["canary_rollout"]["status"] == canary["status"]
    assert phase7_2["canary_rollout"]["checks"] == canary["checks"]
    assert phase7_2["canary_rollout"]["promotion_stages"] == [
        {
            key: stage[key]
            for key in ("variant", "counts", "http_statuses", "request_concurrency", "slo_guard")
        }
        for stage in canary["promotion_stages"]
    ]
    assert phase7_2["canary_rollout"]["rollback_passed"] == canary["rollback"]["pass"]
    assert phase7_2["alerts"] == {
        receipt["alert"]: {
            key: receipt[key] for key in ("status", "injected_fault", "k6_counts", "firing_payload")
        }
        for receipt in alert_receipts
    }


def test_cascade_handoff_refuses_to_promote_the_model_to_a_certified_classifier() -> None:
    payload = release.build_payload("f" * 64)
    handoff = release.build_handoff(payload)
    contract = handoff["integration_contract"]

    assert contract["status"] == "scenario-only-without-joint-cal-predictions"
    assert contract["terminal_failure_rate_for_cal_scenario"] == pytest.approx(0.05)
    assert "leakage-free replacement classifier" in contract["warning"]
    assert handoff["shared_frozen_cal"]["rows"] == 86_972


def test_source_manifest_covers_every_declared_source_and_matches_bytes() -> None:
    committed = json.loads(release.SOURCE_MANIFEST.read_text())
    current = release.build_source_manifest()

    assert committed == current
    assert {row["path"] for row in committed["files"]} == {
        release.relative_path(path) for path in release.SOURCE_PATHS
    }
    assert current["phase"] == 7
    assert {
        "results/phase7_1/raw/phase7_1_sustained_gateway_bench.json",
        "results/phase7_2/raw/gateway_keda_scale.json",
        "results/phase7_2/raw/gpu_cold_start.json",
        "results/phase7_2/raw/canary_release.json",
        "results/phase7_2/raw/alert_ForgeAvailabilityBurnRate.json",
        "results/phase7_2/raw/alert_ForgeLatencyBurnRate.json",
    }.issubset({row["path"] for row in current["files"]})


def test_release_manifest_and_derived_outputs_are_green() -> None:
    result = release.verify_release()

    assert result["source_files_verified"] == len(release.SOURCE_PATHS)
    assert result["derived_outputs_verified"] == 4
    assert result["release_files_verified"] == len(release.RELEASE_FILES)
    assert result["headline_sha256"] == sha256_file(release.HEADLINE_PATH)


def test_hf_archive_and_remote_disk_receipts_close_the_phase6_archive_gate() -> None:
    archive = json.loads((ROOT / "results/phase6/hf_archive_receipt.json").read_text())
    disk = json.loads((ROOT / "results/phase6/remote_disk_audit.json").read_text())

    assert archive["repo_id"] == "Luciss007/frontier-forge-r1b"
    assert archive["private"] is False
    assert archive["verified_artifact_commit"] == ("fd4ae1e1989dcb1641a496bf796031491518983e")
    assert archive["receipt_commit"] == "a717e9c50435fc81b795d5683a22d0efe8191d16"
    assert {item["name"]: item["tree_sha256"] for item in archive["variants"]} == {
        "bf16": "7cf43a2905513f61797b78b7e3fd7ebdacd1cba4fc89abea9ce209401e6e6435",
        "gptq_int4": "c99b42cf0e062cc75f2df8588725d0c29383666f3db0c1ae837ce15bfe6d39d2",
        "bf16_mtp_preserved": ("7878b55f6fe6a9ecb12b9504b1a88d7bc6fef7ba72d91289b6e8d694f6bc75ce"),
    }
    assert all(
        item["file_count"] == sum(item["remote_verification"].values())
        for item in archive["variants"]
    )
    assert disk["remote_only_durable_assets"] == []
    assert disk["credential_cleanup"]["active_hf_token_present_after_logout"] is False
    assert all(
        check["exit_code"] == 0 and check["differences"] == [] for check in disk["sync_checks"]
    )
    log_bundle = ROOT / disk["local_evidence"]["archive_log_bundle"]["path"]
    assert sha256_file(log_bundle) == disk["local_evidence"]["archive_log_bundle"]["sha256"]


def test_demo_sources_are_network_free_and_javascript_is_parseable() -> None:
    for path in (
        ROOT / "demo/index.html",
        ROOT / "demo/assets/styles.css",
        ROOT / "demo/assets/app.js",
    ):
        text = path.read_text().lower()
        assert "http://" not in text
        assert "https://" not in text

    subprocess.run(
        ["node", "--check", str(ROOT / "demo/assets/app.js")],
        check=True,
        capture_output=True,
        text=True,
    )


def test_makefile_phase6_targets_are_real_and_smoke_is_guarded() -> None:
    dry = subprocess.run(
        ["make", "--no-print-directory", "--dry-run", "reproduce-headline", "SMOKE=1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "forge.release" in dry.stdout
    assert "[stub]" not in dry.stdout

    smoke_dry = subprocess.run(
        ["make", "--no-print-directory", "--dry-run", "phase6-smoke", "SMOKE=1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "FORGE_SMOKE_OUTPUT_ROOT=" in smoke_dry.stdout
    assert "FORGE_PHASE4_SMOKE_OUTPUT_ROOT=" in smoke_dry.stdout

    blocked = subprocess.run(
        ["make", "--no-print-directory", "phase6-smoke", "SMOKE=0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode != 0
    assert "requires SMOKE=1" in blocked.stderr


def test_demo_build_rebuilds_and_reseals_release_before_copying_dist() -> None:
    dry = subprocess.run(
        ["make", "--no-print-directory", "--dry-run", "demo-build"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    write_index = dry.stdout.index("forge.release --write")
    build_index = dry.stdout.index("forge.release --demo-build")
    assert write_index < build_index
