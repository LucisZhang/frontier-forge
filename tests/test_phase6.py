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
    assert headline["task_success"] == pytest.approx(0.9905)
    assert headline["ci95"] == [0.986, 0.9945]
    assert headline["paired_delta_vs_r1"]["mean_task_success_delta"] == pytest.approx(0.327)
    assert headline["paired_delta_vs_r1"]["ci95"] == [0.306, 0.345]
    assert payload["training"]["r4_v2"]["status"] == "aborted-zero-reward-variance"


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


def test_release_manifest_and_derived_outputs_are_green() -> None:
    result = release.verify_release()

    assert result["source_files_verified"] == len(release.SOURCE_PATHS)
    assert result["derived_outputs_verified"] == 4
    assert result["release_files_verified"] == len(release.RELEASE_FILES)
    assert result["headline_sha256"] == sha256_file(release.HEADLINE_PATH)


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

    blocked = subprocess.run(
        ["make", "--no-print-directory", "phase6-smoke", "SMOKE=0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode != 0
    assert "requires SMOKE=1" in blocked.stderr
