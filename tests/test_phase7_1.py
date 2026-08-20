from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gateway.bench import phase7_1_bench as bench  # noqa: E402
from gateway.bench import phase7_1_report as report  # noqa: E402


def test_a10_config_is_an_exact_phase5_contract_with_new_hardware_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORGE_GPU_HOURLY_USD", "1.53")
    config = bench._load_config("configs/phase7_1/gateway_r1b_mtp_a10.yaml")
    phase5 = yaml.safe_load((REPO_ROOT / "configs/phase5/gateway_r1b_mtp.yaml").read_text())

    assert config["phase"] == "7.1"
    assert config["hardware"]["gpu_type"] == "NVIDIA A10"
    assert config["hardware"]["hourly_usd"] == 1.53
    assert config["pricing"] == {
        "assumed_cny_per_hour": 11.0,
        "assumed_cny_per_usd": 7.2,
        "rounded_usd_per_hour": 1.53,
    }
    for key in bench._CONTRACT_KEYS:
        assert config[key] == phase5[key]


def test_a10_config_rejects_wrong_runtime_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_GPU_HOURLY_USD", "1.52")
    with pytest.raises(RuntimeError, match="1.53"):
        bench._load_config("configs/phase7_1/gateway_r1b_mtp_a10.yaml")


def _cell(
    *,
    cell_id: str,
    requests: int,
    statuses: dict[str, int],
    reject_overload: int = 0,
    queue: float = 0,
) -> dict[str, Any]:
    return {
        "cell_id": cell_id,
        "requests": requests,
        "errors": sum(count for status, count in statuses.items() if int(status) >= 400),
        "http_status_counts": statuses,
        "gateway_samples": {"queue_depth_max": queue},
        "gateway": {
            "queue_high_watermark_process": queue,
            "routing_decisions": {"reject_overload": reject_overload},
        },
        "client": {"fast_reject": {"p50_s": 0.01 if reject_overload else None}},
    }


def _gate_inputs(*, gateway_5xx: int = 0, rejects: int = 2) -> tuple[dict[str, Any], ...]:
    matrix_pairs = []
    for index in range(9):
        matrix_pairs.append(
            {
                "length_profile": ("short", "mixed", "long")[index // 3],
                "concurrency": (1, 8, 32)[index % 3],
                "direct": _cell(cell_id=f"direct-{index}", requests=20, statuses={"200": 20}),
                "gateway": _cell(
                    cell_id=f"gateway-{index}",
                    requests=20,
                    statuses={"200": 20 - gateway_5xx, "502": gateway_5xx},
                ),
            }
        )
    overload_pairs = []
    request_rows = []
    for multiplier in (2.0, 3.0, 5.0):
        cell_id = f"overload-{multiplier:g}"
        overload_pairs.append(
            {
                "multiplier": multiplier,
                "direct": _cell(cell_id=f"direct-{cell_id}", requests=60, statuses={"200": 60}),
                "gateway": _cell(
                    cell_id=cell_id,
                    requests=60,
                    statuses={
                        "200": 60 - rejects - gateway_5xx,
                        "429": rejects,
                        "502": gateway_5xx,
                    },
                    reject_overload=rejects,
                    queue=4,
                ),
            }
        )
        request_rows.extend(
            {
                "cell_id": cell_id,
                "http_status": 429,
                "retry_after": "1",
                "error_code": "overloaded",
                "client_e2e_s": 0.01,
            }
            for _ in range(rejects)
        )
    return (
        {"pairs": matrix_pairs},
        {"pairs": overload_pairs},
        {"request_artifact": {"path": "unused"}},
        request_rows,
    )


def test_gate_passes_only_with_429_bounded_queue_and_same_box_5xx_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORGE_GPU_HOURLY_USD", "1.53")
    config = bench._load_config("configs/phase7_1/gateway_r1b_mtp_a10.yaml")
    matrix, overload, stage, rows = _gate_inputs()
    monkeypatch.setattr(bench, "_load_request_rows", lambda _stage: rows)

    gate = bench._evaluate_gate(config, matrix, overload, stage)

    assert gate["status"] == "pass"
    assert gate["checks"] == {
        "baseline_completed_before_gateway": True,
        "matched_matrix_receipts": True,
        "overload_receipts": True,
        "429_fast_reject_semantics": True,
        "bounded_queue": True,
        "upstream_5xx_parity": True,
        "non_admission_error_parity": True,
    }


def test_gate_fails_on_admitted_502s_or_missing_429s(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_GPU_HOURLY_USD", "1.53")
    config = bench._load_config("configs/phase7_1/gateway_r1b_mtp_a10.yaml")
    matrix, overload, stage, rows = _gate_inputs(gateway_5xx=4, rejects=0)
    monkeypatch.setattr(bench, "_load_request_rows", lambda _stage: rows)

    gate = bench._evaluate_gate(config, matrix, overload, stage)

    assert gate["status"] == "fail"
    assert gate["checks"]["429_fast_reject_semantics"] is False
    assert gate["checks"]["upstream_5xx_parity"] is False
    assert gate["checks"]["non_admission_error_parity"] is False


def test_readme_red_flag_section_replacement_is_bounded_and_idempotent() -> None:
    source = f"before\n{report.SECTION_START}\nold\n{report.SECTION_END}\nafter\n"
    replacement = "new section\n\n"

    once = report._replace_section(source, replacement)

    assert once == f"before\nnew section\n\n{report.SECTION_END}\nafter\n"
    assert "old" not in once


def test_remote_scripts_pin_mount_rate_order_and_no_public_dashboard_ports() -> None:
    provision = (REPO_ROOT / "scripts/remote/provision_phase7_1_a10.sh").read_text()
    launch = (REPO_ROOT / "scripts/remote/launch_phase7_1_vllm.sh").read_text()
    run = (REPO_ROOT / "scripts/remote/run_phase7_1_bench.sh").read_text()
    gateway = (REPO_ROOT / "scripts/remote/launch_phase7_1_gateway.sh").read_text()

    assert "FORGE_CONFIRM_FORMAT_DEVICE=/dev/vdb" in provision
    assert "/mnt/frontier-forge" in provision
    assert "br_netfilter" in provision
    assert "disable --now nvidia-fabricmanager.service" in provision
    assert "FORGE_GPU_HOURLY_USD=1.53" in launch
    assert "baseline must finish before the gateway starts" in run
    assert "--listen-host 127.0.0.1" in gateway
    assert (
        "security-group" not in provision.lower()
        or "no security-group operation" in provision.lower()
    )


def test_final_receipt_import_is_append_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = {
        "run_id": "phase7-test",
        "config_path": "config.yaml",
        "config_hash": "a" * 64,
        "workload": {"sha256": "b" * 64},
        "git_sha": "c" * 40,
        "model": {},
        "metrics": {},
        "gate": {"status": "fail"},
        "production_blocked": True,
        "cost": {},
        "started_at": "2026-08-20T00:00:00+00:00",
        "finished_at": "2026-08-20T01:00:00+00:00",
        "raw_artifact": "results/phase7_1/raw/test.json",
        "disclosure": {},
    }
    path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(bench, "RUNS_PATH", path)

    assert bench._append_run_record(receipt) is True
    assert bench._append_run_record(receipt) is False
    assert len([line for line in path.read_text().splitlines() if line]) == 1
    assert json.loads(path.read_text())["production_blocked"] is True
