from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gateway.bench import phase7_1_sustained as bench  # noqa: E402
from gateway.bench import phase7_1_sustained_report as report  # noqa: E402


def test_sustained_config_extends_but_does_not_change_finite_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORGE_GPU_HOURLY_USD", "1.53")
    config = bench._load_config("configs/phase7_1/sustained_overload_a10.yaml")

    assert config["run_id"] == "phase7_1_sustained_overload_r1b_bf16_native_mtp_a10"
    assert config["sustained_overload"] == {
        "multipliers": [2, 3, 5],
        "minimum_arrival_duration_s": 120,
        "max_client_concurrency": 768,
        "seed_offset": 7000,
    }
    assert config["overload"]["measurement_requests"] == 60
    assert config["model"]["artifact_path"].startswith("checkpoints/")
    assert config["hardware"]["gpu_type"] == "NVIDIA A10"
    assert config["hardware"]["hourly_usd"] == 1.53


def test_duration_schedule_is_deterministic_and_crosses_minimum() -> None:
    left = bench._poisson_offsets_for_duration(duration_s=120, qps=4, seed=7)
    right = bench._poisson_offsets_for_duration(duration_s=120, qps=4, seed=7)

    assert left == right
    assert left[0] == 0
    assert left[-1] >= 120
    assert left[-2] < 120
    assert all(later > earlier for earlier, later in zip(left[:-1], left[1:], strict=True))


def _cell(
    *,
    cell_id: str,
    requests: int,
    statuses: dict[str, int],
    duration_s: float = 120.1,
    queue: float = 24,
    rejects: int = 0,
) -> dict[str, Any]:
    return {
        "cell_id": cell_id,
        "requests": requests,
        "http_status_counts": statuses,
        "arrival_duration_s": duration_s,
        "gateway_samples": {"queue_depth_max": queue},
        "gateway": {
            "queue_high_watermark_process": queue,
            "routing_decisions": {"reject_overload": rejects},
        },
    }


def _gate_inputs(
    *, queue: float = 24, duration_s: float = 120.1, gateway_5xx: int = 0
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    pairs = []
    request_rows = []
    for multiplier in (2.0, 3.0, 5.0):
        cell_id = f"phase7-1-sustained-{multiplier:g}x-gateway"
        pairs.append(
            {
                "multiplier": multiplier,
                "offered_qps": multiplier * 2,
                "direct": _cell(
                    cell_id=f"phase7-1-sustained-{multiplier:g}x-direct",
                    requests=100,
                    statuses={"200": 100},
                    duration_s=duration_s,
                ),
                "gateway": _cell(
                    cell_id=cell_id,
                    requests=100,
                    statuses={"200": 90 - gateway_5xx, "429": 10, "502": gateway_5xx},
                    duration_s=duration_s,
                    queue=queue,
                    rejects=10,
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
            for _ in range(10)
        )
    return (
        {"pairs": pairs, "schedules_sha256_verified": True},
        {"request_artifact": {"path": "unused"}},
        request_rows,
    )


def test_sustained_gate_requires_duration_saturation_429_and_5xx_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORGE_GPU_HOURLY_USD", "1.53")
    config = bench._load_config("configs/phase7_1/sustained_overload_a10.yaml")
    paired, stage, rows = _gate_inputs()
    monkeypatch.setattr(bench.finite, "_load_request_rows", lambda _stage: rows)

    gate = bench._evaluate_gate(config, paired, stage)

    assert gate["status"] == "pass"
    assert all(gate["checks"].values())


@pytest.mark.parametrize(
    ("queue", "duration_s", "gateway_5xx", "failed_check"),
    [
        (23, 120.1, 0, "queue_saturated_in_every_cell"),
        (24, 119.9, 0, "duration_at_least_120s_per_cell"),
        (24, 120.1, 10, "upstream_5xx_parity"),
    ],
)
def test_sustained_gate_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    queue: float,
    duration_s: float,
    gateway_5xx: int,
    failed_check: str,
) -> None:
    monkeypatch.setenv("FORGE_GPU_HOURLY_USD", "1.53")
    config = bench._load_config("configs/phase7_1/sustained_overload_a10.yaml")
    paired, stage, rows = _gate_inputs(queue=queue, duration_s=duration_s, gateway_5xx=gateway_5xx)
    monkeypatch.setattr(bench.finite, "_load_request_rows", lambda _stage: rows)

    gate = bench._evaluate_gate(config, paired, stage)

    assert gate["status"] == "fail"
    assert gate["checks"][failed_check] is False


def test_resolved_readme_replacement_accepts_legacy_and_is_idempotent() -> None:
    legacy = f"before\n{report.LEGACY_SECTION_START}\nold\n{report.SECTION_END}\nafter\n"
    replacement = f"{report.RESOLVED_SECTION_START}\nnew\n\n"

    once = report._replace_section(legacy, replacement)
    twice = report._replace_section(once, replacement)

    assert once == twice
    assert report.LEGACY_SECTION_START not in once
    assert "old" not in once


def test_sustained_remote_script_preserves_order_and_loopback_only() -> None:
    script = (REPO_ROOT / "scripts/remote/run_phase7_1_sustained.sh").read_text()

    assert "all sustained bare-vLLM cells must finish before the gateway starts" in script
    assert "FORGE_PHASE7_SESSION_STARTED_AT" in script
    assert "FORGE_GPU_HOURLY_USD=1.53" in script
    assert 'PYTHONPATH="${repo_root}/src:${repo_root}' in script
    assert "http://127.0.0.1:8000" in script
    assert "http://127.0.0.1:9000" in script
    assert "security-group" not in script.lower()
