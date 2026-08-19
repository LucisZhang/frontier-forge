from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BENCH = _load_script("phase5_bench", REPO_ROOT / "gateway/bench/phase5_bench.py")
REPORT = _load_script("phase5_report", REPO_ROOT / "gateway/bench/phase5_report.py")


def test_phase5_config_pins_remote_rate_and_mtp_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_GPU_HOURLY_USD", "0.30")
    config = BENCH._load_config("configs/phase5/gateway_r1b_mtp.yaml")

    assert config["phase"] == 5
    assert config["hardware"]["hourly_usd"] == 0.30
    assert config["server"]["speculative_method"] == "mtp"
    assert config["model"]["artifact_path"].endswith("merged_bf16_mtp_preserved")
    assert config["gateway"]["fallback_enabled"] is False
    assert config["overload"]["multipliers"] == [2, 3, 5]


def test_phase5_config_rejects_unpinned_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_GPU_HOURLY_USD", "0.31")
    with pytest.raises(RuntimeError, match="0.30"):
        BENCH._load_config("configs/phase5/gateway_r1b_mtp.yaml")


def test_phase5_length_selection_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_GPU_HOURLY_USD", "0.30")
    config = BENCH._load_config("configs/phase5/gateway_r1b_mtp.yaml")
    rows = [
        {
            "request_id": f"row-{index}",
            "input_token_target": target,
            "prompt_tokens": target,
            "messages": [{"role": "user", "content": f"prompt-{index}"}],
            "label": {"decision": "allow"},
            "max_tokens": 999,
        }
        for target in (800, 1400, 2000)
        for index in range(8)
    ]

    left = BENCH._select_rows(rows, config, profile="mixed", count=9, seed=17)
    right = BENCH._select_rows(rows, config, profile="mixed", count=9, seed=17)

    assert left == right
    assert [row["input_token_target"] for row in left] == [800, 1400, 2000] * 3
    assert [row["max_tokens"] for row in left] == [192, 256] * 4 + [192]
    assert len({row["phase5_request_id"] for row in left}) == len(left)


def test_paired_overhead_fails_closed_on_schedule_mismatch() -> None:
    direct = {
        "request_schedule_sha256": "a",
        "client": {
            "e2e_success": {"p50_s": 1.0, "p95_s": 2.0},
            "ttft": {"p50_s": 0.1, "p95_s": 0.2},
        },
        "successful_request_throughput_per_s": 1.0,
    }
    gateway = {
        **direct,
        "request_schedule_sha256": "b",
    }
    with pytest.raises(RuntimeError, match="not identical"):
        BENCH._paired_overhead(direct, gateway)


def test_paired_overhead_reports_directional_percentages() -> None:
    direct = {
        "request_schedule_sha256": "same",
        "client": {
            "e2e_success": {"p50_s": 1.0, "p95_s": 2.0},
            "ttft": {"p50_s": 0.1, "p95_s": 0.2},
        },
        "successful_request_throughput_per_s": 10.0,
    }
    gateway = {
        "request_schedule_sha256": "same",
        "client": {
            "e2e_success": {"p50_s": 1.1, "p95_s": 2.4},
            "ttft": {"p50_s": 0.11, "p95_s": 0.22},
        },
        "successful_request_throughput_per_s": 9.0,
    }

    result = BENCH._paired_overhead(direct, gateway)

    assert result["e2e_p50_overhead_pct"] == pytest.approx(10.0)
    assert result["e2e_p95_overhead_pct"] == pytest.approx(20.0)
    assert result["throughput_delta_pct"] == pytest.approx(-10.0)


def test_readme_result_markers_replace_idempotently() -> None:
    source = f"before\n{REPORT.START_MARKER}\nold\n{REPORT.END_MARKER}\nafter\n"
    block = f"{REPORT.START_MARKER}\nnew\n{REPORT.END_MARKER}"

    once = REPORT._replace_readme_block(source, block)
    twice = REPORT._replace_readme_block(once, block)

    assert once == twice
    assert "old" not in once
    assert once.count(REPORT.START_MARKER) == 1


def test_resume_claim_does_not_mislabel_admitted_errors_as_fast_rejects() -> None:
    receipt = {
        "metrics": {
            "overload": {
                "pairs": [
                    {
                        "multiplier": 5,
                        "gateway": {
                            "recovery_time_s": 4.485,
                            "error_rate": 14 / 60,
                            "gateway_samples": {"queue_depth_max": 10},
                            "client": {"fast_reject": {"p50_s": 0.005}},
                            "http_status_counts": {"200": 46, "502": 14},
                            "error_semantics": {"error_codes": {"upstream_error": 14}},
                        },
                        "direct": {"recovery_time_s": 4.649, "error_rate": 0.0},
                    }
                ]
            }
        }
    }

    claim = REPORT._resume_claim(receipt, {"p50_median_pct": 0.3})

    assert "通过 admission 的请求产生 HTTP 502/upstream_error 错误响应" in claim
    assert "错误率 23.3%（裸 vLLM 0.0%）" in claim
    assert "快速失败" not in claim
    assert "503 快速拒绝" not in claim


def _table_cell(*, error_rate: float) -> dict[str, Any]:
    return {
        "error_rate": error_rate,
        "client": {
            "ttft": {"p50_s": 0.1, "p95_s": 0.2},
            "itl": {"p50_s": 0.01, "p95_s": 0.02},
            "e2e_success": {"p50_s": 1.0, "p95_s": 2.0},
        },
        "successful_request_throughput_per_s": 1.0,
        "output_tokens_per_s": 10.0,
        "cost_per_1k_successful_tasks_usd": 0.1,
        "vram": {"peak_mib": 100.0},
    }


def test_nonstable_pair_is_marked_as_survivor_biased_not_a_latency_win() -> None:
    pair = {
        "length_profile": "long",
        "concurrency": 8,
        "direct": _table_cell(error_rate=0.0),
        "gateway": _table_cell(error_rate=0.85),
        "overhead": {
            "e2e_p50_overhead_pct": -35.6,
            "e2e_p95_overhead_pct": -54.8,
            "throughput_delta_pct": -35.9,
        },
    }

    table = REPORT._direct_gateway_table({"metrics": {"direct_gateway": {"pairs": [pair]}}})

    assert "NOT a latency win" in table
    assert "gateway error 85.0% vs direct 0.0%" in table


def test_overhead_summary_includes_exact_five_percent_error_boundary() -> None:
    receipt = {
        "metrics": {
            "direct_gateway": {
                "pairs": [
                    {
                        "direct": {"error_rate": 0.0},
                        "gateway": {"error_rate": 1.0 - 19.0 / 20.0},
                        "overhead": {
                            "e2e_p50_overhead_pct": 1.0,
                            "e2e_p95_overhead_pct": 2.0,
                            "ttft_p50_overhead_pct": 3.0,
                            "throughput_delta_pct": -1.0,
                        },
                    }
                ]
            }
        }
    }

    summary = REPORT._overhead_summary(receipt)

    assert summary["stable_pairs"] == 1
    assert summary["p95_median_pct"] == 2.0


def test_makefile_gateway_bench_is_not_a_stub() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text()
    assert "gateway-bench:" in makefile
    assert "gateway/bench/phase5_bench.py" in makefile
    stub_lines = [line for line in makefile.splitlines() if line.startswith("STUB_TARGETS")]
    assert all("gateway-bench" not in line for line in stub_lines)


def test_remote_contract_matches_phase5_config() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs/phase5/gateway_r1b_mtp.yaml").read_text())
    readme = (REPO_ROOT / "gateway/README.md").read_text()

    assert config["direct_gateway"]["concurrency"] == [1, 8, 32]
    assert config["overload"]["multipliers"] == [2, 3, 5]
    assert "prompt/output-length" in readme
    assert "queue high-watermark" in readme
    assert REPORT.START_MARKER in readme
