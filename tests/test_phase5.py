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


def test_makefile_gateway_bench_is_not_a_stub() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text()
    assert "gateway-bench:" in makefile
    assert "gateway/bench/phase5_bench.py" in makefile
    stub_line = next(line for line in makefile.splitlines() if line.startswith("STUB_TARGETS"))
    assert "gateway-bench" not in stub_line


def test_remote_contract_matches_phase5_config() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs/phase5/gateway_r1b_mtp.yaml").read_text())
    readme = (REPO_ROOT / "gateway/README.md").read_text()

    assert config["direct_gateway"]["concurrency"] == [1, 8, 32]
    assert config["overload"]["multipliers"] == [2, 3, 5]
    assert "prompt/output-length" in readme
    assert "queue high-watermark" in readme
    assert REPORT.START_MARKER in readme
