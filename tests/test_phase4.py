from __future__ import annotations

import asyncio
import json
import subprocess
import tomllib
from itertools import pairwise
from pathlib import Path

import pytest

from forge.bench.config import (
    load_phase4_config,
    phase4_config_paths,
    workload_contract_hash,
)
from forge.bench.loadgen import poisson_offsets, run_load_benchmark
from forge.bench.metrics import parse_prometheus, prometheus_delta, summarize_vllm_metrics
from forge.bench.report import _spec_svg
from forge.bench.server_args import server_command
from forge.bench.smoke_server import SmokeServer
from forge.bench.structured import run_structured_benchmark
from forge.bench.workload import _allocation, build_workload, load_workload

ROOT = Path(__file__).resolve().parents[1]


def test_phase4_configs_cover_the_locked_matrix() -> None:
    configs = [load_phase4_config(path) for path in phase4_config_paths()]

    assert len(configs) == 8
    assert len({config["run_id"] for config in configs}) == 8
    assert {config["experiment"] for config in configs} == {
        "serve",
        "spec_decode",
        "structured",
    }
    serve = [config for config in configs if config["experiment"] == "serve"]
    assert {(config["model"]["variant"], config["model"]["precision"]) for config in serve} == {
        ("r1b", "bfloat16"),
        ("r1b", "gptq_int4"),
        ("r3_equivalent_legacy_r4_zero_update", "bfloat16"),
        ("r3_equivalent_legacy_r4_zero_update", "gptq_int4"),
    }
    assert {
        config["structured"]["backend"]
        for config in configs
        if config["experiment"] == "structured"
    } == {
        "xgrammar",
        "outlines",
    }
    assert {workload_contract_hash(config) for config in configs} == {
        workload_contract_hash(configs[0])
    }
    assert {config["hardware"]["hourly_usd"] for config in configs} == {0.30}


def test_phase4_quantization_and_r3_equivalence_are_not_conflated() -> None:
    configs = [load_phase4_config(path) for path in phase4_config_paths()]
    serve = [config for config in configs if config["experiment"] == "serve"]

    for config in serve:
        assert config["model"]["training_time_quantization"] == "qlora_nf4"
        expected = "gptq_int4" if config["model"]["precision"] == "gptq_int4" else "none"
        assert config["model"]["deployment_quantization"] == expected
    r3eq = [config for config in serve if config["model"].get("comparison_only")]
    assert len(r3eq) == 2
    for config in r3eq:
        evidence = config["model"]["equivalence_evidence"]
        assert evidence["adapter_weights_sha256"] == (
            "0188166e07cef16097267da5a68f45bfa3c54d08c68c25cd22b1f5649da37f66"
        )
        assert "zero gradient" in evidence["interpretation"]


def test_phase4_remote_serve_environment_is_pinned_and_conflicted() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert project["dependency-groups"]["remote-serve"] == ["vllm==0.17.0; sys_platform == 'linux'"]
    conflicts = project["tool"]["uv"]["conflicts"]
    assert [{"group": "train"}, {"group": "remote-serve"}] in conflicts
    assert [{"group": "unsloth-train"}, {"group": "remote-serve"}] in conflicts
    assert [{"group": "remote-reference"}, {"group": "remote-serve"}] in conflicts


def test_weighted_workload_and_poisson_schedule_are_fixed_seed() -> None:
    assert _allocation(11, [0.4, 0.4, 0.2]) == [5, 4, 2]
    first = poisson_offsets(count=8, qps=2.0, seed=20260818)
    second = poisson_offsets(count=8, qps=2.0, seed=20260818)

    assert first == second
    assert first[0] == 0.0
    assert all(left <= right for left, right in pairwise(first))


def test_prometheus_metrics_remain_server_side_and_include_spec_acceptance() -> None:
    before = parse_prometheus(
        """
vllm:time_to_first_token_seconds_count 2
vllm:time_to_first_token_seconds_sum 0.2
vllm:time_to_first_token_seconds_bucket{le="0.1"} 1
vllm:time_to_first_token_seconds_bucket{le="0.5"} 2
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 2
vllm:spec_decode_num_draft_tokens_total 10
vllm:spec_decode_num_accepted_tokens_total 5
vllm:spec_decode_num_drafts_total 4
"""
    )
    after = parse_prometheus(
        """
vllm:time_to_first_token_seconds_count 4
vllm:time_to_first_token_seconds_sum 0.6
vllm:time_to_first_token_seconds_bucket{le="0.1"} 2
vllm:time_to_first_token_seconds_bucket{le="0.5"} 4
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 4
vllm:spec_decode_num_draft_tokens_total 30
vllm:spec_decode_num_accepted_tokens_total 17
vllm:spec_decode_num_drafts_total 10
"""
    )
    summary = summarize_vllm_metrics(prometheus_delta(before, after))

    assert summary["measurement_side"] == "server_prometheus"
    assert summary["ttft"]["count"] == 2
    assert summary["speculative"]["acceptance_rate"] == pytest.approx(0.6)
    assert summary["speculative"]["mean_acceptance_length"] == pytest.approx(3.0)


def test_vllm_commands_pin_precision_speculation_and_structured_backends() -> None:
    int4 = server_command("configs/phase4/serve_r1b_gptq_int4.yaml", executable="vllm")
    spec = server_command("configs/phase4/spec_r1b_bf16_qwen05b.yaml", executable="vllm")
    outlines = server_command("configs/phase4/structured_r1b_bf16_outlines.yaml", executable="vllm")

    assert int4[int4.index("--quantization") + 1] == "gptq"
    assert "--language-model-only" in int4
    spec_value = json.loads(spec[spec.index("--speculative-config") + 1])
    assert spec_value == {
        "method": "draft_model",
        "model": "Qwen/Qwen2.5-0.5B",
        "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "num_speculative_tokens": 5,
    }
    structured = json.loads(outlines[outlines.index("--structured-outputs-config") + 1])
    assert structured == {"backend": "outlines", "disable_fallback": True}


def test_local_smoke_loadgen_separates_timings_and_uses_verifier() -> None:
    path = "configs/phase4/serve_r1b_bf16.yaml"
    config = load_phase4_config(path)
    build_workload(path, smoke=True)
    workload = load_workload(config, smoke=True)

    with SmokeServer(workload, model=config["model"]["served_name"], speculative=False) as server:
        points, requests = asyncio.run(
            run_load_benchmark(config, base_url=server.base_url, smoke=True)
        )

    assert len(points) == 1
    assert points[0]["client"]["measurement_side"] == "client_wall_clock_streaming"
    assert points[0]["server"]["measurement_side"] == "server_prometheus"
    assert points[0]["verifier_successes"] == len(requests)
    assert points[0]["cost_per_1k_successful_tasks_usd"] is not None


def test_local_structured_smoke_exercises_tax_and_two_pass() -> None:
    path = "configs/phase4/structured_r1b_bf16_xgrammar.yaml"
    config = load_phase4_config(path)
    build_workload(path, smoke=True)
    workload = load_workload(config, smoke=True)

    with SmokeServer(workload, model=config["model"]["served_name"], speculative=False) as server:
        summary, records = asyncio.run(
            run_structured_benchmark(config, base_url=server.base_url, smoke=True)
        )

    tax = summary["constraint_tax_and_mitigation"]
    assert summary["backend"] == "xgrammar"
    assert len(summary["schema_compile"]) == 4
    assert tax["unconstrained_tool_call_rate"] == 1.0
    assert tax["simultaneous_schema_tool_call_rate"] == 0.0
    assert tax["two_pass_task_success"] == 1.0
    assert len(records) == 2


def test_spec_boundary_svg_is_deterministic_and_marks_both_verdicts() -> None:
    rows = [
        {"qps": 1.0, "p95_delta_s": -0.1, "verdict": "win"},
        {"qps": 2.0, "p95_delta_s": 0.2, "verdict": "lose"},
    ]

    first = _spec_svg(rows)
    second = _spec_svg(rows)

    assert first == second
    assert "green = win" in first
    assert "orange = lose" in first


def test_phase4_remote_scripts_are_safe_for_the_shared_pod() -> None:
    names = (
        "bootstrap_phase4.sh",
        "launch_phase4.sh",
        "run_phase4.sh",
        "sync_phase4.sh",
    )
    for name in names:
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts/remote" / name)],
            check=True,
            capture_output=True,
            text=True,
        )
    launcher = (ROOT / "scripts/remote/launch_phase4.sh").read_text()
    worker = (ROOT / "scripts/remote/run_phase4.sh").read_text()
    bootstrap = (ROOT / "scripts/remote/bootstrap_phase4.sh").read_text()
    combined = launcher + worker

    assert "nvidia-smi pmon -c 1" in launcher
    assert "nvidia-smi pmon -c 1" in worker
    assert 'session="forge-phase4"' in launcher
    assert "waiting without launching" in worker
    assert "sleep 60" in worker
    assert 'kill -TERM -- "-${server_pid}"' in worker
    assert "FORGE_GPU_HOURLY_USD" in combined
    assert "0.30" in combined
    assert "uv sync --active --locked --no-default-groups --group remote-serve" in bootstrap
    assert "shutdown" not in combined.lower()
    assert "reboot" not in combined.lower()


def test_phase4_report_code_is_the_only_phase4_numbered_document_writer() -> None:
    makefile = (ROOT / "Makefile").read_text()
    worker = (ROOT / "scripts/remote/run_phase4.sh").read_text()

    assert "forge.bench.report" in makefile
    assert "make bench-report" in worker
    assert "results/phase4_serving_report.md" in (ROOT / "src/forge/bench/report.py").read_text()
