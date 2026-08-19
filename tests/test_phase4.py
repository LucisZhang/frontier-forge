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
from forge.bench.loadgen import (
    normalize_verifier_input,
    poisson_offsets,
    run_load_benchmark,
)
from forge.bench.metrics import parse_prometheus, prometheus_delta, summarize_vllm_metrics
from forge.bench.mtp_reexport import _read_safetensors_header, _write_raw_safetensors, audit_index
from forge.bench.report import _spec_svg
from forge.bench.runner import validate_existing_receipt
from forge.bench.server_args import server_command
from forge.bench.smoke_server import SmokeServer
from forge.bench.structured import run_structured_benchmark
from forge.bench.system_load import SystemLoadSampler, effective_cpu_count, metrics_contaminated
from forge.bench.workload import _allocation, build_workload, load_workload
from forge.train.config import sha256_file

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
    spec = next(
        config
        for config in configs
        if config["experiment"] == "spec_decode" and config["speculative"]["enabled"]
    )
    assert spec["speculative"]["method"] == "mtp"
    assert spec["speculative"]["method_selection"]["selected"] == "native_mtp"
    assert spec["speculative"]["num_speculative_tokens"] == 1
    corrected = next(config for config in configs if config["run_id"] == "phase4_serve_r1b_bf16_v2")
    assert corrected["supersedes"]["run_id"] == "phase4_serve_r1b_bf16"
    assert corrected["supersedes"]["raw_artifact"].endswith("phase4_serve_r1b_bf16.json")


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
    spec = server_command("configs/phase4/spec_r1b_bf16_mtp.yaml", executable="vllm")
    outlines = server_command("configs/phase4/structured_r1b_bf16_outlines.yaml", executable="vllm")

    assert int4[int4.index("--quantization") + 1] == "gptq"
    assert int4[int4.index("--dtype") + 1] == "float16"
    assert spec[spec.index("--dtype") + 1] == "bfloat16"
    assert "--language-model-only" in int4
    assert "--no-enable-log-requests" in int4
    assert "--disable-log-requests" not in int4
    spec_value = json.loads(spec[spec.index("--speculative-config") + 1])
    assert spec_value == {
        "method": "mtp",
        "num_speculative_tokens": 1,
    }
    structured = json.loads(outlines[outlines.index("--structured-outputs-config") + 1])
    assert structured == {"backend": "outlines", "disable_fallback": True}


def test_verifier_input_normalization_matches_the_locked_phase3_path() -> None:
    payload = '{"product":"mortgage"}'

    assert normalize_verifier_input(f"reasoning\n</think>\n\n{payload}\n") == payload
    assert normalize_verifier_input(payload) == payload


def test_existing_full_receipt_can_be_reused_after_integrity_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = "configs/phase4/serve_r1b_bf16.yaml"
    config = load_phase4_config(config_path)
    raw_path = tmp_path / "receipt.json"
    requests_path = tmp_path / "requests.jsonl"
    workload_path = tmp_path / "workload.jsonl"
    requests_path.write_text('{"request": 1}\n')
    workload_path.write_text('{"workload": 1}\n')
    git_sha = "e1150dc39384141dd25c8b52796c1bffaa730c53"
    raw_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "mode": "full",
                "phase": 4,
                "experiment": config["experiment"],
                "run_id": config["run_id"],
                "config_path": config["_config_path"],
                "config_hash": config["_config_hash"],
                "git_sha": git_sha,
                "model": config["model"],
                "raw_artifact": raw_path.name,
                "request_artifact": {
                    "path": requests_path.name,
                    "sha256": sha256_file(requests_path),
                },
                "workload": {
                    "path": workload_path.name,
                    "contract_hash": workload_contract_hash(config),
                    "sha256": sha256_file(workload_path),
                },
                "cost": {"hourly_usd": 0.30},
            }
        )
    )
    monkeypatch.setattr("forge.bench.runner.phase4_raw_path", lambda *_args, **_kwargs: raw_path)
    monkeypatch.setattr(
        "forge.bench.runner.phase4_requests_path", lambda *_args, **_kwargs: requests_path
    )
    monkeypatch.setattr(
        "forge.bench.runner.phase4_workload_path", lambda *_args, **_kwargs: workload_path
    )
    monkeypatch.setattr("forge.bench.runner.relative_path", lambda path: Path(path).name)

    receipt = validate_existing_receipt(config_path)

    assert receipt["run_id"] == "phase4_serve_r1b_bf16_v2"
    assert receipt["git_sha"] == git_sha


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
    assert points[0]["co_tenancy"]["samples"] > 0
    assert all(item["verifier_input"] == item["output"] for item in requests)


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
    assert summary["co_tenancy"]["samples"] > 0
    assert len(records) == 2
    assert all(item["simultaneous_verifier_input"] for item in records)
    assert all(item["two_pass_verifier_input"] for item in records)


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
    assert "--validate-existing" in worker
    assert "spec_r1b_bf16_mtp.yaml" in worker
    assert "forge.bench.mtp_audit" not in worker
    assert "verify_vllm_qwen35_compat.py" not in worker
    assert "FORGE_VLLM_QWEN35_EXTERNAL_DRAFT_COMPAT" not in worker
    assert "sleep 60" in worker
    assert 'kill -TERM -- "-${server_pid}"' in worker
    assert 'server_log="results/phase4/logs/${run_id}-${FORGE_STARTED_AT//:/}.server.log"' in worker
    assert "FORGE_GPU_HOURLY_USD" in combined
    assert "0.30" in combined
    for cache_variable in (
        "XDG_CACHE_HOME",
        "VLLM_CACHE_ROOT",
        "VLLM_CONFIG_ROOT",
        "CUDA_CACHE_PATH",
        "CUPY_CACHE_DIR",
        "NUMBA_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "HF_HOME",
        "HF_HUB_OFFLINE",
    ):
        assert cache_variable in launcher
    sync_command = '"${uv_bin}" sync --active --locked --no-default-groups --group remote-serve'
    assert sync_command in bootstrap
    assert "FORGE_UV_MIRROR_URL" in bootstrap
    assert "--require-hashes" in bootstrap
    assert "--no-emit-project" in bootstrap
    assert "shutdown" not in combined.lower()
    assert "reboot" not in combined.lower()


def test_mtp_index_audit_and_raw_bundle_preserve_tensor_bytes(tmp_path: Path) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 7},
                "weight_map": {
                    "model.layer.weight": "shard.safetensors",
                    "mtp.fc.weight": "shard.safetensors",
                },
            }
        )
    )
    audit = audit_index(
        index_path=index,
        repository="Qwen/test",
        revision="a" * 40,
        source_url="https://example.invalid/index.json",
    )
    assert audit["decision_branch"] == "native_mtp_reexport"
    assert audit["mtp_weight_keys"] == ["mtp.fc.weight"]

    bundle = tmp_path / "mtp.safetensors"
    payload = b"\x01\x02\x03\x04"
    _write_raw_safetensors(
        bundle,
        {"mtp.fc.weight": ({"dtype": "BF16", "shape": [2]}, payload)},
    )
    header_length, header = _read_safetensors_header(bundle)
    assert header["mtp.fc.weight"]["data_offsets"] == [0, 4]
    assert bundle.read_bytes()[8 + header_length :] == payload


def test_system_load_contamination_gate_uses_half_core_count() -> None:
    sampler = SystemLoadSampler(enabled=True)
    sampler.logical_cpu_count = 16
    sampler.load_threshold = 8
    sampler.samples = [
        {"load1": 7.5, "load5": 5.0, "load15": 4.0, "cpu_utilization": 0.25},
        {"load1": 8.1, "load5": 5.5, "load15": 4.5, "cpu_utilization": 0.50},
    ]
    summary = sampler.summary()

    assert summary["contaminated"] is True
    assert metrics_contaminated({"points": [{"co_tenancy": summary}]}, experiment="spec_decode")


def test_effective_cpu_count_prefers_cgroup_v2_quota(tmp_path: Path) -> None:
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("1600000 100000\n")

    count, source = effective_cpu_count(
        host_logical_cpu_count=128,
        cgroup_v2_cpu_max=cpu_max,
        cgroup_v1_cpu_quota=tmp_path / "missing-quota",
        cgroup_v1_cpu_period=tmp_path / "missing-period",
    )

    assert count == 16
    assert source == "cgroup_v2_cpu.max"


def test_effective_cpu_count_falls_back_to_host_count(tmp_path: Path) -> None:
    count, source = effective_cpu_count(
        host_logical_cpu_count=12,
        cgroup_v2_cpu_max=tmp_path / "missing-max",
        cgroup_v1_cpu_quota=tmp_path / "missing-quota",
        cgroup_v1_cpu_period=tmp_path / "missing-period",
    )

    assert count == 12
    assert source == "os.cpu_count"


def test_phase4_report_code_is_the_only_phase4_numbered_document_writer() -> None:
    makefile = (ROOT / "Makefile").read_text()
    worker = (ROOT / "scripts/remote/run_phase4.sh").read_text()

    assert "forge.bench.report" in makefile
    assert "make bench-report" in worker
    assert "results/phase4_serving_report.md" in (ROOT / "src/forge/bench/report.py").read_text()
