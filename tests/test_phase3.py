from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

import pytest

from forge.train.config import (
    ALL_RUNGS,
    CONFIG_PATHS,
    FULL_MODEL_ID,
    FULL_MODEL_REVISION,
    PHASE2_DATASET_HASH,
    SMOKE_MODEL_ID,
    checkpoint_root,
    configs_by_rung,
    evaluation_root,
    load_config,
    runs_path,
    sha256_file,
    smoke_output_root,
)
from forge.train.data import compact_model_input
from forge.train.evaluate import bootstrap_ci
from forge.train.export import (
    _export_contract,
    _require_complete_merged_export,
    _save_processor_assets,
    durable_export_manifest_path,
)
from forge.train.finalize import run_id
from forge.train.grpo import (
    RewardAudit,
    RewardSignalGuard,
    _completion_text,
    _require_clean_rollout_sample,
    assert_synthetic_gold_reward,
    reward_signal_receipt_path,
)
from forge.train.ledger import billable_records, record_attempt
from forge.train.preflight import actual_gpu_hours, check_config, require_r1_reference_receipt
from forge.train.report import _failed_gpu_attempts
from forge.train.runtime import lora_config, versioned_training_argument
from forge.train.sft import processor_eos_token, text_processing_class

ROOT = Path(__file__).resolve().parents[1]


def test_compact_model_input_never_leaks_evaluation_label() -> None:
    model_input = compact_model_input(
        {
            "complaint_id": 7,
            "narrative": "A test complaint",
            "source_product": "mortgage",
            "source_issue": "Foreclosure",
            "source_company": "Example Bank",
            "label": {"urgency": "high"},
        },
        max_narrative_chars=100,
    )

    assert set(model_input) == {
        "complaint_id",
        "narrative",
        "source_product",
        "source_issue",
        "source_company",
    }
    assert "label" not in model_input


def test_every_phase3_rung_has_one_locked_config() -> None:
    configs = configs_by_rung()

    assert tuple(configs) == ALL_RUNGS
    assert len(CONFIG_PATHS) == 6
    for rung, path in configs.items():
        config = load_config(path)
        assert config["rung"] == rung
        assert config["model"]["full"] == {
            "id": FULL_MODEL_ID,
            "revision": FULL_MODEL_REVISION,
            "dtype": "bfloat16",
        }
        assert config["model"]["smoke"]["id"] == SMOKE_MODEL_ID
        assert config["evaluation"]["bootstrap_resamples"] == 1000
        assert config["budget"]["project_gpu_hour_ceiling"] == 90.0
        assert config["prompt"]["narrative_char_cap_full"] == 3250
        assert config["prompt"]["narrative_char_cap_smoke"] == 800
        assert config["evaluation"]["smoke_max_prompt_tokens"] == 1024


def test_phase3_training_configs_separate_training_and_deployment_quantization() -> None:
    for path in CONFIG_PATHS[1:]:
        config = load_config(path)
        assert config["training"]["quantization_full"] == "qlora_nf4"
        assert config["training"]["quantization_smoke"] == "none"
    r4 = load_config("configs/r4_grpo.yaml")
    assert r4["export"]["deployment_quantization"] == "gptq_int4"
    assert r4["reward"]["implementation"] == "forge.verify.verifier.score"


def test_remote_training_packages_are_linux_only_and_lockable() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    reference = project["dependency-groups"]["remote-reference"]
    unsloth = project["dependency-groups"]["unsloth-train"]

    assert reference == [
        "bitsandbytes==0.50.1; sys_platform == 'linux'",
        "gptqmodel==7.3.2; sys_platform == 'linux'",
        "optimum==2.2.0; sys_platform == 'linux'",
    ]
    assert unsloth == [
        "bitsandbytes==0.50.1; sys_platform == 'linux'",
        "unsloth==2026.8.18; sys_platform == 'linux'",
        "trl==0.24.0",
    ]
    assert [{"group": "train"}, {"group": "unsloth-train"}] in project["tool"]["uv"]["conflicts"]


def test_locked_training_and_serving_forks_are_exact() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    versions: dict[str, set[str]] = {}
    for package in lock["package"]:
        versions.setdefault(package["name"], set()).add(package["version"])

    assert versions["torch"] == {"2.10.0", "2.11.0", "2.13.0"}
    assert versions["transformers"] == {"4.57.6", "5.5.0", "5.15.0"}
    assert versions["trl"] == {"0.24.0", "1.10.0"}
    assert versions["unsloth"] == {"2026.8.18"}
    assert versions["optimum"] == {"2.2.0"}
    assert versions["vllm"] == {"0.17.0"}


def test_versioned_training_argument_only_passes_supported_fields() -> None:
    class LegacyConfig:
        def __init__(self, *, max_prompt_length: int = 512) -> None:
            self.max_prompt_length = max_prompt_length

    class CurrentConfig:
        def __init__(self, *, max_completion_length: int = 256) -> None:
            self.max_completion_length = max_completion_length

    assert versioned_training_argument(LegacyConfig, "max_prompt_length", 1800) == {
        "max_prompt_length": 1800
    }
    assert versioned_training_argument(CurrentConfig, "max_prompt_length", 1800) == {}


def test_unsloth_processor_uses_real_eos_token() -> None:
    class InnerTokenizer:
        eos_token = "<|endoftext|>"

    class Processor:
        eos_token = None
        tokenizer = InnerTokenizer()

    assert processor_eos_token(Processor()) == "<|endoftext|>"
    assert text_processing_class(Processor()) is Processor.tokenizer
    with pytest.raises(RuntimeError, match="usable EOS token"):
        processor_eos_token(object())
    plain = object()
    assert text_processing_class(plain) is plain


def test_unsloth_is_activated_before_trl_imports() -> None:
    for name, trainer_import in (
        ("sft.py", "from trl import SFTConfig"),
        ("dpo.py", "from trl import DPOConfig"),
        ("grpo.py", "from trl import GRPOConfig"),
    ):
        source = (ROOT / "src" / "forge" / "train" / name).read_text()
        backend_section = source.split("def _unsloth_train", 1)[-1] if name == "sft.py" else source
        assert backend_section.index("activate_unsloth_runtime()") < backend_section.index(
            trainer_import
        )
    sft_source = (ROOT / "src" / "forge" / "train" / "sft.py").read_text()
    assert "eos_token=processor_eos_token(tokenizer)" in sft_source


def test_full_sft_lora_excludes_qwen35_vision_tower() -> None:
    for path in CONFIG_PATHS[1:4]:
        config = load_config(path)
        adapter = lora_config(config, smoke=False)
        assert config["training"]["lora"]["scope"] == "language_model_only"
        assert adapter.exclude_modules == r".*\.visual\..*"


def test_phase3_configs_pin_completed_phase2_artifacts() -> None:
    manifest = json.loads((ROOT / "results/phase2_manifest.json").read_text())

    assert manifest["phase2_dataset_hash"] == PHASE2_DATASET_HASH
    for path in (
        "configs/r1_sft_rule.yaml",
        "configs/r2_sft_distilled.yaml",
        "configs/r3_dpo.yaml",
    ):
        config = load_config(path)
        artifact = manifest["artifacts"][
            {"r1": "sft_rule", "r2": "sft_distilled", "r3": "dpo_pairs"}[config["rung"]]
        ]
        assert config["data"]["rows"] == artifact["rows"]
        assert config["data"]["sha256"] == artifact["sha256"]


def test_r1b_is_optional_exact_20k_contamination_screened_ablation() -> None:
    config = load_config("configs/r1b_sft_rule_20k.yaml")

    assert config["optional"] is True
    assert config["data"]["rows"] == 20_000
    assert config["data"]["contamination_token_ngram"] == 13
    assert config["data"]["required_r1_sha256"] == (
        "0d54f4ead643d5b7c54247be396780b77e5566837aa8d19fe03e77b6c4a4215e"
    )


def test_grpo_reward_is_exact_scorer_v2_reward() -> None:
    gold = {
        "product": "mortgage",
        "issue": "Foreclosure",
        "company": "Example Bank",
        "urgency": "high",
        "ambiguity_flag": False,
        "tool_call": {
            "name": "escalate_to_regulator",
            "arguments": {"complaint_id": 7, "reason": "Active foreclosure risk."},
        },
    }
    audit = RewardAudit()

    rewards = audit.reward([json.dumps(gold)], [json.dumps(gold)])

    assert rewards == [1.0]
    assert audit.summary()["scorer_version"] == 2
    assert audit.summary()["completions_scored"] == 1


def test_grpo_think_block_hardening_keeps_only_the_trailing_answer() -> None:
    expected = '{"urgency":"high"}'

    assert _completion_text(f"\n</think>\n\n{expected}") == expected
    assert _completion_text(f"<think>discard me</think>\n{expected}") == expected
    assert _completion_text([{"role": "assistant", "content": expected}]) == expected


def test_grpo_synthetic_gold_smoke_gate_has_positive_reward() -> None:
    assert assert_synthetic_gold_reward() == 1.0


def test_grpo_rollout_archive_requires_raw_bare_json(tmp_path: Path) -> None:
    gold = {
        "product": "mortgage",
        "issue": "Foreclosure",
        "company": "Example Bank",
        "urgency": "high",
        "ambiguity_flag": False,
        "tool_call": {
            "name": "escalate_to_regulator",
            "arguments": {"complaint_id": 7, "reason": "Active foreclosure risk."},
        },
    }
    encoded = json.dumps(gold)
    clean_path = tmp_path / "clean.json"
    clean = RewardAudit(rollout_sample_path=clean_path, rollout_sample_context={"seed": 0})

    assert clean.reward([encoded], [encoded]) == [1.0]
    assert _require_clean_rollout_sample(clean_path)["hardening_required"] is False

    prefixed_path = tmp_path / "prefixed.json"
    prefixed = RewardAudit(rollout_sample_path=prefixed_path)
    assert prefixed.reward([f"\n</think>\n{encoded}"], [encoded]) == [1.0]
    assert not prefixed_path.exists()


def test_grpo_reward_signal_guard_aborts_after_ten_all_zero_std_steps() -> None:
    guard = RewardSignalGuard(opening_steps=10)
    for step in range(1, 10):
        guard.observe(step, {"frac_reward_zero_std": 1.0})
    with pytest.raises(RuntimeError, match="stayed 1.0"):
        guard.observe(10, {"frac_reward_zero_std": 1.0})

    healthy = RewardSignalGuard(opening_steps=10)
    for step in range(1, 11):
        healthy.observe(step, {"frac_reward_zero_std": 0.5 if step == 3 else 1.0})
    assert healthy.summary()["all_zero_std"] is False


def test_r4_v2_has_fresh_pool_paths_ids_and_unchanged_guards() -> None:
    config = load_config("configs/r4_grpo.yaml")
    source = (ROOT / "src/forge/train/grpo.py").read_text()

    assert config["run_revision"] == "phase3_2_fresh_pool"
    assert config["data"]["path"] == "data/phase3_2/r4_v2_grpo_fresh_rule.jsonl"
    assert config["data"]["rows"] == 8_000
    assert config["data"]["contamination_token_ngram"] == 13
    assert config["training"]["num_generations"] == 8
    assert config["training"]["reward_signal_guard_steps"] == 10
    assert config["training"]["completion_logging_steps"] == 3
    assert checkpoint_root(config, seed=0, smoke=False, backend="trl").parts[-4:] == (
        "r4",
        "phase3_2_fresh_pool",
        "trl",
        "s0",
    )
    assert evaluation_root(config, seed=0, smoke=False, backend="trl").parts[-4:] == (
        "r4",
        "phase3_2_fresh_pool",
        "trl",
        "s0",
    )
    assert (
        run_id(
            "r4",
            backend="trl",
            seed=0,
            smoke=False,
            run_revision=config["run_revision"],
        )
        == "r4_grpo_phase3_2_fresh_pool_s0"
    )
    assert reward_signal_receipt_path(config, seed=2, smoke=False, backend="trl") == (
        ROOT / "results/phase3_r4_reward_signal_phase3_2_fresh_pool_s2.json"
    )
    assert '"chat_template_kwargs"' in source
    assert '{"enable_thinking": False}' in source


def test_r4_v2_manifest_proves_disjoint_clean_rule_labeled_pool() -> None:
    config = load_config("configs/r4_grpo.yaml")
    manifest = json.loads((ROOT / "data/phase3_2/manifest.json").read_text())

    assert manifest["status"] == "complete"
    assert manifest["config_hash"] == config["_config_hash"]
    assert manifest["run_revision"] == "phase3_2_fresh_pool"
    assert manifest["artifact"]["rows"] == 8_000
    assert manifest["artifact"]["rule_labels_attached"] is True
    assert manifest["disjointness"] == {
        "fresh_overlap_phase2": 0,
        "fresh_overlap_previous_training": 0,
        "fresh_unique_rows": 8_000,
        "phase2_is_subset_of_previous_training": True,
        "phase2_unique_rows": 1_450,
        "previously_trained_unique_rows": 20_000,
    }
    assert manifest["contamination"]["selected_rows_clean"] is True
    assert manifest["contamination"]["test_rows_scanned"] == {
        "test_drift": 80_000,
        "test_iid": 104_443,
    }
    assert len(manifest["artifact"]["sha256"]) == 64
    assert len(manifest["dataset_hash"]) == 64


def test_original_r4_runs_are_preserved_and_marked_superseded_inconclusive() -> None:
    runs = [json.loads(line) for line in (ROOT / "results/runs.jsonl").read_text().splitlines()]
    statuses = [
        json.loads(line)
        for line in (ROOT / "results/phase3_run_status.jsonl").read_text().splitlines()
    ]
    old_ids = {f"r4_grpo_s{seed}" for seed in range(3)}

    assert old_ids <= {record["run_id"] for record in runs}
    assert {record["run_id"] for record in statuses} == old_ids
    assert {record["status"] for record in statuses} == {"superseded-inconclusive"}


def test_phase6_can_isolate_all_mutable_phase3_smoke_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolated = tmp_path / "phase3"
    monkeypatch.setenv("FORGE_SMOKE_OUTPUT_ROOT", str(isolated))
    config = load_config("configs/r4_grpo.yaml")

    assert smoke_output_root() == isolated
    assert checkpoint_root(config, seed=0, smoke=True, backend="trl") == (
        isolated / "checkpoints/r4/phase3_2_fresh_pool/trl/s0"
    )
    assert evaluation_root(config, seed=0, smoke=True, backend="trl") == (
        isolated / "eval/r4/phase3_2_fresh_pool/trl/s0"
    )
    assert runs_path(smoke=True) == isolated / "runs.jsonl"


def test_r1b_reuses_the_pinned_r4_deployment_export_contract() -> None:
    r1b = load_config("configs/r1b_sft_rule_20k.yaml")
    contract, settings = _export_contract(r1b)

    assert contract["rung"] == "r4"
    assert settings == {
        "merge_dtype": "bfloat16",
        "deployment_quantization": "gptq_int4",
        "group_size": 128,
        "calibration_rows": 128,
    }


def test_phase3_1_receipts_remain_durable_during_the_r4_v2_contract() -> None:
    r1b = load_config("configs/r1b_sft_rule_20k.yaml")
    export_path = durable_export_manifest_path(r1b, seed=0, backend="trl")
    export_receipt = json.loads(export_path.read_text())
    diagnostic = json.loads((ROOT / "results/phase3_r4_reward_signal_diagnostic.json").read_text())
    selection = json.loads((ROOT / "results/phase3_export_selection.json").read_text())
    report = (ROOT / "results/phase3_report.md").read_text()

    assert export_path == ROOT / "results/phase3_export_manifest_r1b_trl_s0.json"
    assert export_receipt["status"] == "complete"
    assert export_receipt["full_precision_export"]["sha256"] == (
        "7cf43a2905513f61797b78b7e3fd7ebdacd1cba4fc89abea9ce209401e6e6435"
    )
    assert export_receipt["deployment_int4_export"]["sha256"] == (
        "c99b42cf0e062cc75f2df8588725d0c29383666f3db0c1ae837ce15bfe6d39d2"
    )
    assert diagnostic["status"] == "aborted-zero-reward-variance"
    assert diagnostic["guard"]["observed_frac_reward_zero_std"] == [1.0] * 10
    assert diagnostic["logged_rollout_window"]["reward_values"] == [1.0]
    assert diagnostic["logged_rollout_window"]["advantage_values"] == [0.0]
    assert diagnostic["disposition"]["unlaunched_seeds"] == [1, 2]
    assert len(diagnostic["evidence"]["remote_log_sha256"]) == 64
    assert diagnostic["evidence"]["remote_log_retention"].startswith(
        "Exact raw log retained on forge-pod"
    )
    assert (
        sha256_file(ROOT / diagnostic["evidence"]["rollout_sample_path"])
        == (diagnostic["evidence"]["rollout_sample_sha256"])
    )
    contract = selection["r4_best_seed_export_contract"]
    assert contract["status"] in {
        "pending-r4-v2",
        "aborted-r4-v2-guard",
        "eligible-ci-significant-win",
        "not-eligible-no-ci-significant-win",
    }
    assert contract["historical_phase3_1_diagnostic"] == (
        "results/phase3_r4_reward_signal_diagnostic.json"
    )
    assert "Phase 3.1 guard result" in report
    assert "R4 v2 fresh-pool verdict" in report


def test_phase3_2_guard_abort_is_final_without_a_fabricated_seed2_delta() -> None:
    config = load_config("configs/r4_grpo.yaml")
    revision = config["run_revision"]
    receipts = {
        seed: json.loads(
            (ROOT / f"results/phase3_r4_reward_signal_{revision}_s{seed}.json").read_text()
        )
        for seed in (0, 1, 2)
    }
    deltas = json.loads((ROOT / "results/phase3_paired_deltas.json").read_text())
    selection = json.loads((ROOT / "results/phase3_export_selection.json").read_text())
    report = (ROOT / "results/phase3_report.md").read_text()

    assert receipts[0]["status"] == "passed-nonzero-reward-variance"
    assert receipts[1]["status"] == "passed-nonzero-reward-variance"
    assert receipts[2]["status"] == "aborted-zero-reward-variance"
    assert receipts[2]["guard"]["observed"] == {str(step): 1.0 for step in range(1, 11)}

    seed_deltas = {int(item["to_seed"]): item for item in deltas["r4_v2_seed_deltas"]}
    assert seed_deltas[0]["status"] == "complete"
    assert seed_deltas[1]["status"] == "complete"
    assert seed_deltas[2]["status"] == "aborted-zero-reward-variance"
    assert seed_deltas[2]["paired_rows"] == 0
    assert "mean_task_success_delta" not in seed_deltas[2]
    assert deltas["r4_v2_aggregate"] == {
        "from": "r3",
        "to": "r4",
        "status": "aborted-zero-reward-variance",
        "verdict": "aborted",
        "aborted_seeds": [2],
        "completed_seeds": [0, 1],
        "reason": (
            "The unchanged ten-step reward-variance guard stopped R4 v2; no "
            "three-seed aggregate or missing paired delta is fabricated."
        ),
    }
    assert selection["r4_best_seed_export_contract"]["status"] == "aborted-r4-v2-guard"
    assert "Final R4 v2 verdict: **ABORTED BY LOCKED GUARD**" in report
    assert "no frozen evaluation or paired delta exists" in report


def test_full_export_preserves_pinned_processor_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    class Processor:
        def save_pretrained(self, output_dir: Path) -> None:
            calls["output_dir"] = output_dir
            (output_dir / "processor_config.json").write_text("{}")

    class AutoProcessor:
        @classmethod
        def from_pretrained(
            cls, model_id: str, *, revision: str, local_files_only: bool
        ) -> Processor:
            calls["model_id"] = model_id
            calls["revision"] = revision
            calls["local_files_only"] = local_files_only
            return Processor()

    monkeypatch.setattr("transformers.AutoProcessor", AutoProcessor)
    config = load_config("configs/r4_grpo.yaml")

    _save_processor_assets(config, tmp_path)

    assert calls == {
        "model_id": FULL_MODEL_ID,
        "revision": FULL_MODEL_REVISION,
        "local_files_only": True,
        "output_dir": tmp_path,
    }
    assert (tmp_path / "processor_config.json").is_file()


def test_full_export_only_reuses_complete_merged_weights(tmp_path: Path) -> None:
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}})
    )

    _require_complete_merged_export(tmp_path)

    (tmp_path / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(RuntimeError, match="missing weight shards"):
        _require_complete_merged_export(tmp_path)


def test_bootstrap_ci_is_fixed_seed_and_bounded() -> None:
    values = [0.0, 1.0, 1.0, 0.0, 1.0]

    first = bootstrap_ci(values, resamples=1000, seed=20260816)
    second = bootstrap_ci(values, resamples=1000, seed=20260816)

    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0


def test_remote_launch_scripts_are_syntax_valid_and_human_triggered() -> None:
    for name in (
        "launch_phase3.sh",
        "run_phase3_rung.sh",
        "launch_phase3_export.sh",
        "run_phase3_export.sh",
        "bootstrap.sh",
        "sync.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts/remote" / name)],
            check=True,
            capture_output=True,
            text=True,
        )
    launcher = (ROOT / "scripts/remote/launch_phase3.sh").read_text()
    worker = (ROOT / "scripts/remote/run_phase3_rung.sh").read_text()
    export_launcher = (ROOT / "scripts/remote/launch_phase3_export.sh").read_text()
    export_worker = (ROOT / "scripts/remote/run_phase3_export.sh").read_text()
    assert "tmux new-session" in launcher
    assert "FORGE_STARTED_AT" in launcher
    assert ".venv-unsloth/bin/python" in launcher
    assert "FORGE_TRAIN_PYTHON" in launcher
    assert "FORGE_GPU_HOURLY_USD" in launcher
    assert "http_proxy" in launcher
    assert "REQUESTS_CA_BUNDLE" in launcher
    assert "HF_HUB_DISABLE_XET" in launcher
    assert "HF_HUB_OFFLINE=1" in launcher
    assert "TRANSFORMERS_OFFLINE=1" in launcher
    assert "PYTORCH_ALLOC_CONF" in launcher
    assert "expandable_segments:True" in launcher
    assert "UNSLOTH_COMPILE_LOCATION" in launcher
    assert "nvidia-smi pmon -c 1" in launcher
    assert 'session="forge-' in launcher
    assert 'reference_python=".venv/bin/python"' in worker
    assert "trap 'exit 130' INT" in worker
    assert "--hourly-usd" in worker
    assert "forge.train.finalize" in worker
    assert "forge.train.ledger" in worker
    assert worker.index("completed=1") < worker.index("forge.train.report")
    assert "nvidia-smi pmon -c 1" in export_launcher
    assert 'session="forge-export-' in export_launcher
    assert "configs/r1b_sft_rule_20k.yaml" in export_launcher
    assert "HF_HUB_OFFLINE=1" in export_launcher
    assert "TRANSFORMERS_OFFLINE=1" in export_launcher
    assert "--operation export --status complete" in export_worker
    assert export_worker.index("completed=1") < export_worker.index("forge.train.report")


def test_gpu_ledger_deduplicates_post_finalize_wrapper_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unique_failed = {
        "ledger_id": "failed_before_finalize",
        "status": "failed",
        "rung": "r0",
        "seed": 0,
        "config_hash": "r0-config",
        "started_at": "2026-08-16T10:24:58Z",
        "gpu_hours": 0.05,
    }
    complete = {
        "ledger_id": "r0_base_s0",
        "status": "complete",
        "config_hash": "r0-config",
        "started_at": "2026-08-16T10:41:24Z",
        "gpu_hours": 0.98,
    }
    overlapping_failed = {
        "ledger_id": "failed_after_finalize",
        "status": "failed",
        "rung": "r0",
        "seed": 0,
        "config_hash": "r0-config",
        "started_at": "2026-08-16T10:41:24Z",
        "gpu_hours": 0.99,
    }
    records = [unique_failed, complete, overlapping_failed]
    ledger = tmp_path / "results" / "phase3_gpu_ledger.jsonl"
    ledger.parent.mkdir()
    ledger.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert [record["ledger_id"] for record in billable_records(records)] == [
        "failed_before_finalize",
        "r0_base_s0",
    ]
    monkeypatch.setattr("forge.train.preflight.REPO_ROOT", tmp_path)
    monkeypatch.setattr("forge.train.report.REPO_ROOT", tmp_path)
    assert actual_gpu_hours() == pytest.approx(1.03)
    assert [record["ledger_id"] for record in _failed_gpu_attempts(smoke=False)] == [
        "failed_before_finalize"
    ]


def test_gpu_ledger_records_completed_r1b_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("forge.train.ledger.REPO_ROOT", tmp_path)
    monkeypatch.setattr("forge.train.ledger.git_sha", lambda: "a" * 40)

    record = record_attempt(
        str(ROOT / "configs/r1b_sft_rule_20k.yaml"),
        backend="trl",
        seed=0,
        started_at="2026-08-18T00:00:00Z",
        finished_at="2026-08-18T01:00:00Z",
        gpu_hours=1.0,
        hourly_usd=0.30,
        exit_code=0,
        operation="export",
        status="complete",
    )

    assert record["status"] == "complete"
    assert record["operation"] == "export"
    assert record["usd"] == pytest.approx(0.30)
    assert record["ledger_id"].startswith("export_")


def test_full_launch_refuses_dirty_phase3_runtime_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forge.train.preflight.phase3_code_status",
        lambda: [" M src/forge/train/sft.py"],
    )

    with pytest.raises(RuntimeError, match="reviewed commit"):
        check_config(
            "configs/r0_base.yaml",
            smoke=False,
            seed=0,
            backend="trl",
            launch=True,
        )


def test_unsloth_crosscheck_refuses_missing_r1_reference_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = load_config("configs/r1_sft_rule.yaml")
    missing = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(
        "forge.train.preflight.evaluation_root", lambda *args, **kwargs: missing.parent
    )

    with pytest.raises(FileNotFoundError, match="R1 TRL reference"):
        require_r1_reference_receipt(config, seed=0)


def test_phase3_make_targets_are_implemented() -> None:
    makefile = (ROOT / "Makefile").read_text()

    for target in ("train-sft", "train-dpo", "train-grpo", "eval", "export-model"):
        recipe = makefile.split(f"{target}:", 1)[1].split("\n\n", 1)[0]
        assert "[stub]" not in recipe
        assert "forge.train." in recipe

    smoke_recipe = makefile.split("phase3-smoke:", 1)[1].split("\n\nsync-up:", 1)[0]
    for path in CONFIG_PATHS[1:]:
        assert str(path) in smoke_recipe
    assert "$(MAKE) eval" in smoke_recipe
    assert "$(MAKE) export-model" in smoke_recipe
    assert "$(MAKE) prepare-r4-v2 SMOKE=1" in smoke_recipe
