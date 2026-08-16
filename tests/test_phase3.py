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
    configs_by_rung,
    load_config,
)
from forge.train.data import compact_model_input
from forge.train.evaluate import bootstrap_ci
from forge.train.grpo import RewardAudit
from forge.train.preflight import check_config, require_r1_reference_receipt
from forge.train.runtime import lora_config, versioned_training_argument

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
    ]
    assert unsloth == [
        "bitsandbytes==0.50.1; sys_platform == 'linux'",
        "unsloth==2026.8.18; sys_platform == 'linux'",
        "trl==0.24.0",
    ]
    assert project["tool"]["uv"]["conflicts"] == [[{"group": "train"}, {"group": "unsloth-train"}]]


def test_locked_reference_and_unsloth_forks_are_exact() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    versions: dict[str, set[str]] = {}
    for package in lock["package"]:
        versions.setdefault(package["name"], set()).add(package["version"])

    assert versions["torch"] == {"2.11.0", "2.13.0"}
    assert versions["transformers"] == {"5.5.0", "5.15.0"}
    assert versions["trl"] == {"0.24.0", "1.10.0"}
    assert versions["unsloth"] == {"2026.8.18"}


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


def test_bootstrap_ci_is_fixed_seed_and_bounded() -> None:
    values = [0.0, 1.0, 1.0, 0.0, 1.0]

    first = bootstrap_ci(values, resamples=1000, seed=20260816)
    second = bootstrap_ci(values, resamples=1000, seed=20260816)

    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0


def test_remote_launch_scripts_are_syntax_valid_and_human_triggered() -> None:
    for name in ("launch_phase3.sh", "run_phase3_rung.sh", "bootstrap.sh", "sync.sh"):
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts/remote" / name)],
            check=True,
            capture_output=True,
            text=True,
        )
    launcher = (ROOT / "scripts/remote/launch_phase3.sh").read_text()
    worker = (ROOT / "scripts/remote/run_phase3_rung.sh").read_text()
    assert "tmux new-session" in launcher
    assert "FORGE_STARTED_AT" in launcher
    assert ".venv-unsloth/bin/python" in launcher
    assert "FORGE_TRAIN_PYTHON" in launcher
    assert "FORGE_GPU_HOURLY_USD" in launcher
    assert 'reference_python=".venv/bin/python"' in worker
    assert "--hourly-usd" in worker
    assert "forge.train.finalize" in worker
    assert "forge.train.ledger" in worker


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
