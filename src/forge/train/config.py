"""Phase 3 configuration, lineage, and path contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATHS = (
    Path("configs/r0_base.yaml"),
    Path("configs/r1_sft_rule.yaml"),
    Path("configs/r1b_sft_rule_20k.yaml"),
    Path("configs/r2_sft_distilled.yaml"),
    Path("configs/r3_dpo.yaml"),
    Path("configs/r4_grpo.yaml"),
)
CORE_RUNGS = ("r0", "r1", "r2", "r3", "r4")
ALL_RUNGS = ("r0", "r1", "r1b", "r2", "r3", "r4")
FULL_MODEL_ID = "Qwen/Qwen3.5-4B-Base"
FULL_MODEL_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
SMOKE_MODEL_ID = "Qwen/Qwen2.5-0.5B"
SMOKE_MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
PHASE2_DATASET_HASH = "28733f8fefc91efbc6e2b24b85df9312e9054946ee6ca4c693f00133a1ac41e4"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    raw = yaml.safe_load(resolved.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Phase 3 config must be an object: {resolved}")
    validate_config(raw, resolved)
    raw["_config_path"] = relative_path(resolved)
    raw["_config_hash"] = sha256_file(resolved)
    return raw


def configs_by_rung() -> dict[str, Path]:
    configs = {load_config(path)["rung"]: resolve_path(path) for path in CONFIG_PATHS}
    if tuple(configs) != ALL_RUNGS:
        raise ValueError(f"Phase 3 config order mismatch: {tuple(configs)}")
    return configs


def validate_config(config: Mapping[str, Any], path: Path | None = None) -> None:
    label = str(path or "<config>")
    if config.get("version") != 1 or config.get("phase") != 3:
        raise ValueError(f"{label}: expected version 1, phase 3")
    rung = config.get("rung")
    if rung not in ALL_RUNGS:
        raise ValueError(f"{label}: invalid rung {rung!r}")
    expected_stage = {
        "r0": "eval",
        "r1": "sft",
        "r1b": "sft",
        "r2": "sft",
        "r3": "dpo",
        "r4": "grpo",
    }[str(rung)]
    if config.get("stage") != expected_stage:
        raise ValueError(f"{label}: {rung} must use stage {expected_stage}")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError(f"{label}: seeds must be a nonempty integer list")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{label}: duplicate seeds are forbidden")
    if rung == "r4" and len(seeds) != 3:
        raise ValueError(f"{label}: headline rung r4 must pin exactly three seeds")
    if rung != "r4" and len(seeds) != 1:
        raise ValueError(f"{label}: non-headline rungs pin one budget-bound seed")
    revision = config.get("run_revision")
    if revision is not None and (
        rung != "r4"
        or not isinstance(revision, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9_]*", revision) is None
    ):
        raise ValueError(f"{label}: run_revision is an r4-only lowercase artifact identifier")
    model = config.get("model", {})
    full = model.get("full", {})
    smoke = model.get("smoke", {})
    if (full.get("id"), full.get("revision")) != (FULL_MODEL_ID, FULL_MODEL_REVISION):
        raise ValueError(f"{label}: full model must be the D1-locked base checkpoint")
    if (smoke.get("id"), smoke.get("revision")) != (SMOKE_MODEL_ID, SMOKE_MODEL_REVISION):
        raise ValueError(f"{label}: smoke model must be the pinned 0.5B stand-in")
    prompt = config.get("prompt", {})
    if prompt.get("input_contract_version") != 2 or prompt.get("scorer_version") != 2:
        raise ValueError(f"{label}: input/scorer versions must remain at v2")
    if prompt.get("narrative_char_cap_full") != 3250:
        raise ValueError(f"{label}: full narrative cap must preserve the audited 2048-token budget")
    if prompt.get("narrative_char_cap_smoke") != 800:
        raise ValueError(
            f"{label}: smoke narrative cap must preserve the audited 1024-token budget"
        )
    if not resolve_path(prompt.get("path", "missing")).is_file():
        raise ValueError(f"{label}: versioned fair-baseline prompt is missing")
    evaluation = config.get("evaluation", {})
    if evaluation.get("bootstrap_resamples") != 1000:
        raise ValueError(f"{label}: Phase 3 requires exactly 1,000 bootstrap resamples")
    if set(evaluation.get("test_splits", {})) != {"test_iid", "test_drift"}:
        raise ValueError(f"{label}: both frozen TEST splits are required")
    budget = config.get("budget", {})
    if (
        budget.get("gpu_type") != "RTX4090"
        or float(budget.get("project_gpu_hour_ceiling", 0)) != 90.0
    ):
        raise ValueError(f"{label}: D6 RTX4090 and 90-hour ceiling must be explicit")
    training = config.get("training")
    if training is not None:
        if training.get("reference_backend") != "trl":
            raise ValueError(f"{label}: TRL must remain the reference backend")
        if training.get("quantization_full") != "qlora_nf4":
            raise ValueError(f"{label}: full training must use D5 QLoRA")
        if training.get("quantization_smoke") != "none":
            raise ValueError(f"{label}: local smoke cannot claim CUDA QLoRA")
        if config.get("stage") == "sft":
            lora = training.get("lora", {})
            if lora.get("scope") != "language_model_only":
                raise ValueError(f"{label}: text-only SFT must not train the vision tower")
            if lora.get("exclude_modules_regex") != r".*\.visual\..*":
                raise ValueError(f"{label}: the Qwen3.5 vision subtree exclusion must stay pinned")
    if rung == "r4":
        data = config.get("data", {})
        if revision != "phase3_2_fresh_pool":
            raise ValueError(f"{label}: D5.1 requires the Phase 3.2 R4 v2 run revision")
        if (
            data.get("path") != "data/phase3_2/r4_v2_grpo_fresh_rule.jsonl"
            or data.get("prepared_manifest") != "data/phase3_2/manifest.json"
            or int(data.get("rows", 0)) != 8_000
            or int(data.get("contamination_token_ngram", 0)) != 13
        ):
            raise ValueError(f"{label}: D5.1 fresh-pool data contract changed")
        if (
            int(training.get("num_generations", 0)) != 8
            or int(training.get("reward_signal_guard_steps", 0)) != 10
        ):
            raise ValueError(f"{label}: D5.1 generation count and variance guard must stay pinned")


def select_seed(config: Mapping[str, Any], requested: int | None) -> int:
    seeds = [int(seed) for seed in config["seeds"]]
    if requested is None:
        return seeds[0]
    if requested not in seeds:
        raise ValueError(f"seed {requested} is not pinned for {config['rung']}: {seeds}")
    return requested


def model_spec(config: Mapping[str, Any], *, smoke: bool) -> Mapping[str, Any]:
    return config["model"]["smoke" if smoke else "full"]


def checkpoint_root(
    config: Mapping[str, Any], *, seed: int, smoke: bool, backend: str = "trl"
) -> Path:
    mode = "smoke" if smoke else "full"
    root = REPO_ROOT / "checkpoints" / mode / str(config["rung"])
    if config.get("run_revision"):
        root /= str(config["run_revision"])
    return root / backend / f"s{seed}"


def adapter_path(
    config: Mapping[str, Any], *, seed: int, smoke: bool, backend: str = "trl"
) -> Path:
    return checkpoint_root(config, seed=seed, smoke=smoke, backend=backend) / "adapter"


def evaluation_root(
    config: Mapping[str, Any], *, seed: int, smoke: bool, backend: str = "trl"
) -> Path:
    mode = "smoke" if smoke else "full"
    root = REPO_ROOT / "data" / mode / "phase3" / "eval" / str(config["rung"])
    if config.get("run_revision"):
        root /= str(config["run_revision"])
    return root / backend / f"s{seed}"


def runs_path(*, smoke: bool) -> Path:
    if smoke:
        return REPO_ROOT / "data" / "smoke" / "phase3" / "runs.jsonl"
    return REPO_ROOT / "results" / "runs.jsonl"


def git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def phase3_code_status() -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "Makefile",
            "pyproject.toml",
            "uv.lock",
            "configs",
            "scripts/remote",
            "src/forge",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def require_full_started_at(smoke: bool) -> str | None:
    value = os.environ.get("FORGE_STARTED_AT")
    if not smoke and not value:
        raise RuntimeError(
            "full runs require FORGE_STARTED_AT from the human-invoked remote launcher"
        )
    return value


def config_dataset_hash(config: Mapping[str, Any], *, smoke: bool) -> str:
    if smoke:
        return canonical_hash(
            {
                "mode": "smoke",
                "rung": config["rung"],
                "source": config.get("data", {}).get("smoke_path", "frozen-test-smoke"),
            }
        )
    data = config.get("data", {})
    if "prepared_manifest" in data:
        manifest_path = resolve_path(data["prepared_manifest"])
        if not manifest_path.is_file():
            target = "prepare-r1b" if config["rung"] == "r1b" else "prepare-r4-v2"
            raise FileNotFoundError(
                f"{config['rung'].upper()} prepared manifest is missing; run make {target}"
            )
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("status") != "complete"
            or manifest.get("config_hash") != config["_config_hash"]
        ):
            raise ValueError(
                f"{config['rung'].upper()} prepared manifest does not match its config"
            )
        return str(manifest["dataset_hash"])
    return str(data.get("phase2_dataset_hash", PHASE2_DATASET_HASH))
