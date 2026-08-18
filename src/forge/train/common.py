"""Shared training-run lifecycle helpers."""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge.train.artifacts import json_safe, sha256_tree, write_json_atomic
from forge.train.config import (
    checkpoint_root,
    config_dataset_hash,
    git_sha,
    model_spec,
    relative_path,
    require_full_started_at,
)
from forge.train.runtime import package_versions


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def complete_training_receipt(
    *,
    config: Mapping[str, Any],
    seed: int,
    smoke: bool,
    backend: str,
    started_at: str,
    started_monotonic: float,
    adapter_dir: Path,
    train_metrics: Mapping[str, Any],
    resume_checkpoint: Path | None,
    reward_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = checkpoint_root(config, seed=seed, smoke=smoke, backend=backend)
    receipt = {
        "version": 1,
        "status": "complete",
        "phase": 3,
        "rung": config["rung"],
        "stage": config["stage"],
        "mode": "smoke" if smoke else "full",
        "backend": backend,
        "seed": seed,
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "git_sha": git_sha(),
        "dataset_hash": config_dataset_hash(config, smoke=smoke),
        "model": dict(model_spec(config, smoke=smoke)),
        "training_time_quantization": (
            config["training"]["quantization_smoke"]
            if smoke
            else config["training"]["quantization_full"]
        ),
        "deployment_quantization": "not_performed_by_training",
        "adapter_path": relative_path(adapter_dir),
        "adapter_sha256": sha256_tree(adapter_dir),
        "resumed_from": relative_path(resume_checkpoint) if resume_checkpoint else None,
        "metrics": json_safe(dict(train_metrics)),
        "reward_audit": json_safe(dict(reward_audit)) if reward_audit else None,
        "packages": package_versions(),
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": time.perf_counter() - started_monotonic,
        "notes": (
            "Local 0.5B unquantized LoRA smoke; not headline evidence."
            if smoke
            else "Authorized remote RTX4090 run; full training uses QLoRA NF4."
        ),
    }
    write_json_atomic(root / "train_metrics.json", receipt)
    return receipt


def begin_run(*, smoke: bool) -> tuple[str, float]:
    passed = require_full_started_at(smoke)
    return (passed or utc_now(), time.perf_counter())


def completed_receipt(
    config: Mapping[str, Any], *, seed: int, smoke: bool, backend: str
) -> dict[str, Any] | None:
    import json

    path = checkpoint_root(config, seed=seed, smoke=smoke, backend=backend) / "train_metrics.json"
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text())
    expected = {
        "status": "complete",
        "config_hash": config["_config_hash"],
        "seed": seed,
        "backend": backend,
        "mode": "smoke" if smoke else "full",
    }
    if all(receipt.get(key) == value for key, value in expected.items()):
        return receipt
    raise RuntimeError(f"existing training receipt conflicts with requested run: {path}")
