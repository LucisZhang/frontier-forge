"""Phase 3 lineage, backend-policy, and GPU-budget preflight checks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.train.config import (
    CONFIG_PATHS,
    PHASE2_DATASET_HASH,
    REPO_ROOT,
    evaluation_root,
    load_config,
    phase3_code_status,
    resolve_path,
    select_seed,
    sha256_file,
)
from forge.train.data import validate_training_data
from forge.train.ledger import billable_records


def backend_policy() -> dict[str, Any]:
    path = REPO_ROOT / "results" / "phase3_backend_policy.json"
    return json.loads(path.read_text())


def require_backend_allowed(config: Mapping[str, Any], *, backend: str, smoke: bool) -> None:
    if backend == "trl":
        return
    if backend != "unsloth":
        raise ValueError("backend must be trl or unsloth")
    if smoke:
        raise RuntimeError("Unsloth is CUDA-only")
    if config["rung"] == "r1":
        return
    policy = backend_policy()
    if not (
        policy.get("status") == "agreement_passed"
        and policy.get("default_backend") == "unsloth"
        and policy.get("unsloth_allowed_after_r1") is True
    ):
        raise RuntimeError(
            "D4 gate: Unsloth cannot run after R1 until paired agreement is recorded"
        )


def _phase2_manifest() -> dict[str, Any]:
    path = REPO_ROOT / "results" / "phase2_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Phase 2 manifest is not complete")
    if manifest.get("phase2_dataset_hash") != PHASE2_DATASET_HASH:
        raise ValueError("Phase 3 configs target a different Phase 2 dataset")
    return manifest


def _context_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    path = REPO_ROOT / "results" / "phase3_context_audit.json"
    if not path.is_file():
        raise FileNotFoundError("Phase 3 context audit is missing; run make phase3-context-audit")
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "complete" or int(receipt.get("violation_count", -1)) != 0:
        raise ValueError("Phase 3 context audit is not green")
    expected = receipt.get("config_hashes", {}).get(config["rung"])
    if expected != config["_config_hash"]:
        raise ValueError(f"Phase 3 context audit is stale for {config['rung']}")
    return receipt


def actual_gpu_hours() -> float:
    path = REPO_ROOT / "results" / "phase3_gpu_ledger.jsonl"
    if not path.is_file():
        return 0.0
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    return sum(float(record["gpu_hours"]) for record in billable_records(records))


def check_budget(config: Mapping[str, Any]) -> dict[str, float]:
    actual = actual_gpu_hours()
    projected = float(config["budget"]["projected_gpu_hours_per_run"])
    ceiling = float(config["budget"]["project_gpu_hour_ceiling"])
    planning_rate = float(config["budget"]["hourly_usd"])
    if actual + projected > ceiling:
        raise RuntimeError(
            f"D6 stop: actual {actual:.3f}h + projected {projected:.3f}h exceeds {ceiling:.1f}h"
        )
    return {
        "actual_gpu_hours": actual,
        "projected_next_gpu_hours": projected,
        "ceiling_gpu_hours": ceiling,
        "projected_total_gpu_hours": actual + projected,
        "planning_rate_usd_per_gpu_hour": planning_rate,
        "projected_next_planning_usd": projected * planning_rate,
    }


def require_r1_reference_receipt(config: Mapping[str, Any], *, seed: int) -> dict[str, str]:
    predictions = evaluation_root(config, seed=seed, smoke=False, backend="trl") / (
        "predictions.jsonl"
    )
    if not predictions.is_file():
        raise FileNotFoundError(
            "D4 gate: run and finalize the R1 TRL reference before the Unsloth cross-check"
        )
    runs = REPO_ROOT / "results" / "runs.jsonl"
    records = [json.loads(line) for line in runs.read_text().splitlines() if line]
    matches = [
        record
        for record in records
        if record.get("run_id") == f"r1_sft_rule_s{seed}"
        and record.get("backend") == "trl"
        and record.get("config_hash") == config["_config_hash"]
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "D4 gate: the current R1 TRL reference needs one immutable production run receipt"
        )
    return {
        "run_id": str(matches[0]["run_id"]),
        "predictions": str(predictions.relative_to(REPO_ROOT)),
    }


def check_launch_hardware() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("full launch requires a CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("D6 requires exactly one visible GPU; multi-GPU launches are forbidden")
    name = torch.cuda.get_device_name(0)
    if "4090" not in name:
        raise RuntimeError(f"D6 requires the RTX 4090 workhorse, found {name!r}")
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": name,
        "visible_device_count": torch.cuda.device_count(),
        "total_memory_bytes": int(properties.total_memory),
        "cuda_version": torch.version.cuda,
    }


def check_config(
    config_path: str | Path,
    *,
    smoke: bool,
    seed: int | None,
    backend: str,
    launch: bool,
) -> dict[str, Any]:
    config = load_config(config_path)
    selected_seed = select_seed(config, seed)
    require_backend_allowed(config, backend=backend, smoke=smoke)
    manifest = _phase2_manifest()
    prompt_path = resolve_path(config["prompt"]["path"])
    checks: dict[str, Any] = {
        "rung": config["rung"],
        "stage": config["stage"],
        "seed": selected_seed,
        "backend": backend,
        "mode": "smoke" if smoke else "full",
        "config_hash": config["_config_hash"],
        "prompt_sha256": sha256_file(prompt_path),
        "phase2_dataset_hash": manifest["phase2_dataset_hash"],
    }
    if config["stage"] != "eval":
        path = validate_training_data(config, smoke=smoke)
        checks["training_data_path"] = str(path.relative_to(REPO_ROOT))
        checks["training_data_sha256"] = sha256_file(path)
    if not smoke:
        context_audit = _context_audit(config)
        checks["context_audit_status"] = context_audit["status"]
        checks["budget"] = check_budget(config)
    if launch and not smoke and config["rung"] == "r1" and backend == "unsloth":
        checks["r1_reference"] = require_r1_reference_receipt(config, seed=selected_seed)
    if launch and not smoke:
        dirty_code = phase3_code_status()
        if dirty_code:
            raise RuntimeError(
                "full launch requires the Phase 3 code/config tree to match a reviewed commit: "
                + ", ".join(dirty_code[:10])
            )
        checks["launch_code_tree_clean"] = True
        checks["launch_hardware"] = check_launch_hardware()
    if launch and config["lineage"].get("parent_rung") not in {None, "r0"}:
        from forge.train.config import adapter_path

        parent = load_config(config["lineage"]["parent_config"])
        parent_adapter = adapter_path(
            parent, seed=int(parent["seeds"][0]), smoke=smoke, backend=backend
        )
        if not (parent_adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"parent rung is not complete: {parent_adapter}")
        checks["parent_adapter"] = str(parent_adapter.relative_to(REPO_ROOT))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backend", choices=("trl", "unsloth"), default="trl")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    paths = CONFIG_PATHS if args.all else (Path(args.config),)
    checks = [
        check_config(
            path,
            smoke=args.smoke,
            seed=args.seed,
            backend=args.backend,
            launch=args.launch,
        )
        for path in paths
    ]
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
