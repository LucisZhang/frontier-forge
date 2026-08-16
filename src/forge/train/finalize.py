"""Append one immutable C2 result after a human-launched remote run completes."""

from __future__ import annotations

import argparse
from typing import Any

from forge.train.artifacts import append_jsonl_once
from forge.train.config import load_config, runs_path


def run_id(rung: str, *, backend: str, seed: int, smoke: bool) -> str:
    names = {
        "r0": "r0_base",
        "r1": "r1_sft_rule",
        "r1b": "r1b_sft_rule_20k",
        "r2": "r2_sft_distilled",
        "r3": "r3_dpo",
        "r4": "r4_grpo",
    }
    suffix = f"_{backend}" if rung == "r1" and backend != "trl" else ""
    prefix = "smoke_" if smoke else ""
    return f"{prefix}{names[rung]}{suffix}_s{seed}"


def _record_from_evaluation(
    receipt: dict[str, Any],
    *,
    started_at: str,
    finished_at: str,
    gpu_hours: float,
    hourly_usd: float,
    smoke: bool,
) -> dict[str, Any]:
    config = load_config(receipt["config_path"])
    metrics = receipt["metrics"]
    backend = str(receipt["backend"])
    seed = int(receipt["seed"])
    cost = {
        "gpu_type": "local-smoke" if smoke else config["budget"]["gpu_type"],
        "gpu_hours": gpu_hours,
        "usd": 0.0 if smoke else gpu_hours * hourly_usd,
    }
    if not smoke:
        cost.update(
            {
                "hourly_usd": hourly_usd,
                "rate_source": "FORGE_GPU_HOURLY_USD supplied by the human launcher",
            }
        )
    return {
        "run_id": run_id(str(config["rung"]), backend=backend, seed=seed, smoke=smoke),
        "phase": 3,
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "git_sha": receipt["git_sha"],
        "dataset_hash": receipt["dataset_hash"],
        "model": receipt["model"]["id"],
        "seed": seed,
        "backend": backend,
        "metrics": {
            "task_success": metrics["task_success"],
            "schema_valid": metrics["schema_valid"],
            "tool_acc": metrics["tool_acc"],
            "field_f1": metrics["field_f1"],
            "abstain_correct": metrics["abstain_correct"],
            "ci95": metrics["ci95"],
            "urgency_match": metrics["urgency_match"],
            "ambiguity_flag_match": metrics["ambiguity_flag_match"],
            "tool_arguments_structural_valid": metrics["tool_arguments_structural_valid"],
            "mean_reward": metrics["mean_reward"],
            "scorer_version": 2,
            "split_metrics": receipt["split_metrics"],
            "general_instruction_accuracy": receipt["general_instruction_regression"]["accuracy"],
            "reward_hacking_probes": receipt["reward_hacking_probes"],
            "failure_categories_nonexclusive": metrics["failure_categories_nonexclusive"],
        },
        "cost": {**cost, "api_usd": 0.0},
        "started_at": started_at,
        "finished_at": finished_at,
        "notes": (
            "SMOKE_ONLY; 0.5B unquantized local path; never a Phase 3 headline."
            if smoke
            else (
                f"Human-launched {backend} run; training_time_quantization="
                f"{receipt['training_time_quantization']}; evaluation_precision="
                f"{receipt['evaluation_precision']}; deployment quantization is separate."
            )
        ),
    }


def finalize_smoke(evaluation_receipt: dict[str, Any]) -> dict[str, Any]:
    finished = str(evaluation_receipt["finished_at"])
    record = _record_from_evaluation(
        evaluation_receipt,
        started_at=finished,
        finished_at=finished,
        gpu_hours=0.0,
        hourly_usd=0.0,
        smoke=True,
    )
    append_jsonl_once(runs_path(smoke=True), record, key="run_id")
    return record


def finalize_full(
    config_path: str,
    *,
    backend: str,
    seed: int,
    started_at: str,
    finished_at: str,
    gpu_hours: float,
    hourly_usd: float,
) -> dict[str, Any]:
    import json

    from forge.train.config import evaluation_root

    config = load_config(config_path)
    metrics_path = evaluation_root(config, seed=seed, smoke=False, backend=backend) / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"cannot finalize without evaluation metrics: {metrics_path}")
    evaluation = json.loads(metrics_path.read_text())
    record = _record_from_evaluation(
        evaluation,
        started_at=started_at,
        finished_at=finished_at,
        gpu_hours=gpu_hours,
        hourly_usd=hourly_usd,
        smoke=False,
    )
    append_jsonl_once(runs_path(smoke=False), record, key="run_id")
    ledger_record = {
        "ledger_id": record["run_id"],
        "status": "complete",
        "gpu_type": record["cost"]["gpu_type"],
        "gpu_hours": gpu_hours,
        "usd": record["cost"]["usd"],
        "hourly_usd": hourly_usd,
        "rate_source": record["cost"]["rate_source"],
        "started_at": started_at,
        "finished_at": finished_at,
        "config_path": record["config_path"],
        "config_hash": record["config_hash"],
        "git_sha": record["git_sha"],
        "notes": "Actual wall-clock receipt from the human-invoked remote rung wrapper.",
    }
    append_jsonl_once(
        runs_path(smoke=False).parent / "phase3_gpu_ledger.jsonl",
        ledger_record,
        key="ledger_id",
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", choices=("trl", "unsloth"), default="trl")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--finished-at", required=True)
    parser.add_argument("--gpu-hours", type=float, required=True)
    parser.add_argument("--hourly-usd", type=float, required=True)
    args = parser.parse_args()
    if args.gpu_hours < 0 or args.hourly_usd <= 0 or not args.started_at or not args.finished_at:
        raise ValueError(
            "finalization requires nonnegative actual time, a positive actual rate, and timestamps"
        )
    record = finalize_full(
        args.config,
        backend=args.backend,
        seed=args.seed,
        started_at=args.started_at,
        finished_at=args.finished_at,
        gpu_hours=args.gpu_hours,
        hourly_usd=args.hourly_usd,
    )
    print(f"appended immutable Phase 3 run: {record['run_id']}")


if __name__ == "__main__":
    main()
