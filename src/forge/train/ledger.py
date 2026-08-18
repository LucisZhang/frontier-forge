"""Record failed remote attempts so GPU cost is never hidden."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import Any

from forge.train.artifacts import append_jsonl_once
from forge.train.config import REPO_ROOT, canonical_hash, git_sha, load_config


def billable_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ledger rows that represent distinct wall-clock GPU attempts.

    A successful finalization and a later wrapper failure can share the same
    config/start timestamp.  The completed receipt owns that interval; retaining
    the failed row is useful evidence, but charging both would double-count it.
    """
    rows = list(records)
    completed_attempts = {
        (str(record["config_hash"]), str(record["started_at"]))
        for record in rows
        if record.get("status") == "complete"
    }
    return [
        record
        for record in rows
        if record.get("status") == "complete"
        or (
            record.get("status") == "failed"
            and (str(record["config_hash"]), str(record["started_at"])) not in completed_attempts
        )
    ]


def record_failure(
    config_path: str,
    *,
    backend: str,
    seed: int,
    started_at: str,
    finished_at: str,
    gpu_hours: float,
    hourly_usd: float,
    exit_code: int,
) -> dict:
    return record_attempt(
        config_path,
        backend=backend,
        seed=seed,
        started_at=started_at,
        finished_at=finished_at,
        gpu_hours=gpu_hours,
        hourly_usd=hourly_usd,
        exit_code=exit_code,
        operation="training",
        status="failed",
    )


def record_attempt(
    config_path: str,
    *,
    backend: str,
    seed: int,
    started_at: str,
    finished_at: str,
    gpu_hours: float,
    hourly_usd: float,
    exit_code: int,
    operation: str,
    status: str,
) -> dict:
    if gpu_hours < 0 or hourly_usd <= 0:
        raise ValueError("GPU ledger requires nonnegative GPU-hours and a positive actual rate")
    if operation not in {"training", "export"} or status not in {"failed", "complete"}:
        raise ValueError("GPU ledger operation/status is invalid")
    if status == "complete" and exit_code != 0:
        raise ValueError("a completed GPU attempt must have exit code 0")
    config = load_config(config_path)
    prefix = "failed" if status == "failed" else operation
    ledger_id = (
        prefix
        + "_"
        + canonical_hash(
            {
                "config_hash": config["_config_hash"],
                "backend": backend,
                "seed": seed,
                "started_at": started_at,
                "operation": operation,
            }
        )[:20]
    )
    record = {
        "ledger_id": ledger_id,
        "status": status,
        "operation": operation,
        "rung": config["rung"],
        "backend": backend,
        "seed": seed,
        "gpu_type": config["budget"]["gpu_type"],
        "gpu_hours": gpu_hours,
        "usd": gpu_hours * hourly_usd,
        "hourly_usd": hourly_usd,
        "rate_source": "FORGE_GPU_HOURLY_USD supplied by the human launcher",
        "exit_code": exit_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "git_sha": git_sha(),
        "notes": (
            "Failed authorized remote attempt retained for the D6 GPU-hour ledger."
            if status == "failed"
            else "Completed authorized remote export receipt for the D6 GPU-hour ledger."
        ),
    }
    append_jsonl_once(
        REPO_ROOT / "results" / "phase3_gpu_ledger.jsonl",
        record,
        key="ledger_id",
    )
    print(f"recorded {status} GPU {operation} attempt: {ledger_id} ({gpu_hours:.3f}h)")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", required=True, choices=("trl", "unsloth"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--finished-at", required=True)
    parser.add_argument("--gpu-hours", required=True, type=float)
    parser.add_argument("--hourly-usd", required=True, type=float)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--operation", choices=("training", "export"), default="training")
    parser.add_argument("--status", choices=("failed", "complete"), default="failed")
    args = parser.parse_args()
    record_attempt(
        args.config,
        backend=args.backend,
        seed=args.seed,
        started_at=args.started_at,
        finished_at=args.finished_at,
        gpu_hours=args.gpu_hours,
        hourly_usd=args.hourly_usd,
        exit_code=args.exit_code,
        operation=args.operation,
        status=args.status,
    )


if __name__ == "__main__":
    main()
