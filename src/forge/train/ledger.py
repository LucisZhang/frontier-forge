"""Record failed remote attempts so GPU cost is never hidden."""

from __future__ import annotations

import argparse

from forge.train.artifacts import append_jsonl_once
from forge.train.config import REPO_ROOT, canonical_hash, git_sha, load_config


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
    if gpu_hours < 0 or hourly_usd <= 0:
        raise ValueError("failure ledger requires nonnegative GPU-hours and a positive actual rate")
    config = load_config(config_path)
    ledger_id = (
        "failed_"
        + canonical_hash(
            {
                "config_hash": config["_config_hash"],
                "backend": backend,
                "seed": seed,
                "started_at": started_at,
            }
        )[:20]
    )
    record = {
        "ledger_id": ledger_id,
        "status": "failed",
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
        "notes": "Failed human-launched attempt retained for the D6 GPU-hour ledger.",
    }
    append_jsonl_once(
        REPO_ROOT / "results" / "phase3_gpu_ledger.jsonl",
        record,
        key="ledger_id",
    )
    print(f"recorded failed GPU attempt: {ledger_id} ({gpu_hours:.3f}h)")
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
    args = parser.parse_args()
    record_failure(
        args.config,
        backend=args.backend,
        seed=args.seed,
        started_at=args.started_at,
        finished_at=args.finished_at,
        gpu_hours=args.gpu_hours,
        hourly_usd=args.hourly_usd,
        exit_code=args.exit_code,
    )


if __name__ == "__main__":
    main()
