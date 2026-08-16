"""Record the D4 paired R1 TRL-versus-Unsloth agreement gate."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from forge.train.artifacts import write_json_atomic
from forge.train.config import REPO_ROOT, evaluation_root, git_sha, load_config
from forge.train.evaluate import bootstrap_ci


def _predictions(path: str | Path) -> dict[tuple[str, int], dict]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    result = {(row["split"], int(row["complaint_id"])): row for row in records}
    if len(result) != len(records):
        raise ValueError(f"duplicate paired-evaluation key in {path}")
    return result


def run(*, seed: int = 0) -> dict:
    config = load_config("configs/r1_sft_rule.yaml")
    trl_root = evaluation_root(config, seed=seed, smoke=False, backend="trl")
    unsloth_root = evaluation_root(config, seed=seed, smoke=False, backend="unsloth")
    trl = _predictions(str(trl_root / "predictions.jsonl"))
    unsloth = _predictions(str(unsloth_root / "predictions.jsonl"))
    if trl.keys() != unsloth.keys():
        raise ValueError("R1 backend cross-check predictions are not paired on identical rows")
    deltas = [
        float(unsloth[key]["score"]["task_success"]) - float(trl[key]["score"]["task_success"])
        for key in sorted(trl)
    ]
    ci = bootstrap_ci(
        deltas,
        resamples=int(config["evaluation"]["bootstrap_resamples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    mean_delta = sum(deltas) / len(deltas)
    passed = ci[0] <= 0.0 <= ci[1]
    receipt = {
        "status": "agreement_passed" if passed else "agreement_failed",
        "phase": 3,
        "rung": "r1",
        "reference_backend": "trl",
        "candidate_backend": "unsloth",
        "agreement_rule": "paired task-success delta 95% bootstrap CI includes zero",
        "paired_rows": len(deltas),
        "mean_task_success_delta_unsloth_minus_trl": mean_delta,
        "paired_delta_ci95": ci,
        "bootstrap_resamples": 1000,
        "bootstrap_seed": int(config["evaluation"]["bootstrap_seed"]),
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "git_sha": git_sha(),
        "recorded_at": datetime.now(UTC).isoformat(),
        "default_backend_after_check": "unsloth" if passed else "trl",
        "notes": "Negative agreement is retained and keeps TRL as the default.",
    }
    write_json_atomic(REPO_ROOT / "results" / "phase3_backend_agreement.json", receipt)
    policy = {
        "phase": 3,
        "status": "agreement_passed" if passed else "agreement_failed",
        "default_backend": "unsloth" if passed else "trl",
        "unsloth_allowed_after_r1": passed,
        "agreement_receipt": "results/phase3_backend_agreement.json",
        "notes": (
            "Paired CI includes zero; Unsloth may become the default."
            if passed
            else "Paired CI excludes zero; TRL remains the default and the result is retained."
        ),
    }
    write_json_atomic(REPO_ROOT / "results" / "phase3_backend_policy.json", policy)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(seed=args.seed)


if __name__ == "__main__":
    main()
