"""Phase 4 local/remote preflight and one-time serving-artifact verification."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge.train.artifacts import sha256_tree, write_json_atomic
from forge.train.config import REPO_ROOT, git_sha, relative_path, sha256_file

from .config import load_phase4_config, phase4_config_paths, read_export_manifest

VERIFICATION_PATH = REPO_ROOT / "results/phase4/artifact_verification.json"


def benchmark_git_sha() -> str:
    value = os.environ.get("FORGE_BENCH_GIT_SHA") or git_sha()
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("FORGE_BENCH_GIT_SHA must be a full lowercase Git SHA")
    return value


def verify_artifacts(config_paths: list[str | Path]) -> dict[str, Any]:
    configs = [load_phase4_config(path) for path in config_paths]
    artifacts: dict[str, dict[str, Any]] = {}
    equivalence: dict[str, Any] = {}
    for config in configs:
        model = config["model"]
        artifact_path = REPO_ROOT / str(model["artifact_path"])
        export = read_export_manifest(config)
        field = (
            "full_precision_export"
            if model["precision"] == "bfloat16"
            else "deployment_int4_export"
        )
        export_artifact = export[field]
        if export_artifact["path"] != model["artifact_path"]:
            raise RuntimeError(f"export path mismatch for {config['_config_path']}")
        if export_artifact["sha256"] != model["artifact_sha256"]:
            raise RuntimeError(f"export manifest hash mismatch for {config['_config_path']}")
        if not artifact_path.is_dir():
            raise FileNotFoundError(artifact_path)
        key = relative_path(artifact_path)
        if key not in artifacts:
            actual = sha256_tree(artifact_path)
            if actual != model["artifact_sha256"]:
                raise RuntimeError(f"artifact tree hash mismatch: {key}")
            artifacts[key] = {
                "sha256": actual,
                "export_manifest": model["export_manifest"],
                "export_manifest_sha256": sha256_file(REPO_ROOT / str(model["export_manifest"])),
                "precision": model["precision"],
                "variants": [model["variant"]],
            }
        elif model["variant"] not in artifacts[key]["variants"]:
            artifacts[key]["variants"].append(model["variant"])
        evidence = model.get("equivalence_evidence")
        if evidence:
            left = REPO_ROOT / str(evidence["left_adapter_weights"])
            right = REPO_ROOT / str(evidence["right_adapter_weights"])
            left_hash = sha256_file(left)
            right_hash = sha256_file(right)
            expected = str(evidence["adapter_weights_sha256"])
            if left_hash != right_hash or left_hash != expected:
                raise RuntimeError("R3-equivalent comparison adapter weights are not identical")
            equivalence[model["variant"]] = {
                "left_adapter_weights": relative_path(left),
                "right_adapter_weights": relative_path(right),
                "sha256": expected,
                "interpretation": evidence["interpretation"],
            }
    receipt = {
        "version": 1,
        "status": "complete",
        "phase": 4,
        "git_sha": benchmark_git_sha(),
        "verified_at": datetime.now(UTC).isoformat(),
        "artifacts": dict(sorted(artifacts.items())),
        "equivalence": equivalence,
    }
    write_json_atomic(VERIFICATION_PATH, receipt)
    return receipt


def require_verified_artifact(config: dict[str, Any]) -> dict[str, Any]:
    if not VERIFICATION_PATH.is_file():
        raise FileNotFoundError("run forge.bench.preflight --verify-artifacts first")
    receipt = json.loads(VERIFICATION_PATH.read_text())
    if receipt.get("status") != "complete":
        raise RuntimeError("Phase 4 artifact verification is incomplete")
    path = str(config["model"]["artifact_path"])
    item = receipt.get("artifacts", {}).get(path)
    if not isinstance(item, dict) or item.get("sha256") != config["model"]["artifact_sha256"]:
        raise RuntimeError(f"artifact has no matching verification receipt: {path}")
    return item


def host_disclosure() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        name, driver, memory = [item.strip() for item in query.splitlines()[0].split(",")]
        result.update(
            {
                "gpu": name,
                "driver_version": driver,
                "gpu_memory_total_mib": int(memory),
            }
        )
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        result.update({"gpu": None, "driver_version": None, "gpu_memory_total_mib": None})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-artifacts", action="store_true")
    parser.add_argument("--config", action="append", default=[])
    args = parser.parse_args()
    paths = args.config or [str(path) for path in phase4_config_paths()]
    if args.verify_artifacts:
        receipt = verify_artifacts(paths)
        print(json.dumps(receipt, sort_keys=True))
    else:
        for path in paths:
            load_phase4_config(path)
        print(json.dumps({"configs": len(paths), "host": host_disclosure()}, sort_keys=True))


if __name__ == "__main__":
    main()
