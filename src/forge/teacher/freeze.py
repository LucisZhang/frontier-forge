"""Verify the human-declared D3.1 label freeze used by Phase 2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from forge.data.ingest import sha256_file
from forge.data.input_contract import INPUT_CONTRACT_VERSION
from forge.verify.verifier import SCORER_VERSION

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "teacher_data.yaml"


def resolve_path(value: str | Path, *, root: Path = REPO_ROOT) -> Path:
    """Resolve a repository-relative configuration path."""

    path = Path(value)
    return path if path.is_absolute() else root / path


def load_teacher_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the Phase 2 config and reject unsupported versions."""

    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("Phase 2 teacher config must be version 1")
    return raw


def _require_hash(path: Path, expected: object, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def verify_frozen_source(
    config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Verify every hash that Phase 2 is allowed to read from Phase 1.2."""

    config_path = Path(config_path)
    config = load_teacher_config(config_path)
    source = config["source"]
    manifest_path = resolve_path(source["dataset_manifest"], root=repo_root)
    rules_path = resolve_path(source["label_rules"], root=repo_root)
    freeze_path = resolve_path(source["label_freeze"], root=repo_root)
    _require_hash(manifest_path, source["dataset_manifest_sha256"], "Phase 1.2 manifest")
    _require_hash(rules_path, source["label_rules_sha256"], "label rules")
    if not freeze_path.is_file():
        raise FileNotFoundError(f"D3.1 freeze declaration is missing: {freeze_path}")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 3:
        raise ValueError("Phase 2 requires dataset version 3")
    if manifest.get("dataset_hash") != source["dataset_hash"]:
        raise ValueError("Phase 2 config and Phase 1.2 dataset hash differ")
    label_rules = manifest.get("label_rules", {})
    if label_rules.get("version") != source["label_rules_version"]:
        raise ValueError("Phase 2 config and manifest label-rules versions differ")
    if label_rules.get("sha256") != source["label_rules_sha256"]:
        raise ValueError("Phase 2 config and manifest label-rules hashes differ")
    if manifest.get("protocol", {}).get("membership_frozen") is not True:
        raise ValueError("Phase 1.2 manifest does not declare frozen split membership")
    if source["input_contract_version"] != INPUT_CONTRACT_VERSION:
        raise ValueError("Phase 2 config does not pin the implemented input contract")
    if source["scorer_version"] != SCORER_VERSION:
        raise ValueError("Phase 2 config does not pin the implemented scorer")

    freeze_text = freeze_path.read_text()
    for frozen_value in (source["dataset_hash"], source["label_rules_sha256"]):
        if str(frozen_value) not in freeze_text:
            raise ValueError(f"freeze declaration does not contain {frozen_value}")
    if "Status: **FROZEN" not in freeze_text:
        raise ValueError("D3.1 freeze declaration is not marked FROZEN")

    checked_splits: dict[str, dict[str, Any]] = {}
    split_config = {"train": source["train"], **source["test_splits"]}
    for name, item in split_config.items():
        path = resolve_path(item["path"], root=repo_root)
        _require_hash(path, item["payload_sha256"], f"{name} payload")
        declared = manifest.get("splits", {}).get(name, {})
        checks = {
            "rows": item["rows"],
            "payload_sha256": item["payload_sha256"],
            "membership_sha256": item["membership_sha256"],
        }
        for key, expected in checks.items():
            if declared.get(key) != expected:
                raise ValueError(f"{name} {key} differs between config and frozen manifest")
        checked_splits[name] = {"path": str(path), **checks}

    selection = config["selection"]
    teacher = config["teacher"]
    planned_max = int(selection["live_candidate_cap"]) * float(
        teacher["planning_cost_per_call_usd"]
    )
    budget = float(teacher["max_budget_usd"])
    if planned_max > budget:
        raise ValueError(
            f"planned worst-case teacher cost ${planned_max:.2f} exceeds ${budget:.2f} cap"
        )

    return {
        "dataset_hash": source["dataset_hash"],
        "dataset_manifest_sha256": source["dataset_manifest_sha256"],
        "label_rules_version": source["label_rules_version"],
        "label_rules_sha256": source["label_rules_sha256"],
        "input_contract_version": source["input_contract_version"],
        "scorer_version": source["scorer_version"],
        "splits": checked_splits,
        "planned_max_api_usd": planned_max,
        "budget_usd": budget,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forge.teacher.freeze")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    result = verify_frozen_source(args.config)
    print(
        "label freeze verified: "
        f"dataset_hash={result['dataset_hash']}; rules=v{result['label_rules_version']}; "
        f"budget_usd={result['budget_usd']:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
