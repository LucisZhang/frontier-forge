"""Archive the D1.2 native-MTP usability decision for an R1b export."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.train.artifacts import write_json_atomic
from forge.train.config import REPO_ROOT, relative_path, sha256_file

from .config import load_phase4_config


def _nested_mtp_layers(config: Mapping[str, Any]) -> int:
    text_config = config.get("text_config")
    value = text_config.get("mtp_num_hidden_layers") if isinstance(text_config, Mapping) else None
    if type(value) is not int or value < 0:
        return 0
    return value


def audit_native_mtp(
    config_path: str | Path,
    *,
    vllm_speculative_source: Path,
    vllm_mtp_loader_source: Path,
) -> dict[str, Any]:
    """Return deterministic evidence about whether the merged export has usable MTP weights."""

    config = load_phase4_config(config_path)
    model_root = REPO_ROOT / str(config["model"]["artifact_path"])
    hf_config_path = model_root / "config.json"
    index_paths = sorted(model_root.glob("*.safetensors.index.json"))
    if len(index_paths) != 1:
        raise RuntimeError(f"expected exactly one safetensors index under {model_root}")
    index_path = index_paths[0]
    hf_config = json.loads(hf_config_path.read_text())
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise RuntimeError(f"invalid safetensors weight_map: {index_path}")
    missing_shards = sorted(
        shard for shard in set(weight_map.values()) if not (model_root / str(shard)).is_file()
    )
    if missing_shards:
        raise RuntimeError(f"safetensors index references missing shards: {missing_shards}")

    selection = config["speculative"]["method_selection"]
    actual_vllm_version = importlib.metadata.version("vllm")
    expected_vllm_version = str(selection["compatibility_vllm_version"])
    if actual_vllm_version != expected_vllm_version:
        raise RuntimeError(
            f"vLLM version drift: expected {expected_vllm_version}, got {actual_vllm_version}"
        )
    actual_speculative_hash = sha256_file(vllm_speculative_source)
    expected_speculative_hash = str(selection["compatibility_source_sha256"])
    if actual_speculative_hash != expected_speculative_hash:
        raise RuntimeError(
            "vLLM speculative.py source drift: "
            f"expected {expected_speculative_hash}, got {actual_speculative_hash}"
        )
    loader_text = vllm_mtp_loader_source.read_text()
    loader_requires_mtp_prefix = 'name.startswith("mtp.")' in loader_text
    if not loader_requires_mtp_prefix:
        raise RuntimeError("Qwen3.5 MTP loader no longer exposes the audited mtp.* contract")

    all_keys = sorted(str(key) for key in weight_map)
    mtp_keys = [key for key in all_keys if key.startswith("mtp.")]
    declared_layers = _nested_mtp_layers(hf_config)
    required_prefixes = (
        "mtp.fc.",
        "mtp.layers.",
        "mtp.norm.",
        "mtp.pre_fc_norm_hidden.",
        "mtp.pre_fc_norm_embedding.",
    )
    present_required_prefixes = [
        prefix for prefix in required_prefixes if any(key.startswith(prefix) for key in mtp_keys)
    ]
    native_mtp_usable = declared_layers > 0 and len(present_required_prefixes) == len(
        required_prefixes
    )
    if native_mtp_usable:
        decision = "native_mtp"
        reason = "export_declares_mtp_and_contains_all_required_mtp_weight_prefixes"
    else:
        decision = "external_draft_fallback"
        reason = (
            f"export_declares_{declared_layers}_mtp_layers_but_contains_"
            f"{len(mtp_keys)}_mtp_weight_keys"
        )

    prior_failures = []
    for item in selection["prior_failure_receipts"]:
        path = REPO_ROOT / str(item)
        if not path.is_file():
            raise FileNotFoundError(path)
        prior_failures.append({"path": relative_path(path), "sha256": sha256_file(path)})

    return {
        "version": 1,
        "status": "complete",
        "phase": 4,
        "decision_rule": "D1.2_native_mtp_first_external_draft_only_if_weights_missing",
        "decision": decision,
        "reason": reason,
        "native_mtp_usable": native_mtp_usable,
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "export": {
            "path": relative_path(model_root),
            "artifact_sha256": config["model"]["artifact_sha256"],
            "config_path": relative_path(hf_config_path),
            "config_sha256": sha256_file(hf_config_path),
            "safetensors_index_path": relative_path(index_path),
            "safetensors_index_sha256": sha256_file(index_path),
            "architecture": hf_config.get("architectures"),
            "model_type": hf_config.get("model_type"),
            "declared_mtp_num_hidden_layers": declared_layers,
            "weight_key_count": len(all_keys),
            "mtp_weight_key_count": len(mtp_keys),
            "mtp_weight_keys": mtp_keys,
            "required_mtp_prefixes": list(required_prefixes),
            "present_required_mtp_prefixes": present_required_prefixes,
        },
        "vllm_contract": {
            "version": actual_vllm_version,
            "speculative_source_path": str(vllm_speculative_source),
            "speculative_source_sha256": actual_speculative_hash,
            "mtp_loader_source_path": str(vllm_mtp_loader_source),
            "mtp_loader_source_sha256": sha256_file(vllm_mtp_loader_source),
            "loader_requires_mtp_prefix": loader_requires_mtp_prefix,
        },
        "prior_failures": prior_failures,
    }


def require_fallback_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the archived evidence before a full external-draft benchmark is recorded."""

    speculative = config.get("speculative")
    if not isinstance(speculative, Mapping) or speculative.get("method") != "draft_model":
        raise RuntimeError(
            "this Phase 4 receipt path only accepts the D1.2 external-draft fallback"
        )
    selection = speculative["method_selection"]
    path = REPO_ROOT / str(selection["native_mtp_audit"])
    if not path.is_file():
        raise FileNotFoundError(f"D1.2 native-MTP audit is missing: {path}")
    audit = json.loads(path.read_text())
    expected = {
        "status": "complete",
        "decision": "external_draft_fallback",
        "native_mtp_usable": False,
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise RuntimeError(f"D1.2 audit mismatch for {key}: {audit.get(key)!r}")
    if audit.get("export", {}).get("artifact_sha256") != config["model"]["artifact_sha256"]:
        raise RuntimeError("D1.2 audit does not match the configured R1b export")
    if audit.get("vllm_contract", {}).get("speculative_source_sha256") != selection.get(
        "compatibility_source_sha256"
    ):
        raise RuntimeError("D1.2 audit does not match the version-guarded compatibility patch")
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "decision": audit["decision"],
        "reason": audit["reason"],
        "native_mtp_usable": audit["native_mtp_usable"],
        "export": audit["export"],
        "vllm_contract": audit["vllm_contract"],
        "prior_failures": audit["prior_failures"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vllm-speculative-source", required=True)
    parser.add_argument("--vllm-mtp-loader-source", required=True)
    args = parser.parse_args()
    receipt = audit_native_mtp(
        args.config,
        vllm_speculative_source=Path(args.vllm_speculative_source),
        vllm_mtp_loader_source=Path(args.vllm_mtp_loader_source),
    )
    if receipt["decision"] != "external_draft_fallback":
        raise RuntimeError("D1.2 requires native MTP when the export has usable MTP weights")
    output = REPO_ROOT / args.output
    write_json_atomic(output, receipt)
    print(json.dumps({"output": relative_path(output), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
