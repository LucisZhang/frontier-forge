"""Phase 4 benchmark configuration and artifact contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from forge.train.config import REPO_ROOT, canonical_json, relative_path, sha256_file

PHASE4_CONFIG_ROOT = REPO_ROOT / "configs" / "phase4"
EXPERIMENTS = frozenset({"serve", "spec_decode", "structured"})
PRECISIONS = frozenset({"bfloat16", "gptq_int4"})


def phase4_config_paths() -> tuple[Path, ...]:
    """Return the checked-in Phase 4 experiment configs in stable order."""

    return tuple(sorted(PHASE4_CONFIG_ROOT.glob("*.yaml")))


def load_phase4_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    raw = yaml.safe_load(resolved.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Phase 4 config must be an object: {resolved}")
    validate_phase4_config(raw, path=resolved)
    raw["_config_path"] = relative_path(resolved)
    raw["_config_hash"] = sha256_file(resolved)
    return raw


def _require_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


def validate_phase4_config(config: Mapping[str, Any], *, path: Path | None = None) -> None:
    label = str(path or "<config>")
    if config.get("version") != 1 or config.get("phase") != 4:
        raise ValueError(f"{label}: expected version 1, phase 4")
    experiment = config.get("experiment")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"{label}: unsupported experiment {experiment!r}")
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not run_id.startswith(f"phase4_{experiment}_"):
        raise ValueError(f"{label}: run_id must be namespaced by the experiment")

    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError(f"{label}: model must be an object")
    if model.get("precision") not in PRECISIONS:
        raise ValueError(f"{label}: unsupported precision {model.get('precision')!r}")
    for key in ("variant", "served_name", "artifact_path", "export_manifest"):
        if not isinstance(model.get(key), str) or not model[key]:
            raise ValueError(f"{label}: model.{key} must be nonempty")
    expected_hash = model.get("artifact_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"{label}: model.artifact_sha256 must be a SHA-256")

    server = config.get("server")
    if not isinstance(server, Mapping):
        raise ValueError(f"{label}: server must be an object")
    _require_positive_int(server.get("port"), f"{label}: server.port")
    _require_positive_int(server.get("max_model_len"), f"{label}: server.max_model_len")
    utilization = _require_positive_number(
        server.get("gpu_memory_utilization"), f"{label}: server.gpu_memory_utilization"
    )
    if utilization > 1:
        raise ValueError(f"{label}: server.gpu_memory_utilization must be <= 1")

    workload = config.get("workload")
    if not isinstance(workload, Mapping):
        raise ValueError(f"{label}: workload must be an object")
    for key in ("rows", "warmup_requests", "measurement_requests", "max_concurrency"):
        _require_positive_int(workload.get(key), f"{label}: workload.{key}")
    _require_positive_number(workload.get("deadline_s"), f"{label}: workload.deadline_s")
    qps = workload.get("arrival_rates_qps")
    if not isinstance(qps, list) or not qps:
        raise ValueError(f"{label}: workload.arrival_rates_qps must be nonempty")
    rates = [_require_positive_number(item, f"{label}: arrival rate") for item in qps]
    if rates != sorted(set(rates)):
        raise ValueError(f"{label}: arrival rates must be sorted and unique")
    targets = workload.get("input_token_targets")
    target_weights = workload.get("input_token_weights")
    output_caps = workload.get("output_token_caps")
    output_weights = workload.get("output_token_weights")
    for values, weights, name in (
        (targets, target_weights, "input"),
        (output_caps, output_weights, "output"),
    ):
        if (
            not isinstance(values, list)
            or not values
            or any(type(item) is not int for item in values)
        ):
            raise ValueError(f"{label}: {name} token values must be nonempty integers")
        if not isinstance(weights, list) or len(weights) != len(values):
            raise ValueError(f"{label}: {name} token weights must match values")
        if any(not isinstance(item, (int, float)) or item <= 0 for item in weights):
            raise ValueError(f"{label}: {name} token weights must be positive")
        if abs(sum(float(item) for item in weights) - 1.0) > 1e-9:
            raise ValueError(f"{label}: {name} token weights must sum to one")

    stability = config.get("stability")
    if not isinstance(stability, Mapping):
        raise ValueError(f"{label}: stability must be an object")
    for key in ("max_error_rate", "max_deadline_miss_rate", "min_achieved_qps_ratio"):
        value = stability.get(key)
        if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
            raise ValueError(f"{label}: stability.{key} must be in [0, 1]")
    _require_positive_number(
        stability.get("max_p95_inflation"), f"{label}: stability.max_p95_inflation"
    )

    if experiment == "spec_decode":
        speculative = config.get("speculative")
        if not isinstance(speculative, Mapping):
            raise ValueError(f"{label}: speculative config is required")
        enabled = speculative.get("enabled")
        if type(enabled) is not bool:
            raise ValueError(f"{label}: speculative.enabled must be boolean")
        if enabled:
            if speculative.get("draft_model") != "Qwen/Qwen2.5-0.5B":
                raise ValueError(f"{label}: D1 pins the 0.5B Qwen-family draft")
            _require_positive_int(
                speculative.get("num_speculative_tokens"),
                f"{label}: speculative.num_speculative_tokens",
            )
    if experiment == "structured":
        structured = config.get("structured")
        if not isinstance(structured, Mapping):
            raise ValueError(f"{label}: structured config is required")
        if structured.get("backend") not in {"xgrammar", "outlines"}:
            raise ValueError(f"{label}: structured backend must be xgrammar or outlines")
        complexities = structured.get("schema_field_counts")
        if (
            not isinstance(complexities, list)
            or len(complexities) < 2
            or any(type(item) is not int or item <= 0 for item in complexities)
        ):
            raise ValueError(f"{label}: at least two schema complexities are required")


def workload_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the hashable workload fields shared by all matched experiments."""

    workload = config["workload"]
    assert isinstance(workload, Mapping)
    keys = (
        "dataset_manifest",
        "split",
        "rows",
        "selection_seed",
        "input_token_targets",
        "input_token_weights",
        "output_token_caps",
        "output_token_weights",
        "warmup_requests",
        "measurement_requests",
        "arrival_rates_qps",
        "max_concurrency",
        "deadline_s",
        "request_seed",
    )
    return {key: workload[key] for key in keys}


def workload_contract_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(workload_contract(config)).encode()).hexdigest()


def phase4_raw_path(config: Mapping[str, Any], *, smoke: bool) -> Path:
    root = REPO_ROOT / ("data/smoke/phase4/raw" if smoke else "results/phase4/raw")
    return root / f"{config['run_id']}.json"


def phase4_requests_path(config: Mapping[str, Any], *, smoke: bool) -> Path:
    root = REPO_ROOT / ("data/smoke/phase4/raw" if smoke else "results/phase4/raw")
    return root / f"{config['run_id']}.requests.jsonl"


def phase4_workload_path(config: Mapping[str, Any], *, smoke: bool) -> Path:
    root = REPO_ROOT / ("data/smoke/phase4" if smoke else "data/full/phase4")
    return root / f"workload-{workload_contract_hash(config)[:16]}.jsonl"


def read_export_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    path = REPO_ROOT / str(config["model"]["export_manifest"])
    value = json.loads(path.read_text())
    if value.get("status") != "complete":
        raise RuntimeError(f"export manifest is not complete: {path}")
    return value
