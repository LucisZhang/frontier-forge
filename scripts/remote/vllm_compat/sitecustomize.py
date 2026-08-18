"""Narrow vLLM 0.17.0 compatibility hook for D1.2 external-draft fallback."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import pathlib
import sys

ENABLE_ENV = "FORGE_VLLM_QWEN35_EXTERNAL_DRAFT_COMPAT"
ACTIVE_ENV = "FORGE_VLLM_QWEN35_EXTERNAL_DRAFT_COMPAT_ACTIVE"
EXPECTED_VERSION = "0.17.0"
EXPECTED_SPECULATIVE_SOURCE_SHA256 = (
    "3c7b6b19d80854619eb01068d70f5e83ce6f17798c643493c836eee4514d76f5"
)


def _abort(message: str) -> None:
    print(f"FORGE vLLM compatibility guard failed: {message}", file=sys.stderr, flush=True)
    raise SystemExit(86)


if os.environ.get(ENABLE_ENV) == "1":
    actual_version = importlib.metadata.version("vllm")
    if actual_version != EXPECTED_VERSION:
        _abort(f"expected vLLM {EXPECTED_VERSION}, got {actual_version}")

    import vllm.config.speculative as speculative_module

    source_path = pathlib.Path(speculative_module.__file__)
    actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_SPECULATIVE_SOURCE_SHA256:
        _abort(
            "speculative.py SHA-256 drifted: "
            f"expected {EXPECTED_SPECULATIVE_SOURCE_SHA256}, got {actual_hash}"
        )

    original_override = speculative_module.SpeculativeConfig.hf_config_override

    def _external_draft_override(hf_config):
        if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
            return hf_config
        return original_override(hf_config)

    speculative_module.SpeculativeConfig.hf_config_override = staticmethod(_external_draft_override)
    os.environ[ACTIVE_ENV] = "1"
    print(
        "FORGE vLLM 0.17.0 compatibility active: preserving Qwen3.5 as an external draft_model",
        file=sys.stderr,
        flush=True,
    )
