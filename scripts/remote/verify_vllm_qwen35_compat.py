"""Fail closed unless the D1.2 Qwen3.5 external-draft hook is active."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os

from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if os.environ.get("FORGE_VLLM_QWEN35_EXTERNAL_DRAFT_COMPAT_ACTIVE") != "1":
        raise RuntimeError("sitecustomize compatibility hook is not active")
    draft_config = Qwen3_5Config.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=True,
    )
    architecture_before = list(draft_config.architectures or [])
    model_type_before = draft_config.model_type
    patched = SpeculativeConfig.hf_config_override(draft_config)
    if patched.model_type != "qwen3_5":
        raise RuntimeError(f"Qwen3.5 draft was still rewritten to {patched.model_type}")
    if list(patched.architectures or []) != architecture_before:
        raise RuntimeError("Qwen3.5 external-draft architecture was unexpectedly rewritten")
    print(
        json.dumps(
            {
                "status": "compatible",
                "vllm_version": importlib.metadata.version("vllm"),
                "model_type_before": model_type_before,
                "model_type_after": patched.model_type,
                "architectures": architecture_before,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
