"""Render the pinned vLLM command for one Phase 4 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.train.config import REPO_ROOT

from .config import load_phase4_config


def server_command(config_path: str | Path, *, executable: str) -> list[str]:
    config = load_phase4_config(config_path)
    model = config["model"]
    server = config["server"]
    command = [
        executable,
        "serve",
        str(REPO_ROOT / model["artifact_path"]),
        "--served-model-name",
        str(model["served_name"]),
        "--host",
        str(server["host"]),
        "--port",
        str(server["port"]),
        "--dtype",
        "float16" if model["precision"] == "gptq_int4" else "bfloat16",
        "--max-model-len",
        str(server["max_model_len"]),
        "--max-num-seqs",
        str(server["max_num_seqs"]),
        "--gpu-memory-utilization",
        str(server["gpu_memory_utilization"]),
        "--generation-config",
        str(server["generation_config"]),
        "--language-model-only",
        "--no-enable-log-requests",
        "--enable-request-id-headers",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        str(server["tool_call_parser"]),
        "--structured-outputs-config",
        json.dumps(
            {
                "backend": server["structured_backend"],
                "disable_fallback": server["structured_backend"] in {"xgrammar", "outlines"},
            },
            separators=(",", ":"),
        ),
    ]
    if model["precision"] == "gptq_int4":
        command.extend(["--quantization", "gptq"])
    speculative = config.get("speculative", {})
    if speculative.get("enabled"):
        speculative_config = {
            "method": speculative["method"],
            "num_speculative_tokens": speculative["num_speculative_tokens"],
        }
        if speculative["method"] == "draft_model":
            speculative_config.update(
                {
                    "model": speculative["draft_model"],
                    "revision": speculative["draft_revision"],
                }
            )
        command.extend(
            [
                "--speculative-config",
                json.dumps(speculative_config, separators=(",", ":")),
            ]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--executable", default=".venv-phase4/bin/vllm")
    parser.add_argument("--null", action="store_true")
    args = parser.parse_args()
    separator = "\0" if args.null else "\n"
    print(separator.join(server_command(args.config, executable=args.executable)), end=separator)


if __name__ == "__main__":
    main()
