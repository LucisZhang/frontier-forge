"""Merge the cumulative LoRA and export separate full-precision and GPTQ int4 artifacts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge.train.artifacts import sha256_tree, write_json_atomic
from forge.train.config import (
    REPO_ROOT,
    adapter_path,
    checkpoint_root,
    git_sha,
    load_config,
    model_spec,
    relative_path,
    select_seed,
    sha256_file,
)
from forge.train.data import narrative_char_limit, prompt_messages
from forge.train.runtime import load_base_model, load_tokenizer, package_versions


def _smoke_export(config: dict[str, Any], *, seed: int, backend: str) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    source = adapter_path(config, seed=seed, smoke=True, backend=backend)
    if not (source / "adapter_config.json").is_file():
        raise FileNotFoundError(f"smoke adapter is missing: {source}")
    root = (
        REPO_ROOT
        / "data"
        / "smoke"
        / "phase3"
        / "export"
        / str(config["rung"])
        / backend
        / f"s{seed}"
    )
    receipt_path = root / "export_manifest.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("config_hash") == config["_config_hash"]:
            print(f"smoke export already complete: {relative_path(receipt_path)}")
            return receipt
        raise RuntimeError("existing smoke export belongs to another config")
    tensor_path = source / "adapter_model.safetensors"
    tensors = load_file(tensor_path)
    name = sorted(tensors)[0]
    values = tensors[name].detach().float().reshape(-1)[:4096].cpu()
    scale = float(values.abs().max().item() / 7.0) if values.numel() else 1.0
    if scale == 0.0:
        scale = 1.0
    quantized = torch.clamp(torch.round(values / scale), -8, 7).to(torch.int16) + 8
    if quantized.numel() % 2:
        quantized = torch.cat((quantized, torch.tensor([8], dtype=torch.int16)))
    packed = (quantized[0::2] | (quantized[1::2] << 4)).to(torch.uint8).numpy().tobytes()
    int4_dir = root / "gptq_int4_contract_rehearsal"
    int4_dir.mkdir(parents=True, exist_ok=True)
    packed_path = int4_dir / "packed_adapter_sample.bin"
    packed_path.write_bytes(packed)
    write_json_atomic(
        int4_dir / "quantization.json",
        {
            "artifact_type": "smoke_contract_rehearsal_not_a_deployable_model",
            "source_tensor": name,
            "source_values": int(values.numel()),
            "bits": 4,
            "scale": scale,
        },
    )
    receipt = {
        "version": 1,
        "status": "complete",
        "phase": 3,
        "mode": "smoke",
        "rung": config["rung"],
        "backend": backend,
        "seed": seed,
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "git_sha": git_sha(),
        "training_time_quantization": "none",
        "full_precision_export": {
            "artifact_type": "adapter_input_only_not_merged_weights",
            "path": relative_path(source),
            "sha256": sha256_tree(source),
        },
        "deployment_int4_export": {
            "artifact_type": "packing_contract_rehearsal_not_deployable",
            "method": "synthetic_int4_pack",
            "path": relative_path(int4_dir),
            "sha256": sha256_tree(int4_dir),
            "packed_sample_sha256": sha256_file(packed_path),
        },
        "notes": "SMOKE_ONLY. Full merge and GPTQModel quantization remain remote GPU actions.",
        "finished_at": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(receipt_path, receipt)
    print(f"smoke export contract complete: {relative_path(receipt_path)}")
    return receipt


def _calibration_texts(config: dict[str, Any], tokenizer: Any) -> list[str]:
    path = REPO_ROOT / "data" / "phase2" / "sft_rule.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    limit = int(config["export"]["calibration_rows"])
    prompt = (REPO_ROOT / config["prompt"]["path"]).read_text().strip()
    texts = []
    for record in records[:limit]:
        messages = prompt_messages(
            record["model_input"],
            prompt=prompt,
            max_narrative_chars=narrative_char_limit(config, smoke=False),
        )
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    if len(texts) != limit:
        raise RuntimeError("GPTQ calibration corpus is smaller than the pinned row count")
    return texts


def _save_processor_assets(config: dict[str, Any], output_dir: Any) -> None:
    """Preserve Qwen 3.5 processor metadata required by GPTQModel reloads."""
    from transformers import AutoProcessor

    spec = model_spec(config, smoke=False)
    try:
        processor = AutoProcessor.from_pretrained(
            spec["id"], revision=spec["revision"], local_files_only=True
        )
    except OSError:
        processor = AutoProcessor.from_pretrained(spec["id"], revision=spec["revision"])
    processor.save_pretrained(output_dir)


def _require_complete_merged_export(output_dir: Path) -> None:
    required = {
        output_dir / "config.json",
        output_dir / "model.safetensors.index.json",
        output_dir / "tokenizer.json",
        output_dir / "tokenizer_config.json",
    }
    missing = sorted(str(path.name) for path in required if not path.is_file())
    if missing:
        raise RuntimeError(f"partial merged BF16 export is missing: {', '.join(missing)}")
    index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    shards = {str(value) for value in index.get("weight_map", {}).values()}
    if not shards or any(not (output_dir / shard).is_file() for shard in shards):
        raise RuntimeError("partial merged BF16 export has missing weight shards")


def _full_export(config: dict[str, Any], *, seed: int, backend: str) -> dict[str, Any]:
    import gc

    import torch
    from gptqmodel import GPTQConfig, GPTQModel
    from peft import PeftModel

    source = adapter_path(config, seed=seed, smoke=False, backend=backend)
    if not (source / "adapter_config.json").is_file():
        raise FileNotFoundError(f"full adapter is missing: {source}")
    root = checkpoint_root(config, seed=seed, smoke=False, backend=backend) / "export"
    receipt_path = root / "export_manifest.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("config_hash") == config["_config_hash"]:
            print(f"full export already complete: {relative_path(receipt_path)}")
            return receipt
        raise RuntimeError("existing full export belongs to another config")
    fp_dir = root / "merged_bf16"
    int4_dir = root / "gptq_int4"
    tokenizer = load_tokenizer(config, smoke=False)
    if fp_dir.is_dir():
        _require_complete_merged_export(fp_dir)
        print(f"reusing complete merged BF16 export: {relative_path(fp_dir)}")
    else:
        base = load_base_model(config, smoke=False, for_training=False, quantized_training=False)
        merged = PeftModel.from_pretrained(base, source, is_trainable=False).merge_and_unload()
        fp_dir.mkdir(parents=True, exist_ok=False)
        merged.save_pretrained(fp_dir, safe_serialization=True, max_shard_size="4GB")
        del merged, base
        gc.collect()
        torch.cuda.empty_cache()
    tokenizer.save_pretrained(fp_dir)
    _save_processor_assets(config, fp_dir)
    fp_hash = sha256_tree(fp_dir)
    quant_config = GPTQConfig(
        bits=4,
        group_size=int(config["export"]["group_size"]),
        desc_act=False,
        act_group_aware=True,
    )
    quantized = GPTQModel.load(str(fp_dir), quant_config)
    quantized.quantize(_calibration_texts(config, tokenizer), batch_size=1)
    quantized.save(str(int4_dir))
    int4_hash = sha256_tree(int4_dir)
    receipt = {
        "version": 1,
        "status": "complete",
        "phase": 3,
        "mode": "full",
        "rung": config["rung"],
        "backend": backend,
        "seed": seed,
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "git_sha": git_sha(),
        "model": dict(model_spec(config, smoke=False)),
        "source_adapter_path": relative_path(source),
        "source_adapter_sha256": sha256_tree(source),
        "training_time_quantization": config["training"]["quantization_full"],
        "full_precision_export": {
            "method": "peft_merge_and_unload",
            "dtype": config["export"]["merge_dtype"],
            "path": relative_path(fp_dir),
            "sha256": fp_hash,
        },
        "deployment_int4_export": {
            "method": config["export"]["deployment_quantization"],
            "group_size": int(config["export"]["group_size"]),
            "calibration_rows": int(config["export"]["calibration_rows"]),
            "path": relative_path(int4_dir),
            "sha256": int4_hash,
        },
        "packages": {
            **package_versions(),
            "gptqmodel": importlib.metadata.version("gptqmodel"),
        },
        "finished_at": datetime.now(UTC).isoformat(),
        "notes": "QLoRA training and GPTQ deployment quantization are separate recorded facts.",
    }
    write_json_atomic(receipt_path, receipt)
    print(f"full export complete: fp={fp_hash} gptq_int4={int4_hash}")
    return receipt


def run(config_path: str, *, seed: int | None, smoke: bool, backend: str = "trl") -> dict[str, Any]:
    config = load_config(config_path)
    if config["rung"] != "r4" or "export" not in config:
        raise ValueError("Phase 3 export is defined for the cumulative R4 adapter")
    selected_seed = select_seed(config, seed)
    return (
        _smoke_export(config, seed=selected_seed, backend=backend)
        if smoke
        else _full_export(config, seed=selected_seed, backend=backend)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backend", choices=("trl", "unsloth"), default="trl")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, seed=args.seed, smoke=args.smoke, backend=args.backend)


if __name__ == "__main__":
    main()
