"""Lazy ML-runtime integration for TRL, PEFT, and the locked model family."""

from __future__ import annotations

import importlib.metadata
import inspect
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from forge.train.config import adapter_path, load_config, model_spec


def package_versions() -> dict[str, str]:
    names = (
        "torch",
        "transformers",
        "trl",
        "peft",
        "datasets",
        "accelerate",
        "numpy",
        "pyarrow",
    )
    return {name: importlib.metadata.version(name) for name in names}


def versioned_training_argument(config_class: type[Any], name: str, value: Any) -> dict[str, Any]:
    """Return a version-specific trainer argument only when that TRL class accepts it."""
    return {name: value} if name in inspect.signature(config_class).parameters else {}


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtime_device(*, smoke: bool) -> str:
    import torch

    requested = os.environ.get("FORGE_SMOKE_DEVICE") if smoke else None
    if requested:
        if requested not in {"cpu", "mps"}:
            raise ValueError("FORGE_SMOKE_DEVICE must be cpu or mps")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return requested
    if smoke:
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("full Phase 3 mode is remote CUDA-only")
    return "cuda"


def _dtype(name: str) -> Any:
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_tokenizer(config: Mapping[str, Any], *, smoke: bool) -> Any:
    from transformers import AutoTokenizer

    spec = model_spec(config, smoke=smoke)
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(
    config: Mapping[str, Any],
    *,
    smoke: bool,
    for_training: bool,
    quantized_training: bool,
) -> Any:
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    spec = model_spec(config, smoke=smoke)
    auto_config = AutoConfig.from_pretrained(spec["id"], revision=spec["revision"])
    kwargs: dict[str, Any] = {
        "revision": spec["revision"],
        "dtype": _dtype(str(spec["dtype"])),
        "low_cpu_mem_usage": True,
    }
    if quantized_training:
        if smoke:
            raise RuntimeError("smoke mode must not claim bitsandbytes QLoRA")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    elif not smoke:
        kwargs["device_map"] = {"": 0}

    if auto_config.model_type == "qwen3_5":
        model_class = getattr(
            transformers,
            "AutoModelForImageTextToText",
            getattr(transformers, "AutoModelForMultimodalLM", None),
        )
        if model_class is None:
            raise RuntimeError("installed Transformers cannot load the D1 Qwen3.5 checkpoint")
    else:
        model_class = AutoModelForCausalLM
    model = model_class.from_pretrained(spec["id"], **kwargs)
    if smoke:
        model.to(runtime_device(smoke=True))
    if for_training:
        model.config.use_cache = False
    return model


def lora_config(config: Mapping[str, Any], *, smoke: bool) -> Any:
    from peft import LoraConfig

    lora = config["training"]["lora"]
    return LoraConfig(
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora["target_modules"]),
        exclude_modules=None if smoke else str(lora["exclude_modules_regex"]),
    )


def load_parent_adapter_model(
    config: Mapping[str, Any],
    *,
    seed: int,
    smoke: bool,
    trainable: bool,
    backend: str = "trl",
) -> Any:
    from peft import PeftModel

    parent_path = config["lineage"].get("parent_config")
    if not parent_path:
        raise ValueError(f"{config['rung']} has no trainable parent adapter")
    parent = load_config(parent_path)
    parent_seed = int(parent["seeds"][0])
    parent_adapter = adapter_path(parent, seed=parent_seed, smoke=smoke, backend=backend)
    if not (parent_adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"parent adapter is missing for {config['rung']}: {parent_adapter}")
    base = load_base_model(
        config,
        smoke=smoke,
        for_training=trainable,
        quantized_training=trainable and not smoke,
    )
    return PeftModel.from_pretrained(base, parent_adapter, is_trainable=trainable)


def load_unsloth_parent_adapter_model(
    config: Mapping[str, Any], *, seed: int, trainable: bool, backend: str = "unsloth"
) -> Any:
    from peft import PeftModel
    from unsloth import FastLanguageModel

    parent_path = config["lineage"].get("parent_config")
    if not parent_path:
        raise ValueError(f"{config['rung']} has no parent adapter")
    parent = load_config(parent_path)
    parent_seed = int(parent["seeds"][0])
    parent_adapter = adapter_path(parent, seed=parent_seed, smoke=False, backend=backend)
    if not (parent_adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Unsloth parent adapter is missing: {parent_adapter}")
    spec = model_spec(config, smoke=False)
    training = config["training"]
    max_seq_length = int(
        training.get(
            "max_length",
            int(training.get("max_prompt_length", 1536))
            + int(training.get("max_completion_length", 384)),
        )
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec["id"],
        revision=spec["revision"],
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
        fast_inference=False,
    )
    model = PeftModel.from_pretrained(model, parent_adapter, is_trainable=trainable)
    return model, tokenizer


def load_rung_model(config: Mapping[str, Any], *, seed: int, smoke: bool, backend: str) -> Any:
    if config["rung"] == "r0":
        return load_base_model(config, smoke=smoke, for_training=False, quantized_training=False)
    from peft import PeftModel

    base = load_base_model(config, smoke=smoke, for_training=False, quantized_training=False)
    path = adapter_path(config, seed=seed, smoke=smoke, backend=backend)
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter is missing: {path}")
    return PeftModel.from_pretrained(base, path, is_trainable=False)


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    return max(checkpoints)[1] if checkpoints else None
