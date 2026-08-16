"""R1/R1b/R2 supervised fine-tuning launcher."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from forge.train.common import begin_run, complete_training_receipt, completed_receipt
from forge.train.config import adapter_path, checkpoint_root, load_config, select_seed
from forge.train.data import load_sft_dataset
from forge.train.runtime import (
    activate_unsloth_runtime,
    latest_checkpoint,
    load_base_model,
    load_tokenizer,
    lora_config,
    runtime_device,
    seed_everything,
)


def processor_eos_token(processing_class: Any) -> str:
    """Resolve the real EOS token from a tokenizer or multimodal processor."""
    value = getattr(processing_class, "eos_token", None)
    if not isinstance(value, str) or not value:
        value = getattr(getattr(processing_class, "tokenizer", None), "eos_token", None)
    if not isinstance(value, str) or not value:
        raise RuntimeError("Unsloth processing class does not expose a usable EOS token")
    return value


def _trl_train(
    config: dict[str, Any], *, seed: int, smoke: bool
) -> tuple[Any, Any, Any, Path | None]:
    from trl import SFTConfig, SFTTrainer

    seed_everything(seed)
    dataset = load_sft_dataset(config, smoke=smoke)
    tokenizer = load_tokenizer(config, smoke=smoke)
    model = load_base_model(
        config,
        smoke=smoke,
        for_training=True,
        quantized_training=not smoke,
    )
    root = checkpoint_root(config, seed=seed, smoke=smoke, backend="trl")
    trainer_dir = root / "trainer"
    training = config["training"]
    args = SFTConfig(
        output_dir=str(trainer_dir),
        seed=seed,
        data_seed=seed,
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(training["epochs"]),
        max_steps=int(training["smoke_max_steps"]) if smoke else -1,
        per_device_train_batch_size=int(training["batch_size"]),
        gradient_accumulation_steps=(1 if smoke else int(training["gradient_accumulation_steps"])),
        max_length=int(training["smoke_max_length"] if smoke else training["max_length"]),
        completion_only_loss=True,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no" if smoke else "steps",
        save_steps=int(training["save_steps"]),
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=not smoke,
        bf16=not smoke,
        fp16=False,
        use_cpu=runtime_device(smoke=smoke) == "cpu",
        dataloader_pin_memory=False,
        optim="adamw_torch" if smoke else "paged_adamw_8bit",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(config, smoke=smoke),
    )
    resume = latest_checkpoint(trainer_dir)
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    output = adapter_path(config, seed=seed, smoke=smoke, backend="trl")
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    return result, trainer, tokenizer, resume


def _unsloth_train(
    config: dict[str, Any], *, seed: int, smoke: bool
) -> tuple[Any, Any, Any, Path | None]:
    if smoke:
        raise RuntimeError("Unsloth is CUDA-only and is not a local smoke backend")
    from forge.train.preflight import require_backend_allowed

    require_backend_allowed(config, backend="unsloth", smoke=False)
    unsloth = activate_unsloth_runtime()
    from trl import SFTConfig, SFTTrainer

    FastLanguageModel = unsloth.FastLanguageModel
    seed_everything(seed)
    spec = config["model"]["full"]
    training = config["training"]
    lora = training["lora"]
    tokenizer: Any
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec["id"],
        revision=spec["revision"],
        max_seq_length=int(training["max_length"]),
        load_in_4bit=True,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias="none",
        target_modules=list(lora["target_modules"]),
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        max_seq_length=int(training["max_length"]),
    )
    root = checkpoint_root(config, seed=seed, smoke=False, backend="unsloth")
    trainer_dir = root / "trainer"
    args = SFTConfig(
        output_dir=str(trainer_dir),
        seed=seed,
        data_seed=seed,
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(training["epochs"]),
        per_device_train_batch_size=int(training["batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        max_length=int(training["max_length"]),
        eos_token=processor_eos_token(tokenizer),
        completion_only_loss=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=int(training["save_steps"]),
        save_total_limit=2,
        report_to="none",
        bf16=True,
        optim="paged_adamw_8bit",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=load_sft_dataset(config, smoke=False),
        processing_class=tokenizer,
    )
    resume = latest_checkpoint(trainer_dir)
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    output = adapter_path(config, seed=seed, smoke=False, backend="unsloth")
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    return result, trainer, tokenizer, resume


def run(config_path: str, *, seed: int | None, smoke: bool, backend: str) -> dict[str, Any]:
    config = load_config(config_path)
    if config["stage"] != "sft":
        raise ValueError(f"train-sft cannot execute {config['stage']} config {config_path}")
    selected_seed = select_seed(config, seed)
    existing = completed_receipt(config, seed=selected_seed, smoke=smoke, backend=backend)
    if existing is not None:
        print(f"SFT already complete: {existing['adapter_path']}")
        return existing
    started_at, started_monotonic = begin_run(smoke=smoke)
    if backend == "trl":
        result, _trainer, _tokenizer, resume = _trl_train(config, seed=selected_seed, smoke=smoke)
    elif backend == "unsloth":
        result, _trainer, _tokenizer, resume = _unsloth_train(
            config, seed=selected_seed, smoke=smoke
        )
    else:
        raise ValueError("backend must be trl or unsloth")
    receipt = complete_training_receipt(
        config=config,
        seed=selected_seed,
        smoke=smoke,
        backend=backend,
        started_at=started_at,
        started_monotonic=started_monotonic,
        adapter_dir=adapter_path(config, seed=selected_seed, smoke=smoke, backend=backend),
        train_metrics=result.metrics,
        resume_checkpoint=resume,
    )
    print(
        f"SFT complete: rung={config['rung']} backend={backend} seed={selected_seed} "
        f"adapter_sha256={receipt['adapter_sha256']}"
    )
    return receipt


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
