"""R3 DPO launcher, continuing the cumulative R2 adapter."""

from __future__ import annotations

import argparse
from typing import Any

from forge.train.common import begin_run, complete_training_receipt, completed_receipt
from forge.train.config import adapter_path, checkpoint_root, load_config, select_seed
from forge.train.data import load_dpo_dataset
from forge.train.runtime import (
    activate_unsloth_runtime,
    latest_checkpoint,
    load_parent_adapter_model,
    load_tokenizer,
    load_unsloth_parent_adapter_model,
    runtime_device,
    seed_everything,
    versioned_training_argument,
)


def run(config_path: str, *, seed: int | None, smoke: bool, backend: str = "trl") -> dict[str, Any]:
    if backend == "unsloth":
        activate_unsloth_runtime()
    from trl import DPOConfig, DPOTrainer

    config = load_config(config_path)
    if config["stage"] != "dpo":
        raise ValueError(f"train-dpo cannot execute {config['stage']} config {config_path}")
    selected_seed = select_seed(config, seed)
    if smoke and backend != "trl":
        raise RuntimeError("Unsloth is CUDA-only and is not a local smoke backend")
    from forge.train.preflight import require_backend_allowed

    require_backend_allowed(config, backend=backend, smoke=smoke)
    existing = completed_receipt(config, seed=selected_seed, smoke=smoke, backend=backend)
    if existing is not None:
        print(f"DPO already complete: {existing['adapter_path']}")
        return existing
    started_at, started_monotonic = begin_run(smoke=smoke)
    seed_everything(selected_seed)
    if backend == "unsloth":
        policy, tokenizer = load_unsloth_parent_adapter_model(
            config, seed=selected_seed, trainable=True
        )
        reference, _ = load_unsloth_parent_adapter_model(
            config, seed=selected_seed, trainable=False
        )
    else:
        tokenizer = load_tokenizer(config, smoke=smoke)
        policy = load_parent_adapter_model(
            config, seed=selected_seed, smoke=smoke, trainable=True, backend=backend
        )
        reference = load_parent_adapter_model(
            config, seed=selected_seed, smoke=smoke, trainable=False, backend=backend
        )
    training = config["training"]
    root = checkpoint_root(config, seed=selected_seed, smoke=smoke, backend=backend)
    trainer_dir = root / "trainer"
    args_kwargs: dict[str, Any] = {
        **versioned_training_argument(
            DPOConfig,
            "max_prompt_length",
            int(training["smoke_max_prompt_length"] if smoke else training["max_prompt_length"]),
        ),
    }
    args = DPOConfig(
        output_dir=str(trainer_dir),
        seed=selected_seed,
        data_seed=selected_seed,
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(training["epochs"]),
        max_steps=int(training["smoke_max_steps"]) if smoke else -1,
        per_device_train_batch_size=int(training["batch_size"]),
        gradient_accumulation_steps=(1 if smoke else int(training["gradient_accumulation_steps"])),
        max_length=int(training["smoke_max_length"] if smoke else training["max_length"]),
        beta=float(training["beta"]),
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
        **args_kwargs,
    )
    trainer = DPOTrainer(
        model=policy,
        ref_model=reference,
        args=args,
        train_dataset=load_dpo_dataset(config, smoke=smoke),
        processing_class=tokenizer,
    )
    resume = latest_checkpoint(trainer_dir)
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    output = adapter_path(config, seed=selected_seed, smoke=smoke, backend=backend)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    receipt = complete_training_receipt(
        config=config,
        seed=selected_seed,
        smoke=smoke,
        backend=backend,
        started_at=started_at,
        started_monotonic=started_monotonic,
        adapter_dir=output,
        train_metrics=result.metrics,
        resume_checkpoint=resume,
    )
    print(f"DPO complete: seed={selected_seed} adapter_sha256={receipt['adapter_sha256']}")
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
