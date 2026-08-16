"""R4 GRPO launcher with the pure scorer-v2 verifier reward."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from forge.train.common import begin_run, complete_training_receipt, completed_receipt
from forge.train.config import adapter_path, checkpoint_root, load_config, select_seed
from forge.train.data import load_grpo_dataset
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
from forge.verify.verifier import score


def _completion_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
        item = value[-1]
        if isinstance(item, Mapping) and isinstance(item.get("content"), str):
            return str(item["content"])
    return str(value)


@dataclass
class RewardAudit:
    rewards: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    abstentions: list[bool] = field(default_factory=list)

    def reward(self, completions: list[object], gold: list[str], **_kwargs: Any) -> list[float]:
        values: list[float] = []
        for completion, expected_json in zip(completions, gold, strict=True):
            text = _completion_text(completion)
            item = score({"label": json.loads(expected_json)}, text)
            values.append(item.reward)
            self.rewards.append(item.reward)
            self.lengths.append(len(text))
            self.abstentions.append('"name":"request_more_info"' in text.replace(" ", ""))
        return values

    def summary(self) -> dict[str, Any]:
        count = len(self.rewards)
        return {
            "scorer_version": 2,
            "completions_scored": count,
            "mean_reward": sum(self.rewards) / count if count else 0.0,
            "mean_completion_chars": sum(self.lengths) / count if count else 0.0,
            "abstention_rate": sum(self.abstentions) / count if count else 0.0,
        }


def run(config_path: str, *, seed: int | None, smoke: bool, backend: str = "trl") -> dict[str, Any]:
    if backend == "unsloth":
        activate_unsloth_runtime()
    from trl import GRPOConfig, GRPOTrainer

    config = load_config(config_path)
    if config["stage"] != "grpo":
        raise ValueError(f"train-grpo cannot execute {config['stage']} config {config_path}")
    selected_seed = select_seed(config, seed)
    if smoke and backend != "trl":
        raise RuntimeError("Unsloth is CUDA-only and is not a local smoke backend")
    from forge.train.preflight import require_backend_allowed

    require_backend_allowed(config, backend=backend, smoke=smoke)
    existing = completed_receipt(config, seed=selected_seed, smoke=smoke, backend=backend)
    if existing is not None:
        print(f"GRPO already complete: {existing['adapter_path']}")
        return existing
    started_at, started_monotonic = begin_run(smoke=smoke)
    seed_everything(selected_seed)
    if backend == "unsloth":
        model, tokenizer = load_unsloth_parent_adapter_model(
            config, seed=selected_seed, trainable=True
        )
    else:
        tokenizer = load_tokenizer(config, smoke=smoke)
        model = load_parent_adapter_model(
            config, seed=selected_seed, smoke=smoke, trainable=True, backend=backend
        )
    training = config["training"]
    root = checkpoint_root(config, seed=selected_seed, smoke=smoke, backend=backend)
    trainer_dir = root / "trainer"
    generations = int(training["smoke_num_generations"] if smoke else training["num_generations"])
    args_kwargs: dict[str, Any] = {
        **versioned_training_argument(
            GRPOConfig,
            "max_prompt_length",
            int(training["smoke_max_prompt_length"] if smoke else training["max_prompt_length"]),
        ),
    }
    args = GRPOConfig(
        output_dir=str(trainer_dir),
        seed=selected_seed,
        data_seed=selected_seed,
        learning_rate=float(training["learning_rate"]),
        max_steps=int(training["smoke_max_steps"] if smoke else training["max_steps"]),
        per_device_train_batch_size=int(training["batch_size"]),
        gradient_accumulation_steps=(
            generations if smoke else int(training["gradient_accumulation_steps"])
        ),
        max_completion_length=int(
            training["smoke_max_completion_length"] if smoke else training["max_completion_length"]
        ),
        num_generations=generations,
        beta=float(training["beta"]),
        use_vllm=False,
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
        remove_unused_columns=False,
        **args_kwargs,
    )
    audit = RewardAudit()
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=audit.reward,
        args=args,
        train_dataset=load_grpo_dataset(config, smoke=smoke),
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
        reward_audit=audit.summary(),
    )
    print(f"GRPO complete: seed={selected_seed} adapter_sha256={receipt['adapter_sha256']}")
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
