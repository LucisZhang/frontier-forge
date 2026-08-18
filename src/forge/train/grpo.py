"""R4 GRPO launcher with the pure scorer-v2 verifier reward."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge.train.artifacts import write_json_atomic
from forge.train.common import begin_run, complete_training_receipt, completed_receipt, utc_now
from forge.train.config import (
    REPO_ROOT,
    adapter_path,
    checkpoint_root,
    git_sha,
    load_config,
    relative_path,
    select_seed,
)
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

THINK_END_MARKER = "</think>"


def _raw_completion_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
        item = value[-1]
        if isinstance(item, Mapping) and isinstance(item.get("content"), str):
            return str(item["content"])
    return str(value)


def _completion_text(value: object) -> str:
    """Normalize a rollout and defensively discard any completed think block."""

    text = _raw_completion_text(value)
    if THINK_END_MARKER in text:
        text = text.rsplit(THINK_END_MARKER, 1)[1]
    return text.strip()


@dataclass
class RewardAudit:
    rewards: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    abstentions: list[bool] = field(default_factory=list)
    rollout_sample_path: Path | None = None
    rollout_sample_context: dict[str, Any] = field(default_factory=dict)
    rollout_sample: dict[str, Any] | None = None

    def _archive_clean_opening_sample(self, raw: str, normalized: str, reward: float) -> None:
        if self.rollout_sample_path is None or self.rollout_sample is not None:
            return
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(parsed, dict):
            return
        sample = {
            "version": 1,
            "status": "clean_json",
            **self.rollout_sample_context,
            "raw_completion": raw,
            "raw_completion_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "raw_json_valid": True,
            "think_block_marker_present": THINK_END_MARKER in raw,
            "hardening_required": THINK_END_MARKER in raw,
            "normalized_completion": normalized,
            "verifier_reward": reward,
        }
        write_json_atomic(self.rollout_sample_path, sample)
        self.rollout_sample = sample

    def reward(self, completions: list[object], gold: list[str], **_kwargs: Any) -> list[float]:
        values: list[float] = []
        for completion, expected_json in zip(completions, gold, strict=True):
            raw = _raw_completion_text(completion)
            text = _completion_text(completion)
            item = score({"label": json.loads(expected_json)}, text)
            values.append(item.reward)
            self.rewards.append(item.reward)
            self.lengths.append(len(text))
            self.abstentions.append('"name":"request_more_info"' in text.replace(" ", ""))
            self._archive_clean_opening_sample(raw, text, item.reward)
        return values

    def summary(self) -> dict[str, Any]:
        count = len(self.rewards)
        return {
            "scorer_version": 2,
            "completions_scored": count,
            "mean_reward": sum(self.rewards) / count if count else 0.0,
            "mean_completion_chars": sum(self.lengths) / count if count else 0.0,
            "abstention_rate": sum(self.abstentions) / count if count else 0.0,
            "rollout_sample_path": (
                relative_path(self.rollout_sample_path) if self.rollout_sample_path else None
            ),
        }


@dataclass
class RewardSignalGuard:
    opening_steps: int = 10
    observations: dict[int, float] = field(default_factory=dict)

    def observe(self, step: int, logs: Mapping[str, Any]) -> None:
        if step < 1 or step > self.opening_steps:
            return
        value = logs.get("frac_reward_zero_std")
        if value is not None:
            self.observations[step] = float(value)
        if step != self.opening_steps:
            return
        missing = sorted(set(range(1, self.opening_steps + 1)) - self.observations.keys())
        if missing:
            raise RuntimeError(
                "GRPO reward-signal guard could not observe frac_reward_zero_std for opening "
                f"steps: {missing}"
            )
        if all(self.observations[index] == 1.0 for index in range(1, self.opening_steps + 1)):
            raise RuntimeError(
                "GRPO reward-signal guard aborted: frac_reward_zero_std stayed 1.0 for "
                f"the first {self.opening_steps} steps"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "opening_steps": self.opening_steps,
            "observed": {str(key): value for key, value in sorted(self.observations.items())},
            "all_zero_std": (
                len(self.observations) == self.opening_steps
                and all(value == 1.0 for value in self.observations.values())
            ),
        }


def _reward_signal_callback(
    guard: RewardSignalGuard,
    audit: RewardAudit,
    *,
    completion_logging_steps: int,
) -> tuple[Any, dict[str, Any]]:
    from transformers import TrainerCallback

    holder: dict[str, Any] = {}

    class Callback(TrainerCallback):
        def on_log(
            self, _args: Any, state: Any, _control: Any, logs: Any = None, **_kwargs: Any
        ) -> None:
            step = int(state.global_step)
            guard.observe(step, logs or {})
            if (
                step == completion_logging_steps
                and audit.rollout_sample_path is not None
                and audit.rollout_sample is None
            ):
                raise RuntimeError(
                    "opening GRPO completion window contained no clean bare-JSON sample"
                )
            trainer = holder.get("trainer")
            if trainer is not None and step > completion_logging_steps:
                trainer.log_completions = False

    return Callback(), holder


def _synthetic_gold() -> dict[str, Any]:
    return {
        "product": "mortgage",
        "issue": "Foreclosure",
        "company": "Example Bank",
        "urgency": "high",
        "ambiguity_flag": False,
        "tool_call": {
            "name": "escalate_to_regulator",
            "arguments": {"complaint_id": 7, "reason": "Active foreclosure risk."},
        },
    }


def assert_synthetic_gold_reward() -> float:
    """Fail before model loading if the exact-gold verifier path has no positive signal."""

    encoded = json.dumps(_synthetic_gold(), separators=(",", ":"), sort_keys=True)
    reward = RewardAudit().reward([encoded], [encoded])[0]
    if reward <= 0.0:
        raise RuntimeError(f"GRPO smoke reward gate failed: synthetic gold reward={reward}")
    return reward


def _require_clean_rollout_sample(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("opening GRPO rollout archive has no bare-JSON completion")
    sample = json.loads(path.read_text())
    if (
        sample.get("status") != "clean_json"
        or sample.get("raw_json_valid") is not True
        or sample.get("think_block_marker_present") is not False
        or sample.get("hardening_required") is not False
    ):
        raise RuntimeError("opening GRPO rollout archive is not clean bare JSON")
    json.loads(str(sample["raw_completion"]))
    return sample


def reward_signal_receipt_path(
    config: Mapping[str, Any], *, seed: int, smoke: bool, backend: str
) -> Path:
    revision = str(config.get("run_revision", "unversioned"))
    if smoke:
        return (
            checkpoint_root(config, seed=seed, smoke=True, backend=backend) / "reward_signal.json"
        )
    return REPO_ROOT / "results" / f"phase3_r4_reward_signal_{revision}_s{seed}.json"


def _write_reward_signal_receipt(
    config: Mapping[str, Any],
    *,
    seed: int,
    smoke: bool,
    backend: str,
    guard: RewardSignalGuard,
    audit: RewardAudit,
    status: str,
    error: Exception | None = None,
) -> dict[str, Any]:
    summary = guard.summary()
    receipt = {
        "version": 1,
        "status": status,
        "phase": 3.2 if config.get("run_revision") == "phase3_2_fresh_pool" else 3,
        "rung": "r4",
        "run_revision": config.get("run_revision"),
        "mode": "smoke" if smoke else "full",
        "seed": seed,
        "backend": backend,
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "git_sha": git_sha(),
        "guard": summary,
        "passed_opening_steps": (
            len(summary["observed"]) == guard.opening_steps and summary["all_zero_std"] is False
        ),
        "reward_audit": audit.summary(),
        "recorded_at": utc_now(),
        "error": (
            {"type": type(error).__name__, "message": str(error)} if error is not None else None
        ),
    }
    write_json_atomic(
        reward_signal_receipt_path(
            config,
            seed=seed,
            smoke=smoke,
            backend=backend,
        ),
        receipt,
    )
    return receipt


def run(config_path: str, *, seed: int | None, smoke: bool, backend: str = "trl") -> dict[str, Any]:
    if backend == "unsloth":
        activate_unsloth_runtime()
    from trl import GRPOConfig, GRPOTrainer

    config = load_config(config_path)
    if config["stage"] != "grpo":
        raise ValueError(f"train-grpo cannot execute {config['stage']} config {config_path}")
    selected_seed = select_seed(config, seed)
    assert_synthetic_gold_reward()
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
    log_opening_completions = not smoke and selected_seed == int(config["seeds"][0])
    args_kwargs: dict[str, Any] = {
        **versioned_training_argument(
            GRPOConfig,
            "max_prompt_length",
            int(training["smoke_max_prompt_length"] if smoke else training["max_prompt_length"]),
        ),
        **versioned_training_argument(
            GRPOConfig,
            "chat_template_kwargs",
            {"enable_thinking": False},
        ),
        **versioned_training_argument(
            GRPOConfig,
            "log_completions",
            log_opening_completions,
        ),
        **versioned_training_argument(
            GRPOConfig,
            "num_completions_to_print",
            2,
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
    rollout_sample_path = (
        REPO_ROOT
        / "results"
        / f"phase3_r4_rollout_sample_{config.get('run_revision', 'unversioned')}_s0.json"
        if log_opening_completions
        else None
    )
    audit = RewardAudit(
        rollout_sample_path=rollout_sample_path,
        rollout_sample_context={
            "phase": 3,
            "rung": "r4",
            "run_revision": config.get("run_revision"),
            "seed": selected_seed,
            "backend": backend,
            "config_path": config["_config_path"],
            "config_hash": config["_config_hash"],
            "git_sha": git_sha(),
            "source_completion_logs": relative_path(trainer_dir / "completions"),
        },
    )
    signal_guard = RewardSignalGuard(opening_steps=int(training["reward_signal_guard_steps"]))
    signal_callback, callback_holder = _reward_signal_callback(
        signal_guard,
        audit,
        completion_logging_steps=int(training["completion_logging_steps"]),
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=audit.reward,
        args=args,
        train_dataset=load_grpo_dataset(config, smoke=smoke),
        processing_class=tokenizer,
    )
    callback_holder["trainer"] = trainer
    trainer.add_callback(signal_callback)
    resume = latest_checkpoint(trainer_dir)
    try:
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        if rollout_sample_path is not None:
            _require_clean_rollout_sample(rollout_sample_path)
        if not smoke:
            guard_summary = signal_guard.summary()
            if (
                len(guard_summary["observed"]) != signal_guard.opening_steps
                or guard_summary["all_zero_std"] is not False
            ):
                raise RuntimeError(
                    "GRPO reward-signal guard did not prove nonzero variance across the "
                    "complete opening window"
                )
    except Exception as error:
        _write_reward_signal_receipt(
            config,
            seed=selected_seed,
            smoke=smoke,
            backend=backend,
            guard=signal_guard,
            audit=audit,
            status=(
                "aborted-zero-reward-variance"
                if "frac_reward_zero_std stayed 1.0" in str(error)
                else "failed-before-guard-proof"
            ),
            error=error,
        )
        raise
    _write_reward_signal_receipt(
        config,
        seed=selected_seed,
        smoke=smoke,
        backend=backend,
        guard=signal_guard,
        audit=audit,
        status=("smoke-not-applicable" if smoke else "passed-nonzero-reward-variance"),
    )
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
        reward_audit={**audit.summary(), "reward_signal_guard": signal_guard.summary()},
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
