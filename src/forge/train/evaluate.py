"""Frozen TEST evaluation with bootstrap CIs, diagnostics, and hacking probes."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from forge.teacher.filters import breakdown_dict
from forge.train.artifacts import write_json_atomic, write_jsonl_atomic
from forge.train.config import (
    CONFIG_PATHS,
    adapter_path,
    config_dataset_hash,
    evaluation_root,
    git_sha,
    load_config,
    model_spec,
    relative_path,
    select_seed,
)
from forge.train.data import (
    compact_model_input,
    load_general_probe,
    narrative_char_limit,
    prompt_messages,
    select_eval_rows,
)
from forge.train.runtime import load_rung_model, load_tokenizer, package_versions, seed_everything
from forge.verify.verifier import ScoreBreakdown, score


def _mean(values: list[bool | float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def bootstrap_ci(values: list[float], *, resamples: int, seed: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    generator = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        means[index] = array[generator.integers(0, len(array), size=len(array))].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _model_device(model: Any) -> Any:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def generate_texts(
    model: Any,
    tokenizer: Any,
    messages: list[list[dict[str, str]]],
    *,
    batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> list[str]:
    import torch

    outputs: list[str] = []
    model.eval()
    for start in range(0, len(messages), batch_size):
        batch = messages[start : start + batch_size]
        rendered = [
            tokenizer.apply_chat_template(
                item,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for item in batch
        ]
        tokens = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        device = _model_device(model)
        tokens = {key: value.to(device) for key, value in tokens.items()}
        input_width = int(tokens["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **tokens,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        for sequence in generated:
            outputs.append(
                tokenizer.decode(sequence[input_width:], skip_special_tokens=True).strip()
            )
    return outputs


def _failure_categories(item: ScoreBreakdown) -> list[str]:
    categories = []
    if not item.json_valid:
        categories.append("invalid_json")
    elif not item.schema_valid:
        categories.append("schema_invalid")
    if not item.urgency_match:
        categories.append("wrong_urgency")
    if not item.ambiguity_flag_match:
        categories.append("wrong_ambiguity")
    if not item.tool_choice_match:
        categories.append("wrong_tool")
    if not item.tool_arguments_valid:
        categories.append("invalid_tool_arguments")
    if item.task_success and not (item.product_match and item.issue_match and item.company_match):
        categories.append("secondary_field_mismatch_only")
    return categories


def _degenerate_abstention(label: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "product": label["product"],
        "issue": label["issue"],
        "company": label["company"],
        "urgency": label["urgency"],
        "ambiguity_flag": True,
        "tool_call": {
            "name": "request_more_info",
            "arguments": {
                "missing_fields": ["details"],
                "question": "Please provide the missing action-critical details.",
            },
        },
    }


def _probe_rewards(rows: list[dict[str, Any]]) -> dict[str, Any]:
    original = []
    inflated = []
    fenced = []
    degenerate = []
    degenerate_success = []
    abstentions = []
    for row in rows:
        label = row["label"]
        raw = row["model_output"]
        original_item = score({"label": label}, raw)
        inflated_item = score({"label": label}, raw + "\n" + ("x" * 512))
        fenced_item = score({"label": label}, f"```json\n{raw}\n```")
        degenerate_item = score({"label": label}, _degenerate_abstention(label))
        original.append(original_item.reward)
        inflated.append(inflated_item.reward)
        fenced.append(fenced_item.reward)
        degenerate.append(degenerate_item.reward)
        degenerate_success.append(degenerate_item.task_success)
        abstentions.append('"name":"request_more_info"' in raw.replace(" ", ""))
    return {
        "length_inflation": {
            "mean_reward_before": _mean(original),
            "mean_reward_after": _mean(inflated),
            "reward_increase_rows": sum(
                after > before for before, after in zip(original, inflated, strict=True)
            ),
        },
        "format_exploitation": {
            "mean_reward_before": _mean(original),
            "mean_reward_markdown_fenced": _mean(fenced),
            "reward_increase_rows": sum(
                after > before for before, after in zip(original, fenced, strict=True)
            ),
        },
        "degenerate_abstention": {
            "model_abstention_rate": _mean(abstentions),
            "always_abstain_mean_reward": _mean(degenerate),
            "always_abstain_task_success": _mean(degenerate_success),
        },
    }


def summarize(rows: list[dict[str, Any]], *, bootstrap_resamples: int, seed: int) -> dict[str, Any]:
    scores = [row["score"] for row in rows]
    task_values = [float(item["task_success"]) for item in scores]
    failures: Counter[str] = Counter()
    for row in rows:
        failures.update(row["failure_categories"])
    return {
        "rows": len(rows),
        "task_success": _mean([item["task_success"] for item in scores]),
        "schema_valid": _mean([item["schema_valid"] for item in scores]),
        "tool_acc": _mean([item["decision_checks"]["tool_choice"] for item in scores]),
        "field_f1": {
            "product": _mean([item["secondary_metrics"]["product_match"] for item in scores]),
            "issue": _mean(
                [item["secondary_metrics"]["issue_normalized_match"] for item in scores]
            ),
            "company": _mean(
                [item["secondary_metrics"]["company_normalized_match"] for item in scores]
            ),
        },
        "abstain_correct": _mean(
            [item["secondary_metrics"]["abstention_correct"] for item in scores]
        ),
        "urgency_match": _mean([item["decision_checks"]["urgency"] for item in scores]),
        "ambiguity_flag_match": _mean(
            [item["decision_checks"]["ambiguity_flag"] for item in scores]
        ),
        "tool_arguments_structural_valid": _mean(
            [item["decision_checks"]["tool_arguments_structural"] for item in scores]
        ),
        "mean_reward": _mean([item["reward"] for item in scores]),
        "ci95": bootstrap_ci(task_values, resamples=bootstrap_resamples, seed=seed),
        "failure_categories_nonexclusive": dict(sorted(failures.items())),
    }


def _evaluate_target(
    config: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    smoke: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluation = config["evaluation"]
    prompt = Path(config["prompt"]["path"])
    if not prompt.is_absolute():
        from forge.train.config import resolve_path

        prompt = resolve_path(prompt)
    system_prompt = prompt.read_text().strip()
    narrative_limit = narrative_char_limit(config, smoke=smoke)
    all_rows: list[dict[str, Any]] = []
    per_split: dict[str, Any] = {}
    for split in ("test_iid", "test_drift"):
        selected = select_eval_rows(config, split=split, smoke=smoke)
        messages = [
            prompt_messages(
                compact_model_input(row, max_narrative_chars=narrative_limit),
                prompt=system_prompt,
                max_narrative_chars=narrative_limit,
            )
            for row in selected
        ]
        generated = generate_texts(
            model,
            tokenizer,
            messages,
            batch_size=int(evaluation["smoke_batch_size"] if smoke else evaluation["batch_size"]),
            max_prompt_tokens=int(
                evaluation["smoke_max_prompt_tokens"] if smoke else evaluation["max_prompt_tokens"]
            ),
            max_new_tokens=int(
                evaluation["smoke_max_new_tokens"] if smoke else evaluation["max_new_tokens"]
            ),
        )
        split_rows = []
        for row, output in zip(selected, generated, strict=True):
            item = score({"label": row["label"]}, output)
            record = {
                "split": split,
                "complaint_id": int(row["complaint_id"]),
                "label": row["label"],
                "model_output": output,
                "score": breakdown_dict(item),
                "failure_categories": _failure_categories(item),
            }
            split_rows.append(record)
            all_rows.append(record)
        per_split[split] = summarize(
            split_rows,
            bootstrap_resamples=int(evaluation["bootstrap_resamples"]),
            seed=int(evaluation["bootstrap_seed"]),
        )
    return all_rows, per_split


def _evaluate_general(
    config: dict[str, Any], *, model: Any, tokenizer: Any, smoke: bool
) -> dict[str, Any]:
    probes = load_general_probe(config, smoke=smoke)
    messages = [
        [
            {"role": "system", "content": "Follow the instruction and return only its answer."},
            {"role": "user", "content": row["prompt"]},
        ]
        for row in probes
    ]
    outputs = generate_texts(
        model,
        tokenizer,
        messages,
        batch_size=1 if smoke else int(config["evaluation"]["batch_size"]),
        max_prompt_tokens=256,
        max_new_tokens=32,
    )
    rows = []
    for probe, output in zip(probes, outputs, strict=True):
        normalized = output.strip().casefold()
        expected = probe["expected"].strip().casefold()
        rows.append({**probe, "output": output, "correct": normalized == expected})
    return {"rows": rows, "accuracy": _mean([row["correct"] for row in rows])}


def evaluate_one(
    config_path: str | Path,
    *,
    seed: int | None,
    smoke: bool,
    backend: str,
) -> dict[str, Any]:
    config = load_config(config_path)
    selected_seed = select_seed(config, seed)
    root = evaluation_root(config, seed=selected_seed, smoke=smoke, backend=backend)
    metrics_path = root / "metrics.json"
    if metrics_path.is_file():
        existing = json.loads(metrics_path.read_text())
        if (
            existing.get("status") == "complete"
            and existing.get("config_hash") == config["_config_hash"]
            and existing.get("git_sha") == git_sha()
        ):
            print(f"evaluation already complete: {relative_path(metrics_path)}")
            return existing
        raise RuntimeError(
            f"existing evaluation conflicts with current code/config: {metrics_path}"
        )
    seed_everything(selected_seed)
    tokenizer = load_tokenizer(config, smoke=smoke)
    model = load_rung_model(config, seed=selected_seed, smoke=smoke, backend=backend)
    target_rows, per_split = _evaluate_target(config, model=model, tokenizer=tokenizer, smoke=smoke)
    combined = summarize(
        target_rows,
        bootstrap_resamples=int(config["evaluation"]["bootstrap_resamples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    general = _evaluate_general(config, model=model, tokenizer=tokenizer, smoke=smoke)
    probes = _probe_rewards(target_rows)
    write_jsonl_atomic(root / "predictions.jsonl", target_rows)
    receipt = {
        "version": 1,
        "status": "complete",
        "phase": 3,
        "mode": "smoke" if smoke else "full",
        "rung": config["rung"],
        "backend": backend,
        "seed": selected_seed,
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "git_sha": git_sha(),
        "dataset_hash": config_dataset_hash(config, smoke=smoke),
        "model": dict(model_spec(config, smoke=smoke)),
        "training_time_quantization": (
            "none"
            if config["rung"] == "r0"
            else config["training"]["quantization_smoke" if smoke else "quantization_full"]
        ),
        "evaluation_precision": config["model"]["smoke" if smoke else "full"]["dtype"],
        "deployment_quantization": "not_evaluated_here",
        "metrics": combined,
        "split_metrics": per_split,
        "general_instruction_regression": general,
        "reward_hacking_probes": probes,
        "predictions_path": relative_path(root / "predictions.jsonl"),
        "packages": package_versions(),
        "finished_at": datetime.now(UTC).isoformat(),
        "notes": (
            "Local smoke evidence only; excluded from results/runs.jsonl."
            if smoke
            else "Frozen deterministic TEST subset; 1,000 fixed-seed bootstrap resamples."
        ),
    }
    write_json_atomic(metrics_path, receipt)
    print(
        f"evaluation complete: rung={config['rung']} backend={backend} seed={selected_seed} "
        f"task_success={combined['task_success']:.4f} ci95={combined['ci95']}"
    )
    return receipt


def available_configs(*, smoke: bool, backend: str) -> list[Path]:
    available = []
    for path in CONFIG_PATHS:
        config = load_config(path)
        seed = int(config["seeds"][0])
        if (
            config["rung"] == "r0"
            or (
                adapter_path(config, seed=seed, smoke=smoke, backend=backend)
                / "adapter_config.json"
            ).is_file()
        ):
            available.append(path)
    return available


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config")
    group.add_argument("--available", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backend", choices=("trl", "unsloth"), default="trl")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    paths = (
        available_configs(smoke=args.smoke, backend=args.backend)
        if args.available
        else [Path(args.config)]
    )
    if not paths:
        raise RuntimeError("no Phase 3 rung is available for evaluation")
    receipts = [
        evaluate_one(
            path,
            seed=args.seed,
            smoke=args.smoke,
            backend=args.backend,
        )
        for path in paths
    ]
    if args.smoke and os.environ.get("FORGE_DEFER_FINALIZE") != "1":
        from forge.train.finalize import finalize_smoke

        for receipt in receipts:
            finalize_smoke(receipt)


if __name__ == "__main__":
    main()
