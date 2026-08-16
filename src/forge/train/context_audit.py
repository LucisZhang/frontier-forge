"""Audit every Phase 3 training sequence against the pinned full-tokenizer budgets."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge.train.artifacts import write_json_atomic
from forge.train.config import REPO_ROOT, canonical_json, git_sha, load_config, model_spec
from forge.train.data import (
    narrative_char_limit,
    prompt_messages,
    select_eval_rows,
    validate_training_data,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _token_count(tokenizer: Any, messages: list[dict[str, str]], *, generate: bool) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generate,
    )
    return len(encoded["input_ids"])


def _max_receipt(values: list[tuple[int, int]], *, limit: int) -> dict[str, Any]:
    maximum = max(values, default=(0, 0), key=lambda item: (item[1], item[0]))
    violations = [complaint_id for complaint_id, tokens in values if tokens > limit]
    return {
        "rows": len(values),
        "limit_tokens": limit,
        "max_tokens": maximum[1],
        "max_complaint_id": maximum[0] or None,
        "violations": len(violations),
        "first_violation_ids": violations[:10],
    }


def _sft_audit(config: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    records = _read_jsonl(validate_training_data(config, smoke=False))
    prompt = (REPO_ROOT / config["prompt"]["path"]).read_text().strip()
    cap = narrative_char_limit(config, smoke=False)
    limit = int(config["training"]["max_length"])
    sequences = []
    for record in records:
        messages = prompt_messages(record["model_input"], prompt=prompt, max_narrative_chars=cap)
        completion = {"role": "assistant", "content": canonical_json(record["target"])}
        sequences.append(
            (
                int(record["complaint_id"]),
                _token_count(tokenizer, messages + [completion], generate=False),
            )
        )
    return _max_receipt(sequences, limit=limit)


def _dpo_audit(config: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    records = _read_jsonl(validate_training_data(config, smoke=False))
    prompt = (REPO_ROOT / config["prompt"]["path"]).read_text().strip()
    cap = narrative_char_limit(config, smoke=False)
    limit = int(config["training"]["max_length"])
    chosen = []
    rejected = []
    prompts = []
    for record in records:
        messages = prompt_messages(record["model_input"], prompt=prompt, max_narrative_chars=cap)
        complaint_id = int(record["complaint_id"])
        prompts.append((complaint_id, _token_count(tokenizer, messages, generate=True)))
        for key, output in (("chosen", chosen), ("rejected", rejected)):
            completion = {
                "role": "assistant",
                "content": canonical_json(json.loads(record[key])),
            }
            output.append(
                (complaint_id, _token_count(tokenizer, messages + [completion], generate=False))
            )
    return {
        "prompt": _max_receipt(
            prompts,
            limit=int(config["training"]["max_prompt_length"]),
        ),
        "chosen": _max_receipt(chosen, limit=limit),
        "rejected": _max_receipt(rejected, limit=limit),
    }


def _grpo_prompt_audit(config: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    records = _read_jsonl(validate_training_data(config, smoke=False))
    prompt = (REPO_ROOT / config["prompt"]["path"]).read_text().strip()
    cap = narrative_char_limit(config, smoke=False)
    values = []
    for record in records:
        messages = prompt_messages(record["model_input"], prompt=prompt, max_narrative_chars=cap)
        values.append(
            (int(record["complaint_id"]), _token_count(tokenizer, messages, generate=True))
        )
    return _max_receipt(values, limit=int(config["training"]["max_prompt_length"]))


def _evaluation_audit(config: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    prompt = (REPO_ROOT / config["prompt"]["path"]).read_text().strip()
    cap = narrative_char_limit(config, smoke=False)
    limit = int(config["evaluation"]["max_prompt_tokens"])
    result = {}
    for split in ("test_iid", "test_drift"):
        values = []
        for record in select_eval_rows(config, split=split, smoke=False):
            messages = prompt_messages(record, prompt=prompt, max_narrative_chars=cap)
            values.append(
                (int(record["complaint_id"]), _token_count(tokenizer, messages, generate=True))
            )
        result[split] = _max_receipt(values, limit=limit)
    return result


def _lora_scope_audit(config: dict[str, Any]) -> dict[str, Any]:
    from accelerate import init_empty_weights
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForImageTextToText

    from forge.train.runtime import lora_config

    spec = model_spec(config, smoke=False)
    model_config = AutoConfig.from_pretrained(spec["id"], revision=spec["revision"])
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(model_config)
    wrapped = get_peft_model(model, lora_config(config, smoke=False))
    names = sorted(wrapped.base_model.targeted_module_names)
    vision = [name for name in names if ".visual." in name]
    unexpected = [name for name in names if ".language_model." not in name]
    return {
        "targeted_modules": len(names),
        "language_model_modules": sum(".language_model." in name for name in names),
        "vision_modules": len(vision),
        "unexpected_modules": len(unexpected),
        "violations": len(vision) + len(unexpected),
        "first_targeted_modules": names[:10],
        "first_unexpected_modules": unexpected[:10],
        "method": "meta-tensor PEFT adapter injection; no model weights loaded",
    }


def run() -> dict[str, Any]:
    from transformers import AutoTokenizer

    configs = {
        rung: load_config(path)
        for rung, path in (
            ("r0", "configs/r0_base.yaml"),
            ("r1", "configs/r1_sft_rule.yaml"),
            ("r1b", "configs/r1b_sft_rule_20k.yaml"),
            ("r2", "configs/r2_sft_distilled.yaml"),
            ("r3", "configs/r3_dpo.yaml"),
            ("r4", "configs/r4_grpo.yaml"),
        )
    }
    spec = model_spec(configs["r1"], smoke=False)
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    audits = {
        "r1_sft": _sft_audit(configs["r1"], tokenizer),
        "r1b_sft": _sft_audit(configs["r1b"], tokenizer),
        "r2_sft": _sft_audit(configs["r2"], tokenizer),
        "r3_dpo": _dpo_audit(configs["r3"], tokenizer),
        "r4_grpo_prompt": _grpo_prompt_audit(configs["r4"], tokenizer),
        "frozen_evaluation": _evaluation_audit(configs["r4"], tokenizer),
        "lora_scope": _lora_scope_audit(configs["r1"]),
    }
    violation_count = 0
    for audit in audits.values():
        if "violations" in audit:
            violation_count += int(audit["violations"])
        else:
            violation_count += sum(int(item["violations"]) for item in audit.values())
    receipt = {
        "version": 1,
        "status": "complete" if violation_count == 0 else "failed",
        "phase": 3,
        "tokenizer": dict(spec),
        "narrative_char_cap_full": narrative_char_limit(configs["r1"], smoke=False),
        "config_hashes": {rung: config["_config_hash"] for rung, config in configs.items()},
        "audits": audits,
        "violation_count": violation_count,
        "git_sha": git_sha(),
        "recorded_at": datetime.now(UTC).isoformat(),
        "notes": "Audit only; no model weights loaded and no frozen source artifact modified.",
    }
    output = REPO_ROOT / "results" / "phase3_context_audit.json"
    write_json_atomic(output, receipt)
    if violation_count:
        raise RuntimeError(f"Phase 3 context audit found {violation_count} over-limit sequences")
    print(f"Phase 3 context audit complete: {output.relative_to(REPO_ROOT)}")
    return receipt


if __name__ == "__main__":
    run()
