"""Build the frozen, length-controlled Phase 4 triage workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.train.artifacts import write_jsonl_atomic
from forge.train.config import REPO_ROOT, canonical_json, load_config, relative_path, sha256_file
from forge.train.data import compact_model_input, prompt_messages, select_eval_rows

from .config import load_phase4_config, phase4_workload_path, workload_contract_hash


def _allocation(total: int, weights: Sequence[float]) -> list[int]:
    raw = [total * float(weight) for weight in weights]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def _stable_rank(seed: int, *parts: object) -> str:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode()).hexdigest()


def _rendered_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return len(token_ids)


def _smoke_token_count(messages: list[dict[str, str]]) -> int:
    rendered = "\n".join(item["content"] for item in messages)
    return max(1, len(rendered.encode("utf-8")) // 4)


def _smoke_rows() -> list[dict[str, Any]]:
    path = REPO_ROOT / "data/smoke/phase2/sft_rule.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    result = []
    for index, record in enumerate(records):
        result.append(
            {
                "split": "test_iid" if index % 2 == 0 else "test_drift",
                "complaint_id": int(record["complaint_id"]),
                "model_input": record["model_input"],
                "label": record["target"],
            }
        )
    if not result:
        raise RuntimeError("smoke Phase 2 records are missing")
    return result


def _full_rows(split: str) -> list[dict[str, Any]]:
    source_config = load_config("configs/r1b_sft_rule_20k.yaml")
    splits = ("test_iid", "test_drift") if split == "mixed" else (split,)
    result: list[dict[str, Any]] = []
    for name in splits:
        for row in select_eval_rows(source_config, split=name, smoke=False):
            result.append(
                {
                    "split": name,
                    "complaint_id": int(row["complaint_id"]),
                    "model_input": compact_model_input(row, max_narrative_chars=3250),
                    "label": row["label"],
                }
            )
    return result


def _load_tokenizer(config: Mapping[str, Any]) -> Any:
    from transformers import AutoTokenizer

    model_path = REPO_ROOT / str(config["model"]["artifact_path"])
    if not model_path.is_dir():
        raise FileNotFoundError(f"serving artifact is missing: {model_path}")
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


def _choose_rows(
    candidates: list[dict[str, Any]],
    *,
    targets: list[int],
    weights: list[float],
    total: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(candidates) < total:
        raise RuntimeError(f"workload needs {total} unique rows but only {len(candidates)} exist")
    counts = _allocation(total, weights)
    unused = {int(item["complaint_id"]): item for item in candidates}
    selected: list[dict[str, Any]] = []
    for target, count in zip(targets, counts, strict=True):
        nearest = sorted(
            unused.values(),
            key=lambda item: (
                abs(int(item["prompt_tokens"]) - target),
                _stable_rank(seed, target, item["complaint_id"]),
            ),
        )[:count]
        for item in nearest:
            chosen = dict(item)
            chosen["input_token_target"] = target
            selected.append(chosen)
            del unused[int(item["complaint_id"])]
    return selected


def _assign_output_caps(
    rows: list[dict[str, Any]], *, caps: list[int], weights: list[float], seed: int
) -> None:
    values: list[int] = []
    for cap, count in zip(caps, _allocation(len(rows), weights), strict=True):
        values.extend([cap] * count)
    random.Random(seed).shuffle(values)
    for row, value in zip(rows, values, strict=True):
        row["max_tokens"] = value


def build_workload(config_path: str | Path, *, smoke: bool) -> dict[str, Any]:
    config = load_phase4_config(config_path)
    output_path = phase4_workload_path(config, smoke=smoke)
    if output_path.is_file():
        records = [json.loads(line) for line in output_path.read_text().splitlines() if line]
        if records and all(
            item.get("workload_contract_hash") == workload_contract_hash(config) for item in records
        ):
            return {
                "path": relative_path(output_path),
                "sha256": sha256_file(output_path),
                "rows": len(records),
                "reused": True,
            }
        raise RuntimeError(f"existing workload conflicts with config: {output_path}")

    workload = config["workload"]
    prompt = (REPO_ROOT / "configs/train_prompts/triage_v2_compact.txt").read_text().strip()
    rows = _smoke_rows() if smoke else _full_rows(str(workload["split"]))
    tokenizer = None if smoke else _load_tokenizer(config)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        messages = prompt_messages(row["model_input"], prompt=prompt, max_narrative_chars=3250)
        candidates.append(
            {
                "split": row["split"],
                "complaint_id": int(row["complaint_id"]),
                "messages": messages,
                "label": row["label"],
                "prompt_tokens": (
                    _smoke_token_count(messages)
                    if tokenizer is None
                    else _rendered_token_count(tokenizer, messages)
                ),
            }
        )
    total = min(int(workload["rows"]), len(candidates)) if smoke else int(workload["rows"])
    selected = _choose_rows(
        candidates,
        targets=[int(item) for item in workload["input_token_targets"]],
        weights=[float(item) for item in workload["input_token_weights"]],
        total=total,
        seed=int(workload["selection_seed"]),
    )
    _assign_output_caps(
        selected,
        caps=[int(item) for item in workload["output_token_caps"]],
        weights=[float(item) for item in workload["output_token_weights"]],
        seed=int(workload["request_seed"]),
    )
    contract_hash = workload_contract_hash(config)
    for index, row in enumerate(selected):
        row["request_id"] = f"p4-{index:04d}-{row['complaint_id']}"
        row["workload_contract_hash"] = contract_hash
        row["label_sha256"] = hashlib.sha256(canonical_json(row["label"]).encode()).hexdigest()
    selected.sort(key=lambda item: item["request_id"])
    write_jsonl_atomic(output_path, selected)
    return {
        "path": relative_path(output_path),
        "sha256": sha256_file(output_path),
        "rows": len(selected),
        "reused": False,
    }


def load_workload(config: Mapping[str, Any], *, smoke: bool) -> list[dict[str, Any]]:
    path = phase4_workload_path(config, smoke=smoke)
    if not path.is_file():
        raise FileNotFoundError(f"prepare the workload first: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    expected = workload_contract_hash(config)
    if not rows or any(row.get("workload_contract_hash") != expected for row in rows):
        raise RuntimeError("workload contract hash mismatch")
    ids = [row["request_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("workload request ids must be unique")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_workload(args.config, smoke=args.smoke), sort_keys=True))


if __name__ == "__main__":
    main()
