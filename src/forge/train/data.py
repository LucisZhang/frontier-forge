"""Frozen-data adapters for SFT, DPO, GRPO, and evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb

from forge.train.config import canonical_json, resolve_path, sha256_file


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _rank(seed: int, complaint_id: int) -> bytes:
    return hashlib.blake2b(f"{seed}:{complaint_id}".encode(), digest_size=16).digest()


def compact_model_input(value: Mapping[str, Any], *, max_narrative_chars: int) -> dict[str, Any]:
    result = {
        "complaint_id": int(value["complaint_id"]),
        "narrative": str(value["narrative"]),
        "source_product": value.get("source_product"),
        "source_issue": value.get("source_issue"),
        "source_company": value.get("source_company"),
    }
    narrative = str(result["narrative"])
    if len(narrative) > max_narrative_chars:
        half = max_narrative_chars // 2
        result["narrative"] = narrative[:half] + "\n[...middle truncated...]\n" + narrative[-half:]
    return result


def prompt_messages(
    model_input: Mapping[str, Any], *, prompt: str, max_narrative_chars: int
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": canonical_json(
                compact_model_input(model_input, max_narrative_chars=max_narrative_chars)
            ),
        },
    ]


def narrative_char_limit(config: Mapping[str, Any], *, smoke: bool) -> int:
    prompt = config["prompt"]
    key = "narrative_char_cap_smoke" if smoke else "narrative_char_cap_full"
    return int(prompt[key])


def training_data_path(config: Mapping[str, Any], *, smoke: bool) -> Path:
    data = config["data"]
    return resolve_path(data["smoke_path"] if smoke else data["path"])


def validate_training_data(config: Mapping[str, Any], *, smoke: bool) -> Path:
    path = training_data_path(config, smoke=smoke)
    if not path.is_file():
        raise FileNotFoundError(f"training corpus is missing: {path}")
    if not smoke and config["rung"] != "r1b":
        if sha256_file(path) != config["data"]["sha256"]:
            raise ValueError(f"frozen training corpus hash mismatch: {path}")
        rows = sum(1 for line in path.read_text().splitlines() if line)
        if rows != int(config["data"]["rows"]):
            raise ValueError(f"frozen training corpus row mismatch: {path}")
    if not smoke and config["rung"] == "r1b":
        manifest = json.loads(resolve_path(config["data"]["prepared_manifest"]).read_text())
        artifact = manifest["artifact"]
        if sha256_file(path) != artifact["sha256"] or int(artifact["rows"]) != 20_000:
            raise ValueError("R1b artifact differs from its contamination-screened manifest")
    return path


def load_sft_dataset(config: Mapping[str, Any], *, smoke: bool) -> Any:
    from datasets import Dataset

    path = validate_training_data(config, smoke=smoke)
    records = _read_jsonl(path)
    if smoke:
        records = records[: int(config["training"]["smoke_max_rows"])]
    prompt = resolve_path(config["prompt"]["path"]).read_text().strip()
    narrative_limit = narrative_char_limit(config, smoke=smoke)
    rows = []
    for record in records:
        target = record.get("target")
        if not isinstance(target, dict):
            raise ValueError(f"SFT target is not a JSON object for {record.get('complaint_id')}")
        rows.append(
            {
                "prompt": prompt_messages(
                    record["model_input"],
                    prompt=prompt,
                    max_narrative_chars=narrative_limit,
                ),
                "completion": [{"role": "assistant", "content": canonical_json(target)}],
                "complaint_id": int(record["complaint_id"]),
            }
        )
    return Dataset.from_list(rows)


def load_dpo_dataset(config: Mapping[str, Any], *, smoke: bool) -> Any:
    from datasets import Dataset

    records = _read_jsonl(validate_training_data(config, smoke=smoke))
    if smoke:
        records = records[: int(config["training"]["smoke_max_rows"])]
    prompt = resolve_path(config["prompt"]["path"]).read_text().strip()
    narrative_limit = narrative_char_limit(config, smoke=smoke)
    rows = []
    for record in records:
        chosen = json.loads(record["chosen"])
        rejected = json.loads(record["rejected"])
        rows.append(
            {
                "prompt": prompt_messages(
                    record["model_input"],
                    prompt=prompt,
                    max_narrative_chars=narrative_limit,
                ),
                "chosen": [{"role": "assistant", "content": canonical_json(chosen)}],
                "rejected": [{"role": "assistant", "content": canonical_json(rejected)}],
                "complaint_id": int(record["complaint_id"]),
            }
        )
    return Dataset.from_list(rows)


def load_grpo_dataset(config: Mapping[str, Any], *, smoke: bool) -> Any:
    from datasets import Dataset

    records = _read_jsonl(validate_training_data(config, smoke=smoke))
    if smoke:
        records = records[: int(config["training"]["smoke_max_rows"])]
    prompt = resolve_path(config["prompt"]["path"]).read_text().strip()
    narrative_limit = narrative_char_limit(config, smoke=smoke)
    rows = []
    for record in records:
        rows.append(
            {
                "prompt": prompt_messages(
                    record["model_input"],
                    prompt=prompt,
                    max_narrative_chars=narrative_limit,
                ),
                "gold": canonical_json(record["target"]),
                "complaint_id": int(record["complaint_id"]),
            }
        )
    return Dataset.from_list(rows)


def _smoke_split_path(split: str) -> Path:
    return resolve_path(f"data/smoke/splits/{split}.parquet")


def select_eval_rows(config: Mapping[str, Any], *, split: str, smoke: bool) -> list[dict[str, Any]]:
    split_config = config["evaluation"]["test_splits"][split]
    path = _smoke_split_path(split) if smoke else resolve_path(split_config["path"])
    if not path.is_file():
        raise FileNotFoundError(f"evaluation split is missing: {path}")
    if not smoke and sha256_file(path) != split_config["payload_sha256"]:
        raise ValueError(f"frozen {split} payload hash mismatch")
    limit = int(
        config["evaluation"]["smoke_rows_per_split"]
        if smoke
        else config["evaluation"]["rows_per_split"]
    )
    seed = int(config["evaluation"]["selection_seed"])
    con = duckdb.connect()
    try:
        ids = [
            int(row[0])
            for row in con.execute(
                "SELECT complaint_id FROM read_parquet(?)", [str(path)]
            ).fetchall()
        ]
        chosen = sorted(ids, key=lambda value: (_rank(seed, value), value))[:limit]
        placeholders = ",".join("?" for _ in chosen)
        rows = con.execute(
            f"""
            SELECT complaint_id, narrative, source_product, source_issue,
                   source_company, label_json
            FROM read_parquet(?)
            WHERE complaint_id IN ({placeholders})
            """,
            [str(path), *chosen],
        ).fetchall()
    finally:
        con.close()
    by_id = {
        int(row[0]): {
            "complaint_id": int(row[0]),
            "narrative": row[1],
            "source_product": row[2],
            "source_issue": row[3],
            "source_company": row[4],
            "label": json.loads(row[5]),
        }
        for row in rows
    }
    if len(by_id) != limit:
        raise RuntimeError(f"selected {limit} {split} rows but loaded {len(by_id)}")
    return [by_id[complaint_id] for complaint_id in chosen]


def load_general_probe(config: Mapping[str, Any], *, smoke: bool) -> list[dict[str, str]]:
    records = _read_jsonl(resolve_path(config["evaluation"]["general_probe_path"]))
    return records[:2] if smoke else records
