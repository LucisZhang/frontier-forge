"""Materialize the optional contamination-screened R1b 20k rule-label corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb

from forge.data.input_contract import build_model_input
from forge.teacher.filters import contamination_audit
from forge.train.artifacts import write_json_atomic, write_jsonl_atomic
from forge.train.config import (
    REPO_ROOT,
    canonical_hash,
    load_config,
    relative_path,
    resolve_path,
    sha256_file,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _rank(seed: int, complaint_id: int) -> bytes:
    return hashlib.blake2b(f"{seed}:{complaint_id}".encode(), digest_size=16).digest()


def _source_rows(path: Path, *, exclude: set[int], limit: int, seed: int) -> list[dict[str, Any]]:
    con = duckdb.connect()
    try:
        ids = [
            int(row[0])
            for row in con.execute(
                "SELECT complaint_id FROM read_parquet(?)", [str(path)]
            ).fetchall()
            if int(row[0]) not in exclude
        ]
        selected = sorted(ids, key=lambda value: (_rank(seed, value), value))[:limit]
        placeholders = ",".join("?" for _ in selected)
        rows = con.execute(
            f"""
            SELECT complaint_id, narrative, source_product, source_issue,
                   source_company, label_json
            FROM read_parquet(?)
            WHERE complaint_id IN ({placeholders})
            """,
            [str(path), *selected],
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
            "target": json.loads(row[5]),
        }
        for row in rows
    }
    if len(by_id) != len(selected):
        raise RuntimeError("R1b deterministic candidate selection failed to reload every row")
    return [by_id[complaint_id] for complaint_id in selected]


def _record(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    return {
        "complaint_id": int(row["complaint_id"]),
        "input_contract_version": 2,
        "model_input": build_model_input(row),
        "target": row["target"],
        "target_source": source,
    }


def _validate_existing(output: Path, manifest_path: Path, *, expected_rows: int) -> dict | None:
    if not output.exists() and not manifest_path.exists():
        return None
    if not output.is_file() or not manifest_path.is_file():
        raise RuntimeError("R1b output and manifest must either both exist or both be absent")
    manifest = json.loads(manifest_path.read_text())
    artifact = manifest["artifact"]
    if (
        manifest.get("status") != "complete"
        or int(artifact["rows"]) != expected_rows
        or sha256_file(output) != artifact["sha256"]
    ):
        raise RuntimeError("existing R1b corpus conflicts with its frozen preparation receipt")
    return manifest


def materialize(*, smoke: bool) -> dict[str, Any]:
    config = load_config("configs/r1b_sft_rule_20k.yaml")
    data = config["data"]
    output = resolve_path(data["smoke_path"] if smoke else data["path"])
    manifest_path = (
        output.parent / "manifest.json" if smoke else resolve_path(data["prepared_manifest"])
    )
    target_rows = 10 if smoke else int(data["rows"])
    existing = _validate_existing(output, manifest_path, expected_rows=target_rows)
    if existing is not None and existing.get("config_hash") != config["_config_hash"]:
        if not smoke:
            raise RuntimeError("full R1b corpus belongs to a different frozen config")
        existing = None
    if existing is not None:
        print(
            f"R1b already materialized: rows={target_rows} sha256={existing['artifact']['sha256']}"
        )
        return existing

    r1_path = resolve_path(
        "data/smoke/phase2/sft_rule.jsonl" if smoke else data["required_r1_path"]
    )
    if not r1_path.is_file():
        raise FileNotFoundError(f"matched R1 corpus is missing: {r1_path}")
    if not smoke and sha256_file(r1_path) != data["required_r1_sha256"]:
        raise ValueError("R1b base coverage differs from the frozen R1 corpus")
    base = _read_jsonl(r1_path)
    if len(base) > target_rows:
        base = base[:target_rows]
    base_ids = {int(row["complaint_id"]) for row in base}
    needed = target_rows - len(base)

    source_path = resolve_path(
        "data/smoke/splits/train.parquet" if smoke else data["source_train_path"]
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"frozen TRAIN source is missing: {source_path}")
    if not smoke and sha256_file(source_path) != data["source_train_payload_sha256"]:
        raise ValueError("R1b source TRAIN payload hash mismatch")
    candidate_limit = min(40, 50 - len(base_ids)) if smoke else int(data["candidate_rows"])
    source = _source_rows(
        source_path,
        exclude=base_ids,
        limit=candidate_limit,
        seed=int(data["selection_seed"]),
    )
    candidate_records = [_record(row, source="frozen_label_rules_v3_r1b") for row in source]
    test_paths = {
        split: resolve_path(
            f"data/smoke/splits/{split}.parquet"
            if smoke
            else config["evaluation"]["test_splits"][split]["path"]
        )
        for split in ("test_iid", "test_drift")
    }
    clean, quarantine, scanned = contamination_audit(
        candidate_records,
        test_paths=test_paths,
        token_ngram=int(data["contamination_token_ngram"]),
    )
    if len(clean) < needed:
        raise RuntimeError(
            f"R1b candidate cap left {len(clean)} clean additions; {needed} required. "
            "Stop for a human-approved config change rather than silently resampling."
        )
    additions = clean[:needed]
    base_normalized = [
        {
            "complaint_id": int(row["complaint_id"]),
            "input_contract_version": 2,
            "model_input": row["model_input"],
            "target": row["target"],
            "target_source": row.get("target_source", "frozen_label_rules_v3"),
        }
        for row in base
    ]
    records = base_normalized + additions
    if len(records) != target_rows or len({row["complaint_id"] for row in records}) != target_rows:
        raise RuntimeError("R1b output is not the requested unique coverage")
    write_jsonl_atomic(output, records)
    artifact_sha = sha256_file(output)
    audit = {
        "version": 1,
        "status": "complete",
        "mode": "smoke" if smoke else "full",
        "policy": "Phase 2 exact normalized token n-gram quarantine",
        "token_ngram": int(data["contamination_token_ngram"]),
        "candidate_rows": len(candidate_records),
        "candidate_clean_rows": len(clean),
        "candidate_quarantined_rows": len(quarantine),
        "selected_added_rows": len(additions),
        "inherited_r1_rows": len(base_normalized),
        "test_rows_scanned": scanned,
        "quarantine": quarantine,
    }
    audit_path = (
        output.parent / "r1b_contamination.json"
        if smoke
        else REPO_ROOT / "results" / "phase3_r1b_contamination.json"
    )
    write_json_atomic(audit_path, audit)
    dataset_hash = canonical_hash(
        {
            "artifact_sha256": artifact_sha,
            "source_train_sha256": (
                sha256_file(source_path) if smoke else data["source_train_payload_sha256"]
            ),
            "r1_sha256": sha256_file(r1_path),
            "contamination_audit_sha256": sha256_file(audit_path),
            "selection_seed": int(data["selection_seed"]),
        }
    )
    manifest = {
        "version": 1,
        "status": "complete",
        "phase": 3,
        "mode": "smoke" if smoke else "full",
        "rung": "r1b",
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "dataset_hash": dataset_hash,
        "artifact": {
            "path": relative_path(output),
            "rows": len(records),
            "sha256": artifact_sha,
        },
        "coverage": {
            "r1_is_exact_subset": all(
                records[index]["complaint_id"] == base_normalized[index]["complaint_id"]
                for index in range(len(base_normalized))
            ),
            "inherited_r1_rows": len(base_normalized),
            "added_rule_rows": len(additions),
        },
        "contamination": {
            "audit_path": relative_path(audit_path),
            "audit_sha256": sha256_file(audit_path),
            "quarantined_candidates": len(quarantine),
            "selected_added_rows_clean": True,
        },
        "source": {
            "train_path": relative_path(source_path),
            "train_sha256": sha256_file(source_path),
            "r1_path": relative_path(r1_path),
            "r1_sha256": sha256_file(r1_path),
        },
    }
    write_json_atomic(manifest_path, manifest)
    print(f"R1b materialized: rows={len(records)} sha256={artifact_sha}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    materialize(smoke=args.smoke)


if __name__ == "__main__":
    main()
