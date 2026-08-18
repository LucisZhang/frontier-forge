"""Materialize the D5.1 fresh, contamination-screened R4 v2 prompt pool."""

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
        if not selected:
            return []
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
        raise RuntimeError("R4 v2 candidate selection failed to reload every selected row")
    return [by_id[complaint_id] for complaint_id in selected]


def _record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "complaint_id": int(row["complaint_id"]),
        "input_contract_version": 2,
        "model_input": build_model_input(row),
        "target": row["target"],
        "target_source": "frozen_label_rules_v3_r4_v2_fresh_pool",
    }


def _validate_existing(
    output: Path,
    manifest_path: Path,
    *,
    expected_rows: int,
    config_hash: str,
) -> dict[str, Any] | None:
    if not output.exists() and not manifest_path.exists():
        return None
    if not output.is_file() or not manifest_path.is_file():
        raise RuntimeError("R4 v2 pool and manifest must either both exist or both be absent")
    manifest = json.loads(manifest_path.read_text())
    artifact = manifest.get("artifact", {})
    if (
        manifest.get("status") != "complete"
        or manifest.get("config_hash") != config_hash
        or int(artifact.get("rows", -1)) != expected_rows
        or artifact.get("path") != relative_path(output)
        or sha256_file(output) != artifact.get("sha256")
    ):
        raise RuntimeError("existing R4 v2 pool conflicts with its immutable manifest")
    return manifest


def materialize(*, smoke: bool) -> dict[str, Any]:
    config = load_config("configs/r4_grpo.yaml")
    if config.get("run_revision") != "phase3_2_fresh_pool":
        raise ValueError("fresh-pool builder requires the D5.1 R4 v2 config")
    data = config["data"]
    output = resolve_path(data["smoke_path"] if smoke else data["path"])
    manifest_path = (
        output.parent / "manifest.json" if smoke else resolve_path(data["prepared_manifest"])
    )
    target_rows = min(8, int(data["rows"])) if smoke else int(data["rows"])
    existing = _validate_existing(
        output,
        manifest_path,
        expected_rows=target_rows,
        config_hash=config["_config_hash"],
    )
    if existing is not None:
        print(
            "R4 v2 fresh pool already materialized: "
            f"rows={target_rows} sha256={existing['artifact']['sha256']}"
        )
        return existing

    previous_path = resolve_path(
        "data/smoke/phase3/r1b_sft_rule.jsonl" if smoke else data["required_previous_training_path"]
    )
    phase2_path = resolve_path(
        "data/smoke/phase2/sft_rule.jsonl" if smoke else data["required_phase2_path"]
    )
    for label, path in (("previous training", previous_path), ("Phase 2", phase2_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} corpus is missing: {path}")
    if not smoke:
        if sha256_file(previous_path) != data["required_previous_training_sha256"]:
            raise ValueError("R4 v2 previous-training corpus hash mismatch")
        if sha256_file(phase2_path) != data["required_phase2_sha256"]:
            raise ValueError("R4 v2 Phase 2 corpus hash mismatch")

    previous = _read_jsonl(previous_path)
    phase2 = _read_jsonl(phase2_path)
    previous_ids = {int(row["complaint_id"]) for row in previous}
    phase2_ids = {int(row["complaint_id"]) for row in phase2}
    expected_previous_rows = (
        len(previous) if smoke else int(data["required_previous_training_rows"])
    )
    expected_phase2_rows = len(phase2) if smoke else int(data["required_phase2_rows"])
    if len(previous) != expected_previous_rows or len(previous_ids) != expected_previous_rows:
        raise RuntimeError("R4 v2 previous-training exclusion set is not unique and complete")
    if len(phase2) != expected_phase2_rows or len(phase2_ids) != expected_phase2_rows:
        raise RuntimeError("R4 v2 Phase 2 exclusion set is not unique and complete")
    if not phase2_ids <= previous_ids:
        raise RuntimeError("R1b no longer contains every Phase 2 training complaint")

    source_path = resolve_path(
        "data/smoke/splits/train.parquet" if smoke else data["source_train_path"]
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"frozen TRAIN source is missing: {source_path}")
    if not smoke and sha256_file(source_path) != data["source_train_payload_sha256"]:
        raise ValueError("R4 v2 source TRAIN payload hash mismatch")
    candidate_limit = 50 if smoke else int(data["candidate_rows"])
    source = _source_rows(
        source_path,
        exclude=previous_ids,
        limit=candidate_limit,
        seed=int(data["selection_seed"]),
    )
    candidate_records = [_record(row) for row in source]
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
    if len(clean) < target_rows:
        raise RuntimeError(
            f"R4 v2 candidate cap left {len(clean)} clean prompts; {target_rows} required. "
            "Stop for a human-approved config change rather than silently resampling."
        )
    records = clean[:target_rows]
    fresh_ids = {int(row["complaint_id"]) for row in records}
    if len(records) != target_rows or len(fresh_ids) != target_rows:
        raise RuntimeError("R4 v2 output is not the requested unique prompt pool")
    if fresh_ids & previous_ids or fresh_ids & phase2_ids:
        raise RuntimeError("R4 v2 prompt pool overlaps a previously trained complaint")
    if not all(
        isinstance(row.get("target"), dict)
        and row.get("target_source") == "frozen_label_rules_v3_r4_v2_fresh_pool"
        for row in records
    ):
        raise RuntimeError("R4 v2 fresh pool is missing frozen rule labels")

    write_jsonl_atomic(output, records)
    artifact_sha = sha256_file(output)
    audit = {
        "version": 1,
        "status": "complete",
        "phase": 3.2,
        "mode": "smoke" if smoke else "full",
        "policy": "Phase 2 exact normalized token n-gram quarantine",
        "token_ngram": int(data["contamination_token_ngram"]),
        "candidate_rows": len(candidate_records),
        "candidate_clean_rows": len(clean),
        "candidate_quarantined_rows": len(quarantine),
        "selected_fresh_rows": len(records),
        "test_rows_scanned": scanned,
        "disjointness": {
            "previously_trained_unique_rows": len(previous_ids),
            "phase2_unique_rows": len(phase2_ids),
            "phase2_is_subset_of_previous_training": phase2_ids <= previous_ids,
            "fresh_unique_rows": len(fresh_ids),
            "fresh_overlap_previous_training": len(fresh_ids & previous_ids),
            "fresh_overlap_phase2": len(fresh_ids & phase2_ids),
        },
        "quarantine": quarantine,
    }
    audit_path = (
        output.parent / "contamination_audit.json"
        if smoke
        else REPO_ROOT / "results" / "phase3_2_r4_v2_contamination.json"
    )
    write_json_atomic(audit_path, audit)
    source_sha = sha256_file(source_path)
    previous_sha = sha256_file(previous_path)
    phase2_sha = sha256_file(phase2_path)
    audit_sha = sha256_file(audit_path)
    dataset_hash = canonical_hash(
        {
            "artifact_sha256": artifact_sha,
            "source_train_sha256": source_sha,
            "previous_training_sha256": previous_sha,
            "phase2_sha256": phase2_sha,
            "contamination_audit_sha256": audit_sha,
            "selection_seed": int(data["selection_seed"]),
        }
    )
    manifest = {
        "version": 1,
        "status": "complete",
        "phase": 3.2,
        "mode": "smoke" if smoke else "full",
        "rung": "r4",
        "run_revision": config["run_revision"],
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "dataset_hash": dataset_hash,
        "artifact": {
            "path": relative_path(output),
            "rows": len(records),
            "sha256": artifact_sha,
            "rule_labels_attached": True,
        },
        "selection": {
            "candidate_rows": len(candidate_records),
            "clean_candidate_rows": len(clean),
            "selected_rows": len(records),
            "selection_seed": int(data["selection_seed"]),
        },
        "disjointness": audit["disjointness"],
        "contamination": {
            "audit_path": relative_path(audit_path),
            "audit_sha256": audit_sha,
            "quarantined_candidates": len(quarantine),
            "selected_rows_clean": True,
            "test_rows_scanned": scanned,
        },
        "source": {
            "train_path": relative_path(source_path),
            "train_sha256": source_sha,
            "previous_training_path": relative_path(previous_path),
            "previous_training_sha256": previous_sha,
            "phase2_path": relative_path(phase2_path),
            "phase2_sha256": phase2_sha,
        },
    }
    write_json_atomic(manifest_path, manifest)
    print(f"R4 v2 fresh pool materialized: rows={len(records)} sha256={artifact_sha}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    materialize(smoke=args.smoke)


if __name__ == "__main__":
    main()
