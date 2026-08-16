"""Reproduce and verify the Phase 2 filter funnel from raw teacher receipts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import duckdb

from forge.data.ingest import sha256_file
from forge.teacher.filters import contamination_audit, minhash_deduplicate
from forge.teacher.freeze import (
    DEFAULT_CONFIG_PATH,
    load_teacher_config,
    resolve_path,
    verify_frozen_source,
)
from forge.teacher.generate import (
    DPO_NAME,
    FILTER_LOG_NAME,
    LEDGER_NAME,
    MANIFEST_NAME,
    RAW_LOG_NAME,
    SFT_DISTILLED_NAME,
    SFT_RULE_NAME,
    _successful_records,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} differs: expected {expected}, got {actual}")


def _verify_membership(
    complaint_ids: list[int], *, train_path: Path, test_paths: dict[str, Path]
) -> None:
    placeholders = ",".join("?" for _ in complaint_ids)
    con = duckdb.connect()
    try:
        train_count = con.execute(
            f"SELECT count(*) FROM read_parquet(?) WHERE complaint_id IN ({placeholders})",
            [str(train_path), *complaint_ids],
        ).fetchone()
        if train_count is None or int(train_count[0]) != len(complaint_ids):
            raise ValueError("teacher raw log contains an id outside frozen TRAIN")
        for split, path in test_paths.items():
            overlap = con.execute(
                f"SELECT count(*) FROM read_parquet(?) WHERE complaint_id IN ({placeholders})",
                [str(path), *complaint_ids],
            ).fetchone()
            if overlap is not None and int(overlap[0]):
                raise ValueError(f"teacher raw log contains {overlap[0]} {split} ids")
    finally:
        con.close()


def run_teacher_audit(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    smoke: bool = False,
) -> dict[str, Any]:
    """Recompute funnel membership and verify every materialized receipt."""

    config_path = Path(config_path)
    config = load_teacher_config(config_path)
    frozen = verify_frozen_source(config_path)
    output_dir = resolve_path(
        config["outputs"]["smoke_dir"] if smoke else config["outputs"]["full_dir"]
    )
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Phase 2 manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected_mode = "smoke" if smoke else "live"
    if manifest.get("status") != "complete" or manifest.get("mode") != expected_mode:
        raise ValueError("Phase 2 manifest is not the requested completed mode")
    if manifest.get("source", {}).get("dataset_hash") != frozen["dataset_hash"]:
        raise ValueError("Phase 2 manifest targets a different frozen dataset")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise ValueError("Phase 2 manifest config hash mismatch")

    raw_path = output_dir / RAW_LOG_NAME
    ledger_path = output_dir / LEDGER_NAME
    raw_records = _read_jsonl(raw_path)
    ledger = json.loads(ledger_path.read_text())
    if sha256_file(raw_path) != manifest["generation"]["raw_log_sha256"]:
        raise ValueError("raw teacher log hash mismatch")
    if sha256_file(raw_path) != ledger["raw_log_sha256"]:
        raise ValueError("generation ledger raw-log hash mismatch")
    if any(
        "teacher_model_id" not in record
        or "prompt_sha256" not in record
        or (record.get("status") == "ok" and "raw_response" not in record)
        for record in raw_records
    ):
        raise ValueError("teacher receipt is missing model, prompt, or raw response provenance")

    successful = _successful_records(raw_records)
    latest = sorted(successful.values(), key=lambda record: int(record["sequence"]))
    complaint_ids = [int(record["complaint_id"]) for record in latest]
    if len(complaint_ids) != manifest["generation"]["selected_rows"]:
        raise ValueError("successful teacher response count differs from selected rows")
    if len(complaint_ids) != len(set(complaint_ids)):
        raise ValueError("duplicate successful complaint id in teacher log")
    train_path = resolve_path(config["source"]["train"]["path"])
    test_paths = {
        name: resolve_path(item["path"]) for name, item in config["source"]["test_splits"].items()
    }
    _verify_membership(complaint_ids, train_path=train_path, test_paths=test_paths)

    schema_valid = [record for record in latest if record["score"]["schema_valid"]]
    verifier_valid = [
        record
        for record in schema_valid
        if record["score"]["task_success"]
        and record["score"]["secondary_metrics"]["tool_arguments_semantic_valid"]
    ]
    minhash = config["filter"]["minhash"]
    deduplicated, duplicates = minhash_deduplicate(
        verifier_valid,
        token_ngram=int(minhash["token_ngram"]),
        permutations=int(minhash["permutations"]),
        bands=int(minhash["bands"]),
        similarity_threshold=float(minhash["similarity_threshold"]),
    )
    clean, quarantine, scanned = contamination_audit(
        deduplicated,
        test_paths=test_paths,
        token_ngram=int(config["filter"]["contamination"]["token_ngram"]),
    )
    reproduced_counts = [
        len(latest),
        len(latest),
        len(schema_valid),
        len(verifier_valid),
        len(deduplicated),
        len(clean),
    ]
    declared_counts = [int(stage["count"]) for stage in manifest["filter_funnel"]]
    if reproduced_counts != declared_counts:
        raise ValueError(
            f"filter funnel mismatch: declared={declared_counts}, reproduced={reproduced_counts}"
        )
    if len(duplicates) != manifest["minhash"]["duplicates_removed"]:
        raise ValueError("MinHash duplicate count mismatch")
    if scanned != manifest["contamination"]["test_rows_scanned"]:
        raise ValueError("contamination TEST scan counts mismatch")
    if len(quarantine) != manifest["contamination"]["quarantined_rows"]:
        raise ValueError("contamination quarantine count mismatch")

    outputs = config["outputs"]
    quarantine_path = (
        output_dir / "phase2_contamination_quarantine.json"
        if smoke
        else resolve_path(outputs["quarantine"])
    )
    quarantine_receipt = json.loads(quarantine_path.read_text())
    if quarantine_receipt["rows"] != quarantine:
        raise ValueError("committed quarantine receipt does not reproduce from raw logs")
    if sha256_file(quarantine_path) != manifest["contamination"]["quarantine_sha256"]:
        raise ValueError("quarantine receipt hash mismatch")

    artifact_paths = {
        "raw_teacher_generations": raw_path,
        "filter_funnel_log": output_dir / FILTER_LOG_NAME,
        "sft_rule": output_dir / SFT_RULE_NAME,
        "sft_distilled": output_dir / SFT_DISTILLED_NAME,
        "dpo_pairs": output_dir / DPO_NAME,
    }
    for name, path in artifact_paths.items():
        declared = manifest["artifacts"][name]
        if sha256_file(path) != declared["sha256"]:
            raise ValueError(f"{name} SHA-256 mismatch")
        if len(_read_jsonl(path)) != declared["rows"]:
            raise ValueError(f"{name} row count mismatch")

    filter_log = _read_jsonl(output_dir / FILTER_LOG_NAME)
    if len(filter_log) != len(latest):
        raise ValueError("filter decision log does not cover every selected response")
    rule = _read_jsonl(output_dir / SFT_RULE_NAME)
    distilled = _read_jsonl(output_dir / SFT_DISTILLED_NAME)
    dpo = _read_jsonl(output_dir / DPO_NAME)
    rule_ids = [int(record["complaint_id"]) for record in rule]
    distilled_ids = [int(record["complaint_id"]) for record in distilled]
    dpo_ids = [int(record["complaint_id"]) for record in dpo]
    clean_ids = [int(record["complaint_id"]) for record in clean]
    if not (rule_ids == distilled_ids == dpo_ids == clean_ids):
        raise ValueError("rule, distilled, DPO, and reproduced clean coverage differ")
    if any(
        float(record["chosen_score"]["reward"]) <= float(record["rejected_score"]["reward"])
        for record in dpo
    ):
        raise ValueError("DPO pair does not strictly prefer the chosen verifier score")
    if any(
        "teacher_model_id" not in record
        or "prompt_sha256" not in record
        or "raw_response" not in record
        for record in distilled
    ):
        raise ValueError("distilled corpus lost teacher provenance")

    unique_cost = sum(
        float(record.get("reported_cost_usd") or 0.0) for record in successful.values()
    )
    declared_unique_cost = float(
        manifest["cost"].get("provider_receipted_unique_api_usd", manifest["cost"]["api_usd"])
    )
    _assert_close(unique_cost, declared_unique_cost, "manifest unique-response API cost")
    _assert_close(unique_cost, float(ledger["reported_api_usd"]), "generation-ledger API cost")

    account_cost = float(manifest["cost"]["api_usd"])
    incident_receipt = manifest.get("receipts", {}).get("transport_incident_path")
    if incident_receipt is None:
        _assert_close(account_cost, unique_cost, "manifest account API cost")
    else:
        incident_path = resolve_path(incident_receipt)
        if sha256_file(incident_path) != manifest["receipts"]["transport_incident_sha256"]:
            raise ValueError("transport-incident receipt hash mismatch")
        incident = json.loads(incident_path.read_text())
        if incident.get("status") != "reconciled":
            raise ValueError("transport incident is not account-reconciled")
        raw_by_sequence = {int(record["sequence"]): record for record in latest}
        replay_cost = 0.0
        for replay in incident["replay_receipts"]:
            raw = raw_by_sequence.get(int(replay["sequence"]))
            if raw is None:
                raise ValueError("transport replay sequence is absent from the raw log")
            if int(raw["complaint_id"]) != int(replay["complaint_id"]) or raw.get(
                "response_id"
            ) != replay.get("response_id"):
                raise ValueError("transport replay receipt differs from the raw log")
            _assert_close(
                float(raw["reported_cost_usd"]),
                float(replay["reported_cost_usd"]),
                "transport replay response cost",
            )
            replay_cost += float(replay["reported_cost_usd"])
        _assert_close(
            replay_cost,
            float(incident["cost"]["replayed_batch_charge_usd"]),
            "transport replay batch cost",
        )
        _assert_close(
            unique_cost + replay_cost,
            account_cost,
            "account-reconciled Phase 2 API cost",
        )
        snapshot = incident["account_reconciliation"]["current_key_usage_snapshot"]
        prior = incident["account_reconciliation"]["pre_phase2_receipt"]
        prior_path = resolve_path(prior["path"])
        prior_receipt = json.loads(prior_path.read_text())
        _assert_close(
            float(prior_receipt["reported_api_usd"]),
            float(prior["api_usd"]),
            "pre-Phase-2 current-key receipt cost",
        )
        for window in ("usage_daily_usd", "usage_weekly_usd", "usage_monthly_usd"):
            _assert_close(
                float(snapshot[window]),
                float(snapshot["usage_total_usd"]),
                f"current-key {window}",
            )
        _assert_close(
            float(snapshot["usage_total_usd"]) - float(prior["api_usd"]),
            account_cost,
            "current-key Phase 2 usage delta",
        )
        _assert_close(
            account_cost,
            float(incident["cost"]["account_reconciled_phase2_api_usd"]),
            "transport-incident Phase 2 API cost",
        )

    if account_cost > float(manifest["cost"]["run_cap_usd"]):
        raise ValueError("reproduced teacher API cost exceeds run cap")

    outputs = config["outputs"]
    data_card_path = (
        output_dir / "phase2_data_card.md" if smoke else resolve_path(outputs["data_card"])
    )
    cost_ledger_path = (
        output_dir / "phase2_cost_ledger.json" if smoke else resolve_path(outputs["cost_ledger"])
    )
    if sha256_file(data_card_path) != manifest["receipts"]["data_card_sha256"]:
        raise ValueError("data-card hash mismatch")
    if sha256_file(cost_ledger_path) != manifest["receipts"]["cost_ledger_sha256"]:
        raise ValueError("cost-ledger hash mismatch")
    cost_ledger = json.loads(cost_ledger_path.read_text())
    ledger_account_cost = float(
        cost_ledger.get(
            "account_reconciled_api_usd", cost_ledger.get("provider_reported_api_usd", 0.0)
        )
    )
    _assert_close(account_cost, ledger_account_cost, "tracked cost-ledger API cost")
    if cost_ledger.get("within_run_cap") is not True:
        raise ValueError("tracked cost ledger does not declare the run within cap")

    return {
        "status": "pass",
        "mode": expected_mode,
        "network_calls": 0,
        "selected_rows": len(latest),
        "schema_valid_rows": len(schema_valid),
        "verifier_accepted_rows": len(verifier_valid),
        "minhash_unique_rows": len(deduplicated),
        "contamination_clean_rows": len(clean),
        "quarantined_rows": len(quarantine),
        "corpus_rows": len(rule),
        "dpo_pairs": len(dpo),
        "api_usd": account_cost,
        "unique_response_api_usd": unique_cost,
        "phase2_dataset_hash": manifest["phase2_dataset_hash"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forge.teacher.audit")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    result = run_teacher_audit(config_path=args.config, smoke=args.smoke)
    print(
        f"teacher audit: {result['status']}; mode={result['mode']}; "
        f"selected={result['selected_rows']}; corpus_rows={result['corpus_rows']}; "
        f"quarantined={result['quarantined_rows']}; api_usd={result['api_usd']:.6f}; "
        "network_calls=0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
