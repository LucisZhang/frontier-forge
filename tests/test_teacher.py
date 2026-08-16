from __future__ import annotations

import http.client
import json
from pathlib import Path

import duckdb
import pytest

from forge.data.labels import derive_label
from forge.teacher.filters import (
    contamination_audit,
    minhash_deduplicate,
    perturb_near_miss,
)
from forge.teacher.freeze import DEFAULT_CONFIG_PATH, PayloadMissing, verify_frozen_source
from forge.teacher.generate import _call_with_retry, _mock_record
from forge.verify.schema import is_schema_valid


def source_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "complaint_id": 42,
        "product": "Credit reporting",
        "issue": "Incorrect information on your report",
        "company": "Acme Financial",
        "narrative": (
            "The consumer disputes incorrect information on a credit report and asks "
            "the company to investigate the account with the supplied evidence."
        ),
    }
    row.update(overrides)
    return row


def teacher_record(complaint_id: int, narrative: str) -> dict[str, object]:
    row = source_row(complaint_id=complaint_id, narrative=narrative)
    return {
        "complaint_id": complaint_id,
        "model_input": {
            "complaint_id": complaint_id,
            "narrative": narrative,
            "source_product": row["product"],
            "source_issue": row["issue"],
            "source_company": row["company"],
        },
        "rule_label": derive_label(row),
    }


def _write_test_parquet(path: Path, rows: list[tuple[int, str]]) -> None:
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE test_rows (complaint_id BIGINT, narrative VARCHAR)")
        con.executemany("INSERT INTO test_rows VALUES (?, ?)", rows)
        quoted = str(path).replace("'", "''")
        con.execute(f"COPY test_rows TO '{quoted}' (FORMAT PARQUET)")
    finally:
        con.close()


def test_phase2_config_pins_the_declared_frozen_lineage() -> None:
    # Manifest-level checks (tracked in git) must always run and pass.
    frozen = verify_frozen_source(DEFAULT_CONFIG_PATH, check_payloads=False)

    assert frozen["dataset_hash"] == (
        "2f2498c95ea224e48cfc7ee9c705ecef00563c616b0ec14512800706cd8f2573"
    )
    assert frozen["label_rules_version"] == 3
    assert frozen["splits"]["train"]["rows"] == 300_000
    assert frozen["planned_max_api_usd"] == 19.2

    # Payload-level checks (gitignored data files) only run where the data
    # is present; skip when absent, but fail on a real hash mismatch.
    try:
        payload_frozen = verify_frozen_source(DEFAULT_CONFIG_PATH, check_payloads=True)
    except PayloadMissing:
        pytest.skip("data payloads not present in this environment")
    else:
        assert payload_frozen["splits"]["train"]["rows"] == 300_000


def test_mock_teacher_record_retains_required_provenance() -> None:
    row = source_row()
    row["rule_label"] = derive_label(row)
    record = _mock_record(
        row=row,
        sequence=0,
        fingerprint="frozen-run",
        model="mock/phase2-deterministic-v1",
        prompt_sha256="a" * 64,
    )

    assert record["status"] == "ok"
    assert record["teacher_model_id"] == "mock/phase2-deterministic-v1"
    assert record["prompt_sha256"] == "a" * 64
    assert record["raw_response"]["choices"]
    assert record["score"]["task_success"] is True


def test_teacher_retry_handles_truncated_http_response(monkeypatch) -> None:
    calls = 0

    def flaky_request(**_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise http.client.IncompleteRead(b"{}", 10)
        return {"id": "retry-succeeded"}

    monkeypatch.setattr("forge.teacher.generate._request", flaky_request)
    monkeypatch.setattr("forge.teacher.generate.time.sleep", lambda _seconds: None)

    response, error = _call_with_retry(max_retries=2)

    assert response == {"id": "retry-succeeded"}
    assert error is None
    assert calls == 2


def test_minhash_dedup_keeps_first_normalized_duplicate() -> None:
    first = teacher_record(
        1,
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.",
    )
    duplicate = teacher_record(
        2,
        "  ALPHA beta gamma delta epsilon zeta eta theta iota kappa lambda mu  ",
    )
    distinct = teacher_record(
        3,
        "A completely distinct complaint narrative about another payment problem.",
    )

    kept, removed = minhash_deduplicate(
        [first, duplicate, distinct],
        token_ngram=3,
        permutations=32,
        bands=8,
        similarity_threshold=0.8,
    )

    assert [record["complaint_id"] for record in kept] == [1, 3]
    assert removed == [
        {
            "complaint_id": 2,
            "duplicate_of": 1,
            "estimated_similarity": 1.0,
            "reason": "exact_normalized_text",
        }
    ]


def test_contamination_audit_quarantines_any_exact_long_ngram(tmp_path: Path) -> None:
    contaminated = teacher_record(
        10,
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen",
    )
    clean_record = teacher_record(
        11,
        "red orange yellow green blue indigo violet copper silver gold quartz cedar maple",
    )
    test_path = tmp_path / "test.parquet"
    _write_test_parquet(
        test_path,
        [
            (
                99,
                "prefix one two three four five six seven eight nine ten eleven twelve "
                "thirteen suffix",
            )
        ],
    )

    clean, quarantine, scanned = contamination_audit(
        [contaminated, clean_record],
        test_paths={"test_iid": test_path},
        token_ngram=13,
    )

    assert [record["complaint_id"] for record in clean] == [11]
    assert scanned == {"test_iid": 1}
    assert len(quarantine) == 1
    assert quarantine[0]["complaint_id"] == 10
    assert quarantine[0]["test_complaint_id"] == 99


def test_dpo_perturbation_is_schema_valid_and_strictly_lower_scoring() -> None:
    row = source_row()
    rule_label = derive_label(row)
    model_input = {
        "complaint_id": row["complaint_id"],
        "narrative": row["narrative"],
        "source_product": row["product"],
        "source_issue": row["issue"],
        "source_company": row["company"],
    }

    rejected, perturbation, chosen_score, rejected_score = perturb_near_miss(
        chosen=json.loads(json.dumps(rule_label)),
        model_input=model_input,
        rule_label=rule_label,
        taxonomy=("urgency_cycle", "tool_choice_substitution", "ambiguity_flip"),
    )

    assert perturbation in {
        "urgency_cycle",
        "tool_choice_substitution",
        "ambiguity_flip",
    }
    assert is_schema_valid(rejected)
    assert chosen_score.task_success is True
    assert rejected_score.reward < chosen_score.reward
