from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.data.api_calibrate import _metrics as api_metrics
from forge.data.calibrate import evaluate_stand_in, stand_in_prediction
from forge.data.ingest import run_ingest, sha256_file
from forge.data.labels import derive_label, load_rules
from forge.data.splits import SPLITS, run_splits
from forge.data.spot_label import _parse_teacher_content, _reported_cost
from forge.verify.schema import PRODUCTS, TASK_SCHEMA, TOOL_NAMES, export_schema, is_schema_valid

ROOT = Path(__file__).resolve().parents[1]


def source_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "complaint_id": 42,
        "product": "Credit reporting",
        "issue": "Incorrect information on your report",
        "company": "Acme Financial",
        "narrative": (
            "The consumer disputes incorrect information shown on a credit report and "
            "asks the company to investigate it."
        ),
    }
    row.update(overrides)
    return row


def test_exported_task_schema_is_generated_from_python_source(tmp_path: Path) -> None:
    generated = tmp_path / "task_schema.json"
    export_schema(generated)

    assert json.loads(generated.read_text()) == TASK_SCHEMA
    assert json.loads((ROOT / "configs" / "task_schema.json").read_text()) == TASK_SCHEMA


def test_schema_has_exact_locked_product_and_tool_enums() -> None:
    assert tuple(TASK_SCHEMA["properties"]["product"]["enum"]) == PRODUCTS
    variants = TASK_SCHEMA["properties"]["tool_call"]["oneOf"]
    assert tuple(item["properties"]["name"]["const"] for item in variants) == TOOL_NAMES
    assert len(PRODUCTS) == 9
    assert len(TOOL_NAMES) == 5


def test_label_rules_cover_every_schema_product() -> None:
    rules = load_rules()

    assert set(rules.product_map.values()) == set(PRODUCTS)


def test_high_urgency_routes_to_regulator() -> None:
    label = derive_label(
        source_row(narrative="Foreclosure is scheduled tomorrow and needs review.")
    )

    assert label["urgency"] == "high"
    assert label["tool_call"]["name"] == "escalate_to_regulator"
    assert is_schema_valid(label)


def test_short_narrative_abstains_and_requests_details() -> None:
    label = derive_label(source_row(narrative="Not sure."))

    assert label["ambiguity_flag"] is True
    assert label["tool_call"]["name"] == "request_more_info"
    assert "details" in label["tool_call"]["arguments"]["missing_fields"]


def test_missing_company_abstains() -> None:
    label = derive_label(source_row(company=None))

    assert label["company"] is None
    assert label["ambiguity_flag"] is True
    assert label["tool_call"]["arguments"]["missing_fields"] == ["company"]


def test_refund_language_starts_refund_workflow() -> None:
    label = derive_label(
        source_row(
            issue="Problem with a card purchase",
            narrative="The consumer asks for reimbursement and supplied evidence for the purchase.",
        )
    )

    assert label["tool_call"]["name"] == "start_refund_workflow"
    assert label["tool_call"]["arguments"]["evidence_required"] is True


def test_resolved_language_closes_without_action_before_other_rules() -> None:
    label = derive_label(
        source_row(
            narrative=(
                "The issue was resolved and the consumer confirms no further review is needed."
            )
        )
    )

    assert label["tool_call"] == {
        "name": "close_no_action",
        "arguments": {"reason": "already_resolved"},
    }


def test_default_rule_routes_to_company() -> None:
    label = derive_label(source_row())

    assert label["tool_call"]["name"] == "route_to_company"


def test_unknown_product_fails_closed() -> None:
    with pytest.raises(ValueError, match="unmapped CFPB product"):
        derive_label(source_row(product="Imaginary financial product"))


def test_stand_in_always_emits_schema_valid_output() -> None:
    prediction = stand_in_prediction("A mortgage foreclosure complaint involving escrow.")

    assert prediction["product"] == "mortgage"
    assert is_schema_valid(prediction)


def test_teacher_audit_parser_preserves_provider_fence_as_raw_but_extracts_json() -> None:
    assert _parse_teacher_content('```json\n{"urgency":"low"}\n```') == {"urgency": "low"}


def test_teacher_cost_receipt_requires_reported_usage_cost() -> None:
    assert _reported_cost({"usage": {"cost": "0.0123"}}) == pytest.approx(0.0123)
    assert _reported_cost({"usage": {"total_tokens": 100}}) is None


def test_api_calibration_metrics_keep_schema_failures_in_denominator() -> None:
    records = [
        {
            "score": {
                "schema_valid": True,
                "field_matches": {
                    "product": True,
                    "issue": False,
                    "company": False,
                    "urgency": True,
                    "ambiguity_flag": True,
                },
                "tool_choice_match": False,
                "tool_arguments_match": False,
                "abstention_correct": True,
                "task_success": False,
                "reward": 0.6,
            }
        },
        {
            "score": {
                "schema_valid": False,
                "field_matches": {
                    "product": False,
                    "issue": False,
                    "company": False,
                    "urgency": False,
                    "ambiguity_flag": False,
                },
                "tool_choice_match": False,
                "tool_arguments_match": False,
                "abstention_correct": False,
                "task_success": False,
                "reward": 0.0,
            }
        },
    ]

    metrics = api_metrics(records)

    assert metrics["samples"] == 2
    assert metrics["schema_valid"] == 0.5
    assert metrics["product_match"] == 0.5
    assert metrics["task_success"] == 0.0


def test_calibration_metrics_are_bounded() -> None:
    label = derive_label(source_row())
    metrics = evaluate_stand_in(
        [{"complaint_id": 42, "narrative": source_row()["narrative"], "label": label}]
    )

    for key, value in metrics.items():
        if key not in {"samples", "task_success_count", "task_success_ci95_wilson"}:
            assert 0.0 <= float(value) <= 1.0


def test_smoke_ingest_and_splits_are_idempotent(tmp_path: Path) -> None:
    ingest_path = tmp_path / "ingest" / "labeled_rows.parquet"
    ingest_manifest = tmp_path / "ingest" / "manifest.json"
    first_ingest, first_ingest_noop = run_ingest(
        output_path=ingest_path,
        manifest_path=ingest_manifest,
        smoke=True,
    )
    second_ingest, second_ingest_noop = run_ingest(
        output_path=ingest_path,
        manifest_path=ingest_manifest,
        smoke=True,
    )

    assert not first_ingest_noop
    assert second_ingest_noop
    assert first_ingest == second_ingest
    assert first_ingest["artifact"]["rows"] == 50
    assert first_ingest["artifact"]["sha256"] == sha256_file(ingest_path)

    output_dir = tmp_path / "splits"
    splits_manifest = output_dir / "manifest.json"
    audit_path = tmp_path / "audit.md"
    first_splits, first_splits_noop = run_splits(
        ingest_path=ingest_path,
        ingest_manifest_path=ingest_manifest,
        output_dir=output_dir,
        manifest_path=splits_manifest,
        audit_path=audit_path,
        smoke=True,
    )
    second_splits, second_splits_noop = run_splits(
        ingest_path=ingest_path,
        ingest_manifest_path=ingest_manifest,
        output_dir=output_dir,
        manifest_path=splits_manifest,
        audit_path=audit_path,
        smoke=True,
    )

    assert not first_splits_noop
    assert second_splits_noop
    assert first_splits == second_splits
    assert tuple(first_splits["splits"]) == SPLITS
    assert sum(item["rows"] for item in first_splits["splits"].values()) == 50
    assert first_splits["audit"]["rows"] == 50
    assert first_splits["protocol"]["cross_split_complaint_id_overlap"] == 0


def test_non_smoke_split_rerun_refuses_changed_frozen_audit(tmp_path: Path) -> None:
    ingest_path = tmp_path / "ingest.parquet"
    ingest_manifest = tmp_path / "ingest.json"
    run_ingest(output_path=ingest_path, manifest_path=ingest_manifest, smoke=True)
    output_dir = tmp_path / "splits"
    manifest_path = output_dir / "manifest.json"
    audit_path = tmp_path / "audit.md"
    run_splits(
        ingest_path=ingest_path,
        ingest_manifest_path=ingest_manifest,
        output_dir=output_dir,
        manifest_path=manifest_path,
        audit_path=audit_path,
        smoke=True,
    )
    audit_path.write_text("tampered")

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        run_splits(
            ingest_path=ingest_path,
            ingest_manifest_path=ingest_manifest,
            output_dir=output_dir,
            manifest_path=manifest_path,
            audit_path=audit_path,
            smoke=False,
        )
