from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from forge.data.api_calibrate import _metrics as api_metrics
from forge.data.calibrate import evaluate_stand_in, stand_in_prediction
from forge.data.ingest import run_ingest, sha256_file
from forge.data.input_contract import INPUT_CONTRACT_VERSION, build_model_input
from forge.data.labels import TOOL_PRECEDENCE, derive_label, load_rules
from forge.data.phase1_2 import (
    ChangedRow,
    _stratified_strong_action_sample,
    run_phase1_2_labels,
)
from forge.data.relabel import run_relabel
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

    assert rules.version == 3
    assert set(rules.product_map.values()) == set(PRODUCTS)
    assert rules.tool_priority == TOOL_PRECEDENCE


def test_tool_priority_config_must_match_code_precedence(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs" / "label_rules.yaml").read_text())
    raw["tools"]["priority"][0:2] = reversed(raw["tools"]["priority"][0:2])
    path = tmp_path / "bad-priority.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    with pytest.raises(ValueError, match=r"tools\.priority must exactly match"):
        load_rules(path)


def test_high_urgency_routes_to_regulator() -> None:
    label = derive_label(
        source_row(narrative="Foreclosure is scheduled tomorrow and needs review.")
    )

    assert label["urgency"] == "high"
    assert label["tool_call"]["name"] == "escalate_to_regulator"
    assert is_schema_valid(label)


@pytest.mark.parametrize(
    "source_issue",
    (
        "Credit monitoring or identity theft protection services",
        "Identity theft protection or other monitoring services",
        "Loan modification,collection,foreclosure",
    ),
)
def test_source_issue_taxonomy_cannot_trigger_escalation(source_issue: str) -> None:
    label = derive_label(
        source_row(
            issue=source_issue,
            narrative=(
                "The consumer describes a routine service problem, provides account history, "
                "and asks the company for a written response."
            ),
        )
    )

    assert label["urgency"] != "high"
    assert label["tool_call"]["name"] == "route_to_company"


def test_source_issue_refund_word_cannot_trigger_refund_workflow() -> None:
    label = derive_label(
        source_row(
            issue="Lost or stolen refund",
            narrative=(
                "The consumer describes a routine service problem, provides account history, "
                "and asks the company for a written response."
            ),
        )
    )

    assert label["tool_call"]["name"] == "route_to_company"


def test_short_narrative_abstains_and_requests_details() -> None:
    label = derive_label(source_row(narrative="Not sure."))

    assert label["ambiguity_flag"] is True
    assert label["tool_call"]["name"] == "request_more_info"
    assert "details" in label["tool_call"]["arguments"]["missing_fields"]


def test_long_narrative_not_sure_phrase_does_not_auto_flag_ambiguity() -> None:
    narrative = (
        "I am not sure why the bank used that explanation, but the complaint provides "
        "the account history, disputed transaction dates, prior contacts, requested "
        "resolution, and supporting evidence in enough detail for the company to act. "
        "The consumer requests a complete investigation and written response."
    )

    label = derive_label(source_row(narrative=narrative))

    assert len(narrative) > 200
    assert label["ambiguity_flag"] is False
    assert label["tool_call"]["name"] == "route_to_company"


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
    prediction = stand_in_prediction(
        source_row(
            product="Mortgage",
            issue="Foreclosure",
            narrative="A mortgage foreclosure complaint involving escrow.",
        )
    )

    assert prediction["product"] == "mortgage"
    assert is_schema_valid(prediction)


def test_input_contract_v2_includes_narrative_and_source_metadata() -> None:
    visible = build_model_input(source_row())

    assert INPUT_CONTRACT_VERSION == 2
    assert visible == {
        "complaint_id": 42,
        "narrative": source_row()["narrative"],
        "source_product": "Credit reporting",
        "source_issue": "Incorrect information on your report",
        "source_company": "Acme Financial",
    }


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
                "decision_checks": {
                    "urgency": True,
                    "ambiguity_flag": True,
                    "tool_choice": False,
                    "tool_arguments_structural": True,
                },
                "secondary_metrics": {
                    "product_match": True,
                    "issue_normalized_match": False,
                    "company_normalized_match": False,
                    "tool_arguments_semantic_valid": True,
                    "abstention_correct": True,
                },
                "task_success": False,
                "reward": 0.6,
            }
        },
        {
            "score": {
                "schema_valid": False,
                "decision_checks": {
                    "urgency": False,
                    "ambiguity_flag": False,
                    "tool_choice": False,
                    "tool_arguments_structural": False,
                },
                "secondary_metrics": {
                    "product_match": False,
                    "issue_normalized_match": False,
                    "company_normalized_match": False,
                    "tool_arguments_semantic_valid": False,
                    "abstention_correct": False,
                },
                "task_success": False,
                "reward": 0.0,
            }
        },
    ]

    metrics = api_metrics(records)

    assert metrics["samples"] == 2
    assert metrics["schema_valid"] == 0.5
    assert metrics["secondary_metrics"]["product_match"] == 0.5
    assert metrics["task_success"] == 0.0


def test_calibration_metrics_are_bounded() -> None:
    label = derive_label(source_row())
    metrics = evaluate_stand_in([{**source_row(), "label": label}])

    for key, value in metrics.items():
        if key == "secondary_metrics":
            assert all(0.0 <= float(item) <= 1.0 for item in value.values())
        elif key not in {
            "scorer_version",
            "samples",
            "task_success_count",
            "task_success_ci95_wilson",
        }:
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


def test_relabel_versions_labels_without_changing_membership(tmp_path: Path) -> None:
    ingest_path = tmp_path / "ingest" / "labeled_rows.parquet"
    ingest_manifest = tmp_path / "ingest" / "manifest.json"
    run_ingest(output_path=ingest_path, manifest_path=ingest_manifest, smoke=True)
    phase1_dir = tmp_path / "phase1"
    phase1_manifest_path = phase1_dir / "manifest.json"
    run_splits(
        ingest_path=ingest_path,
        ingest_manifest_path=ingest_manifest,
        output_dir=phase1_dir,
        manifest_path=phase1_manifest_path,
        audit_path=tmp_path / "phase1_audit.md",
        smoke=True,
    )
    phase1_manifest = json.loads(phase1_manifest_path.read_text())
    phase1_manifest["frozen"] = True
    phase1_manifest_path.write_text(json.dumps(phase1_manifest))

    output_dir = tmp_path / "phase1_1"
    manifest_path = output_dir / "manifest.json"
    audit_path = tmp_path / "phase1_1_audit.md"
    v2_rules = yaml.safe_load((ROOT / "configs" / "label_rules.yaml").read_text())
    v2_rules["version"] = 2
    v2_rules_path = tmp_path / "label_rules_v2.yaml"
    v2_rules_path.write_text(yaml.safe_dump(v2_rules, sort_keys=False))
    first, first_noop = run_relabel(
        phase1_split_dir=phase1_dir,
        phase1_manifest_path=phase1_manifest_path,
        rules_path=v2_rules_path,
        output_dir=output_dir,
        manifest_path=manifest_path,
        audit_path=audit_path,
    )
    second, second_noop = run_relabel(
        phase1_split_dir=phase1_dir,
        phase1_manifest_path=phase1_manifest_path,
        rules_path=v2_rules_path,
        output_dir=output_dir,
        manifest_path=manifest_path,
        audit_path=audit_path,
    )

    assert not first_noop
    assert second_noop
    assert first == second
    assert first["version"] == 2
    assert all(item["membership_matches_phase1"] for item in first["splits"].values())

    phase1_2_dir = tmp_path / "phase1_2" / "splits"
    phase1_2_manifest_path = tmp_path / "phase1_2" / "manifest.json"
    phase1_2_audit_path = tmp_path / "phase1_2_audit.md"
    third, third_noop = run_phase1_2_labels(
        phase1_split_dir=phase1_dir,
        phase1_manifest_path=phase1_manifest_path,
        phase1_1_split_dir=output_dir,
        phase1_1_manifest_path=manifest_path,
        output_dir=phase1_2_dir,
        manifest_path=phase1_2_manifest_path,
        audit_path=phase1_2_audit_path,
    )
    fourth, fourth_noop = run_phase1_2_labels(
        phase1_split_dir=phase1_dir,
        phase1_manifest_path=phase1_manifest_path,
        phase1_1_split_dir=output_dir,
        phase1_1_manifest_path=manifest_path,
        output_dir=phase1_2_dir,
        manifest_path=phase1_2_manifest_path,
        audit_path=phase1_2_audit_path,
    )

    assert not third_noop
    assert fourth_noop
    assert third == fourth
    assert third["version"] == 3
    assert third["dataset_hash"] != first["dataset_hash"]
    assert all(item["membership_matches_phase1"] for item in third["splits"].values())


def test_strong_action_audit_stratifies_rare_transitions() -> None:
    rows: list[ChangedRow] = []
    for complaint_id in range(100, 180):
        rows.append(
            ChangedRow(
                split=SPLITS[complaint_id % len(SPLITS)],
                complaint_id=complaint_id,
                date_received="2022-01-01",
                source_product="Credit reporting",
                source_issue=f"issue-{complaint_id % 3}",
                source_company="Acme",
                narrative="Routine complaint narrative with enough detail for routing.",
                old_label={
                    "urgency": "high",
                    "tool_call": {"name": "escalate_to_regulator"},
                },
                new_label={
                    "urgency": "low",
                    "tool_call": {"name": "route_to_company"},
                },
            )
        )
    for complaint_id in (900, 901):
        rows.append(
            ChangedRow(
                split="cal",
                complaint_id=complaint_id,
                date_received="2022-01-01",
                source_product="Money transfers",
                source_issue="Lost or stolen refund",
                source_company="Acme",
                narrative="Routine complaint narrative with enough detail for routing.",
                old_label={
                    "urgency": "low",
                    "tool_call": {"name": "start_refund_workflow"},
                },
                new_label={
                    "urgency": "low",
                    "tool_call": {"name": "route_to_company"},
                },
            )
        )

    population, selected = _stratified_strong_action_sample(rows, cap=10)

    assert len(population) == 82
    assert len(selected) == 10
    assert {row.complaint_id for row in selected} >= {900, 901}
    assert {row.transition for row in selected} == {
        "escalate_to_regulator -> route_to_company",
        "start_refund_workflow -> route_to_company",
    }


def test_phase1_2_runner_has_no_network_dependencies() -> None:
    source = (ROOT / "src" / "forge" / "data" / "phase1_2.py").read_text()

    assert "import socket" not in source
    assert "import urllib" not in source
    assert "import requests" not in source
    assert "import httpx" not in source


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
