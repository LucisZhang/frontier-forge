from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from forge.verify.schema import is_schema_valid
from forge.verify.verifier import SCORER_VERSION, score


def gold_label() -> dict[str, Any]:
    return {
        "product": "credit_reporting",
        "issue": "Incorrect information on your report",
        "company": "Acme Financial Services",
        "urgency": "medium",
        "ambiguity_flag": False,
        "tool_call": {
            "name": "route_to_company",
            "arguments": {
                "company": "Acme Financial Services",
                "issue": "Incorrect information on your report",
            },
        },
    }


def _sample() -> dict[str, Any]:
    return {"complaint_id": 123, "label": gold_label()}


def _delete(path: tuple[str, ...]) -> Callable[[dict[str, Any]], None]:
    def mutate(value: dict[str, Any]) -> None:
        parent: dict[str, Any] = value
        for key in path[:-1]:
            parent = parent[key]
        del parent[path[-1]]

    return mutate


def _set(path: tuple[str, ...], replacement: object) -> Callable[[dict[str, Any]], None]:
    def mutate(value: dict[str, Any]) -> None:
        parent: dict[str, Any] = value
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = replacement

    return mutate


MALFORMED_OUTPUTS = (
    "",
    "{",
    '{"product":',
    "not-json",
    "```json\n{}\n```",
    b"\xff\xfe",
    '"just a string"',
    "[]",
    "null",
    "123",
)


@pytest.mark.parametrize("output", MALFORMED_OUTPUTS)
def test_malformed_or_non_object_output_scores_zero(output: object) -> None:
    result = score(_sample(), output)

    assert result.scorer_version == 2 == SCORER_VERSION
    assert not result.schema_valid
    assert not result.task_success
    assert result.reward == 0.0
    assert result.errors


SCHEMA_MUTATIONS: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("missing_product", _delete(("product",))),
    ("missing_issue", _delete(("issue",))),
    ("missing_company", _delete(("company",))),
    ("missing_urgency", _delete(("urgency",))),
    ("missing_ambiguity", _delete(("ambiguity_flag",))),
    ("missing_tool_call", _delete(("tool_call",))),
    ("empty_issue", _set(("issue",), "")),
    ("empty_company", _set(("company",), "")),
    ("hallucinated_product", _set(("product",), "crypto_wallet")),
    ("product_wrong_case", _set(("product",), "Credit_Reporting")),
    ("hallucinated_urgency", _set(("urgency",), "critical")),
    ("urgency_wrong_case", _set(("urgency",), "MEDIUM")),
    ("ambiguity_as_string", _set(("ambiguity_flag",), "false")),
    ("ambiguity_as_integer", _set(("ambiguity_flag",), 0)),
    ("tool_not_object", _set(("tool_call",), "route_to_company")),
    ("missing_tool_name", _delete(("tool_call", "name"))),
    ("missing_tool_args", _delete(("tool_call", "arguments"))),
    ("unknown_tool", _set(("tool_call", "name"), "send_email")),
    ("missing_route_company", _delete(("tool_call", "arguments", "company"))),
    ("missing_route_issue", _delete(("tool_call", "arguments", "issue"))),
    ("empty_route_company", _set(("tool_call", "arguments", "company"), "")),
    ("empty_route_issue", _set(("tool_call", "arguments", "issue"), "")),
)


def _extra_top(value: dict[str, Any]) -> None:
    value["confidence"] = 0.99


def _extra_tool(value: dict[str, Any]) -> None:
    value["tool_call"]["id"] = "call-1"


def _extra_argument(value: dict[str, Any]) -> None:
    value["tool_call"]["arguments"]["priority"] = "high"


SCHEMA_MUTATIONS += (
    ("extra_top_level", _extra_top),
    ("extra_tool_field", _extra_tool),
    ("extra_argument", _extra_argument),
)


@pytest.mark.parametrize(("case_name", "mutate"), SCHEMA_MUTATIONS, ids=lambda value: value)
def test_schema_adversaries_fail_closed(
    case_name: str, mutate: Callable[[dict[str, Any]], None]
) -> None:
    del case_name
    prediction = gold_label()
    mutate(prediction)

    result = score(_sample(), prediction)

    assert result.json_valid
    assert not result.schema_valid
    assert not result.task_success
    assert result.errors


WRONG_PRODUCTS = (
    "card",
    "debt_collection",
    "deposit_account",
    "money_service",
    "mortgage",
    "payday_personal_loan",
    "student_loan",
    "vehicle_loan",
)


@pytest.mark.parametrize("product", WRONG_PRODUCTS)
def test_valid_but_wrong_product_is_detected(product: str) -> None:
    prediction = gold_label()
    prediction["product"] = product

    result = score(_sample(), prediction)

    assert result.schema_valid
    assert not result.product_match
    assert result.task_success
    assert result.reward == 1.0


@pytest.mark.parametrize(
    "issue",
    (
        "Problem with an investigation",
        "Incorrect information on a different report",
        "Incorrect information on your reports",
    ),
)
def test_valid_but_wrong_issue_is_detected(issue: str) -> None:
    prediction = gold_label()
    prediction["issue"] = issue
    prediction["tool_call"]["arguments"]["issue"] = issue

    result = score(_sample(), prediction)

    assert result.schema_valid
    assert not result.issue_match
    assert result.tool_arguments_valid
    assert result.task_success


@pytest.mark.parametrize(
    "company",
    ("Acme Bank", "Acme Financial Service", "Different Company"),
)
def test_valid_but_wrong_company_is_detected(company: str) -> None:
    prediction = gold_label()
    prediction["company"] = company
    prediction["tool_call"]["arguments"]["company"] = company

    result = score(_sample(), prediction)

    assert result.schema_valid
    assert not result.company_match
    assert result.tool_arguments_valid
    assert result.task_success


@pytest.mark.parametrize("urgency", ("low", "high"))
def test_valid_but_wrong_urgency_is_detected(urgency: str) -> None:
    prediction = gold_label()
    prediction["urgency"] = urgency

    result = score(_sample(), prediction)

    assert result.schema_valid
    assert not result.urgency_match
    assert not result.task_success


WRONG_TOOLS = (
    {
        "name": "close_no_action",
        "arguments": {"reason": "no_consumer_harm_detected"},
    },
    {
        "name": "start_refund_workflow",
        "arguments": {
            "company": "Acme Financial Services",
            "issue": "Incorrect information on your report",
            "evidence_required": True,
        },
    },
    {
        "name": "escalate_to_regulator",
        "arguments": {"complaint_id": 123, "reason": "Incorrect information on your report"},
    },
)


@pytest.mark.parametrize("tool_call", WRONG_TOOLS)
def test_wrong_tool_with_plausible_arguments_is_detected(tool_call: dict[str, Any]) -> None:
    prediction = gold_label()
    prediction["tool_call"] = copy.deepcopy(tool_call)

    result = score(_sample(), prediction)

    assert result.schema_valid
    assert not result.tool_choice_match
    assert result.tool_arguments_valid
    assert not result.task_success


def test_over_abstention_is_detected() -> None:
    prediction = gold_label()
    prediction["ambiguity_flag"] = True
    prediction["tool_call"] = {
        "name": "request_more_info",
        "arguments": {
            "missing_fields": ["company"],
            "question": "Which company is involved?",
        },
    }

    result = score(_sample(), prediction)

    assert result.schema_valid
    assert not result.ambiguity_flag_match
    assert not result.abstention_correct
    assert not result.task_success


@pytest.mark.parametrize(
    ("issue", "company"),
    (
        ("  Incorrect   information on your report  ", " ACME FINANCIAL SERVICES "),
        ("Incorrect\ninformation\ton your report", "Acme\nFinancial\tServices"),
        ("Ｉｎｃｏｒｒｅｃｔ information on your report", "Ａｃｍｅ Financial Services"),
        ("incorrect INFORMATION ON YOUR REPORT", "acme financial services"),
        ("Incorrect information on your report", "Acme Financial   Services"),
    ),
)
def test_unicode_case_and_whitespace_normalization(issue: str, company: str) -> None:
    prediction = gold_label()
    prediction["issue"] = issue
    prediction["company"] = company
    prediction["tool_call"]["arguments"] = {"company": company, "issue": issue}

    result = score(_sample(), json.dumps(prediction, ensure_ascii=False))

    assert result.schema_valid
    assert result.issue_match
    assert result.company_match
    assert result.tool_arguments_valid
    assert result.task_success
    assert result.reward == 1.0


def test_zero_width_unicode_is_not_silently_erased() -> None:
    prediction = gold_label()
    prediction["company"] = "Acme\u200b Financial Services"
    prediction["tool_call"]["arguments"]["company"] = prediction["company"]

    result = score(_sample(), prediction)

    assert result.schema_valid
    assert not result.company_match
    assert result.tool_arguments_valid
    assert result.task_success


def test_correct_decision_with_wrong_issue_text_is_success_under_v2() -> None:
    prediction = gold_label()
    prediction["issue"] = "A different but non-empty normalized issue"
    prediction["tool_call"]["arguments"]["issue"] = "Different operational wording"

    result = score(_sample(), prediction)

    assert result.scorer_version == 2
    assert not result.secondary_metrics["issue_normalized_match"]
    assert result.task_success
    assert result.reward == 1.0


def test_nonverbatim_escalation_reason_is_accepted() -> None:
    expected = gold_label()
    expected["urgency"] = "high"
    expected["tool_call"] = {
        "name": "escalate_to_regulator",
        "arguments": {"complaint_id": 123, "reason": "identity theft"},
    }
    prediction = copy.deepcopy(expected)
    prediction["tool_call"]["arguments"] = {
        "complaint_id": 999,
        "reason": "Immediate regulatory review is warranted for consumer harm.",
    }

    result = score({"label": expected}, prediction)

    assert result.tool_arguments_valid
    assert result.tool_arguments_semantic_valid
    assert result.task_success


def test_request_more_info_question_does_not_require_template_equality() -> None:
    expected = gold_label()
    expected["ambiguity_flag"] = True
    expected["tool_call"] = {
        "name": "request_more_info",
        "arguments": {
            "missing_fields": ["company"],
            "question": "Please provide the missing company for complaint 123.",
        },
    }
    prediction = copy.deepcopy(expected)
    prediction["tool_call"]["arguments"] = {
        "missing_fields": ["details"],
        "question": "Could you clarify the material facts needed to route this complaint?",
    }

    result = score({"label": expected}, prediction)

    assert result.tool_arguments_valid
    assert result.tool_arguments_semantic_valid
    assert result.task_success


def test_structurally_invalid_tool_arguments_fail_task_success() -> None:
    prediction = gold_label()
    del prediction["tool_call"]["arguments"]["issue"]

    result = score(_sample(), prediction)

    assert not result.schema_valid
    assert not result.tool_arguments_valid
    assert not result.task_success


def test_placeholder_free_text_is_secondary_and_never_template_compared() -> None:
    expected = gold_label()
    expected["ambiguity_flag"] = True
    expected["tool_call"] = {
        "name": "request_more_info",
        "arguments": {
            "missing_fields": ["details"],
            "question": "Please provide more details.",
        },
    }
    prediction = copy.deepcopy(expected)
    prediction["tool_call"]["arguments"]["question"] = "..."

    result = score({"label": expected}, prediction)

    assert result.schema_valid
    assert result.tool_arguments_valid
    assert not result.tool_arguments_semantic_valid
    assert result.task_success


def test_metadata_mismatch_cannot_change_decision_reward() -> None:
    correct = score(_sample(), gold_label())
    mismatched = gold_label()
    mismatched["product"] = "mortgage"
    mismatched["issue"] = "Unrelated issue text"
    mismatched["company"] = "Different Company"
    mismatched["tool_call"]["arguments"] = {
        "company": "Different Company",
        "issue": "Unrelated issue text",
    }

    result = score(_sample(), mismatched)

    assert not result.product_match
    assert not result.issue_match
    assert not result.company_match
    assert result.task_success == correct.task_success
    assert result.reward == correct.reward == 1.0


def test_secondary_metrics_are_explicitly_separate_from_decision_checks() -> None:
    result = score(_sample(), gold_label())

    assert set(result.decision_checks) == {
        "urgency",
        "ambiguity_flag",
        "tool_choice",
        "tool_arguments_structural",
    }
    assert set(result.secondary_metrics) == {
        "product_match",
        "issue_normalized_match",
        "company_normalized_match",
        "tool_arguments_semantic_valid",
        "abstention_correct",
    }


def test_extra_fields_preserve_partial_diagnostics_but_never_task_success() -> None:
    prediction = gold_label()
    prediction["explanation"] = "plausible but forbidden"

    result = score(_sample(), prediction)

    assert not result.schema_valid
    assert all(result.field_matches.values())
    assert result.tool_choice_match
    assert result.tool_arguments_valid
    assert not result.task_success
    assert result.reward == 0.8


def test_expected_label_can_be_read_from_canonical_json() -> None:
    result = score({"label_json": json.dumps(gold_label())}, gold_label())

    assert result.task_success


def test_invalid_expected_label_fails_loud() -> None:
    expected = gold_label()
    expected["product"] = "hallucinated"

    with pytest.raises(ValueError, match="sample label violates"):
        score({"label": expected}, gold_label())


def test_verifier_module_has_no_network_dependencies() -> None:
    module_path = Path(__file__).parents[1] / "src" / "forge" / "verify" / "verifier.py"
    source = module_path.read_text()

    assert "import socket" not in source
    assert "import urllib" not in source
    assert "import requests" not in source
    assert "import httpx" not in source


def test_gold_fixture_satisfies_schema() -> None:
    assert is_schema_valid(gold_label())
