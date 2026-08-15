"""Pure, deterministic scoring for structured CFPB triage outputs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from forge.data.labels import comparison_text
from forge.verify.schema import (
    TOOL_ARGUMENT_SCHEMAS,
    is_schema_valid,
    schema_errors,
)

SCORER_VERSION = 2
_TOP_LEVEL_FIELDS = (
    "product",
    "issue",
    "company",
    "urgency",
    "ambiguity_flag",
)
_PLACEHOLDER_TEXT = frozenset(
    {
        "...",
        "n/a",
        "na",
        "none",
        "null",
        "unknown",
        "reason",
        "question",
        "string",
        "todo",
    }
)
_WORD = re.compile(r"[^\W_]", re.UNICODE)
_ARGUMENT_VALIDATORS = {
    name: Draft202012Validator(schema) for name, schema in TOOL_ARGUMENT_SCHEMAS.items()
}


@dataclass(frozen=True)
class ScoreBreakdown:
    """Auditable scorer-v2 result used by evaluation and future GRPO reward."""

    scorer_version: int
    json_valid: bool
    schema_valid: bool
    product_match: bool
    issue_match: bool
    company_match: bool
    urgency_match: bool
    ambiguity_flag_match: bool
    tool_choice_match: bool
    tool_arguments_valid: bool
    tool_arguments_semantic_valid: bool
    abstention_correct: bool
    task_success: bool
    reward: float
    errors: tuple[str, ...] = ()

    @property
    def field_matches(self) -> dict[str, bool]:
        """Per-field diagnostics; metadata matches do not affect task success."""

        return {
            "product": self.product_match,
            "issue": self.issue_match,
            "company": self.company_match,
            "urgency": self.urgency_match,
            "ambiguity_flag": self.ambiguity_flag_match,
        }

    @property
    def decision_checks(self) -> dict[str, bool]:
        """The four D3.1 checks that define task success after schema validity."""

        return {
            "urgency": self.urgency_match,
            "ambiguity_flag": self.ambiguity_flag_match,
            "tool_choice": self.tool_choice_match,
            "tool_arguments_structural": self.tool_arguments_valid,
        }

    @property
    def secondary_metrics(self) -> dict[str, bool]:
        """Diagnostics excluded from task success and every RL reward."""

        return {
            "product_match": self.product_match,
            "issue_normalized_match": self.issue_match,
            "company_normalized_match": self.company_match,
            "tool_arguments_semantic_valid": self.tool_arguments_semantic_valid,
            "abstention_correct": self.abstention_correct,
        }


def _expected_label(sample: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("label", "expected"):
        value = sample.get(key)
        if isinstance(value, Mapping):
            return value
    label_json = sample.get("label_json")
    if isinstance(label_json, str):
        value = json.loads(label_json)
        if isinstance(value, Mapping):
            return value
    if all(field in sample for field in (*_TOP_LEVEL_FIELDS, "tool_call")):
        return sample
    raise ValueError("sample must contain label, expected, label_json, or the v1 label fields")


def _parse_output(value: object) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            return None, (f"invalid UTF-8: {exc}",)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            return None, (f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}",)
    if not isinstance(value, Mapping):
        return None, ("model output must be a JSON object",)
    return value, ()


def _normalized_equal(expected: object, predicted: object) -> bool:
    if isinstance(expected, str) and isinstance(predicted, str):
        return comparison_text(expected) == comparison_text(predicted)
    if expected is None or predicted is None:
        return expected is predicted
    if isinstance(expected, bool) or isinstance(predicted, bool):
        return expected is predicted
    return type(expected) is type(predicted) and expected == predicted


def _top_level_match(expected: Mapping[str, Any], predicted: Mapping[str, Any], key: str) -> bool:
    if key not in predicted:
        return False
    if key in {"issue", "company"}:
        return _normalized_equal(expected[key], predicted[key])
    return type(expected[key]) is type(predicted[key]) and expected[key] == predicted[key]


def _tool_parts(value: object) -> tuple[object, object]:
    if not isinstance(value, Mapping):
        return None, None
    return value.get("name"), value.get("arguments")


def _is_abstention(value: Mapping[str, Any]) -> bool:
    name, _ = _tool_parts(value.get("tool_call"))
    return value.get("ambiguity_flag") is True and name == "request_more_info"


def _tool_arguments_valid(name: object, arguments: object) -> bool:
    if not isinstance(name, str) or name not in _ARGUMENT_VALIDATORS:
        return False
    return not tuple(_ARGUMENT_VALIDATORS[name].iter_errors(arguments))


def _meaningful_text(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = comparison_text(value)
    return (
        len(normalized) >= 3
        and normalized not in _PLACEHOLDER_TEXT
        and _WORD.search(normalized) is not None
    )


def _tool_arguments_semantic_valid(name: object, arguments: object) -> bool:
    """Check minimal grounded meaning without comparing free text verbatim."""

    if not _tool_arguments_valid(name, arguments) or not isinstance(arguments, Mapping):
        return False
    if name == "request_more_info":
        return _meaningful_text(arguments.get("question"))
    if name == "escalate_to_regulator":
        return _meaningful_text(arguments.get("reason"))
    if name in {"route_to_company", "start_refund_workflow"}:
        return _meaningful_text(arguments.get("company")) and _meaningful_text(
            arguments.get("issue")
        )
    return name == "close_no_action"


def _zero(errors: tuple[str, ...]) -> ScoreBreakdown:
    return ScoreBreakdown(
        scorer_version=SCORER_VERSION,
        json_valid=False,
        schema_valid=False,
        product_match=False,
        issue_match=False,
        company_match=False,
        urgency_match=False,
        ambiguity_flag_match=False,
        tool_choice_match=False,
        tool_arguments_valid=False,
        tool_arguments_semantic_valid=False,
        abstention_correct=False,
        task_success=False,
        reward=0.0,
        errors=errors,
    )


def score(sample: Mapping[str, Any], model_output_json: object) -> ScoreBreakdown:
    """Score one output under D3.1 scorer v2 with no I/O or model calls.

    ``task_success`` requires a valid output plus correct urgency, ambiguity,
    tool choice, and structurally valid arguments.  Product, normalized issue,
    normalized company, and non-verbatim free-text semantics remain visible as
    secondary metrics but cannot affect task success or reward.
    """

    expected = _expected_label(sample)
    if not is_schema_valid(expected):
        raise ValueError(f"sample label violates task schema: {schema_errors(expected)}")

    predicted, parse_errors = _parse_output(model_output_json)
    if predicted is None:
        return _zero(parse_errors)

    valid = is_schema_valid(predicted)
    errors = schema_errors(predicted)
    product_match = _top_level_match(expected, predicted, "product")
    issue_match = _top_level_match(expected, predicted, "issue")
    company_match = _top_level_match(expected, predicted, "company")
    urgency_match = _top_level_match(expected, predicted, "urgency")
    ambiguity_match = _top_level_match(expected, predicted, "ambiguity_flag")

    expected_name, _ = _tool_parts(expected["tool_call"])
    predicted_name, predicted_arguments = _tool_parts(predicted.get("tool_call"))
    tool_choice_match = isinstance(predicted_name, str) and predicted_name == expected_name
    tool_arguments_valid = _tool_arguments_valid(predicted_name, predicted_arguments)
    tool_arguments_semantic_valid = _tool_arguments_semantic_valid(
        predicted_name, predicted_arguments
    )
    abstention_correct = _is_abstention(expected) == _is_abstention(predicted)

    decision_checks = (
        urgency_match,
        ambiguity_match,
        tool_choice_match,
        tool_arguments_valid,
    )
    task_success = valid and all(decision_checks)
    # Schema compliance and the four decision checks receive equal weight.
    # Source-metadata normalization and free-text wording are deliberately absent.
    reward = round(0.2 * sum((valid, *decision_checks)), 6)

    return ScoreBreakdown(
        scorer_version=SCORER_VERSION,
        json_valid=True,
        schema_valid=valid,
        product_match=product_match,
        issue_match=issue_match,
        company_match=company_match,
        urgency_match=urgency_match,
        ambiguity_flag_match=ambiguity_match,
        tool_choice_match=tool_choice_match,
        tool_arguments_valid=tool_arguments_valid,
        tool_arguments_semantic_valid=tool_arguments_semantic_valid,
        abstention_correct=abstention_correct,
        task_success=task_success,
        reward=reward,
        errors=errors,
    )
