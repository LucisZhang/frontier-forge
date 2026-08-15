"""Pure, deterministic scoring for structured CFPB triage outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from forge.data.labels import comparison_text
from forge.verify.schema import is_schema_valid, schema_errors

_TOP_LEVEL_FIELDS = (
    "product",
    "issue",
    "company",
    "urgency",
    "ambiguity_flag",
)


@dataclass(frozen=True)
class ScoreBreakdown:
    """Auditable verifier result used by evaluation and later GRPO reward code."""

    json_valid: bool
    schema_valid: bool
    product_match: bool
    issue_match: bool
    company_match: bool
    urgency_match: bool
    ambiguity_flag_match: bool
    tool_choice_match: bool
    tool_arguments_match: bool
    abstention_correct: bool
    task_success: bool
    reward: float
    errors: tuple[str, ...] = ()

    @property
    def field_matches(self) -> dict[str, bool]:
        """Per-field exact/normalized matches with stable public field names."""

        return {
            "product": self.product_match,
            "issue": self.issue_match,
            "company": self.company_match,
            "urgency": self.urgency_match,
            "ambiguity_flag": self.ambiguity_flag_match,
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
    if isinstance(expected, Mapping) and isinstance(predicted, Mapping):
        return set(expected) == set(predicted) and all(
            _normalized_equal(expected[key], predicted[key]) for key in expected
        )
    if isinstance(expected, list) and isinstance(predicted, list):
        return len(expected) == len(predicted) and all(
            _normalized_equal(left, right) for left, right in zip(expected, predicted, strict=True)
        )
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


def _zero(errors: tuple[str, ...]) -> ScoreBreakdown:
    return ScoreBreakdown(
        json_valid=False,
        schema_valid=False,
        product_match=False,
        issue_match=False,
        company_match=False,
        urgency_match=False,
        ambiguity_flag_match=False,
        tool_choice_match=False,
        tool_arguments_match=False,
        abstention_correct=False,
        task_success=False,
        reward=0.0,
        errors=errors,
    )


def score(sample: Mapping[str, Any], model_output_json: object) -> ScoreBreakdown:
    """Score one output against its frozen label with no I/O or model calls.

    String fields ``issue``, ``company``, and tool-argument strings use Unicode
    NFKC, whitespace-collapse, and case-insensitive comparison. Enum values,
    booleans, numbers, list order, and object keys remain strict.
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

    expected_name, expected_arguments = _tool_parts(expected["tool_call"])
    predicted_name, predicted_arguments = _tool_parts(predicted.get("tool_call"))
    tool_choice_match = isinstance(predicted_name, str) and predicted_name == expected_name
    tool_arguments_match = tool_choice_match and _normalized_equal(
        expected_arguments, predicted_arguments
    )
    abstention_correct = _is_abstention(expected) == _is_abstention(predicted)

    correctness = (
        product_match,
        issue_match,
        company_match,
        urgency_match,
        ambiguity_match,
        tool_choice_match,
        tool_arguments_match,
        abstention_correct,
    )
    task_success = valid and all(correctness)
    # Partial reward is intentionally transparent and bounded. A schema-valid
    # output owns 20%; the eight semantic checks share the remaining 80%.
    reward = round((0.2 if valid else 0.0) + 0.1 * sum(correctness), 6)

    return ScoreBreakdown(
        json_valid=True,
        schema_valid=valid,
        product_match=product_match,
        issue_match=issue_match,
        company_match=company_match,
        urgency_match=urgency_match,
        ambiguity_flag_match=ambiguity_match,
        tool_choice_match=tool_choice_match,
        tool_arguments_match=tool_arguments_match,
        abstention_correct=abstention_correct,
        task_success=task_success,
        reward=reward,
        errors=errors,
    )
