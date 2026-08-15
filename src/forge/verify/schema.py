"""Canonical structured-ticket schema for the CFPB triage task.

This module is the single source of truth for every producer and verifier.  The
checked-in ``configs/task_schema.json`` is generated from :data:`TASK_SCHEMA`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

PRODUCTS: tuple[str, ...] = (
    "card",
    "credit_reporting",
    "debt_collection",
    "deposit_account",
    "money_service",
    "mortgage",
    "payday_personal_loan",
    "student_loan",
    "vehicle_loan",
)

TOOL_NAMES: tuple[str, ...] = (
    "escalate_to_regulator",
    "request_more_info",
    "route_to_company",
    "start_refund_workflow",
    "close_no_action",
)

MISSING_FIELDS: tuple[str, ...] = ("product", "issue", "company", "details")

TOOL_ARGUMENT_SCHEMAS: dict[str, dict[str, Any]] = {
    "escalate_to_regulator": {
        "type": "object",
        "additionalProperties": False,
        "required": ["complaint_id", "reason"],
        "properties": {
            "complaint_id": {"type": "integer", "minimum": 1},
            "reason": {"type": "string", "minLength": 1},
        },
    },
    "request_more_info": {
        "type": "object",
        "additionalProperties": False,
        "required": ["missing_fields", "question"],
        "properties": {
            "missing_fields": {
                "type": "array",
                "items": {"type": "string", "enum": list(MISSING_FIELDS)},
                "minItems": 1,
                "uniqueItems": True,
            },
            "question": {"type": "string", "minLength": 1},
        },
    },
    "route_to_company": {
        "type": "object",
        "additionalProperties": False,
        "required": ["company", "issue"],
        "properties": {
            "company": {"type": "string", "minLength": 1},
            "issue": {"type": "string", "minLength": 1},
        },
    },
    "start_refund_workflow": {
        "type": "object",
        "additionalProperties": False,
        "required": ["company", "issue", "evidence_required"],
        "properties": {
            "company": {"type": "string", "minLength": 1},
            "issue": {"type": "string", "minLength": 1},
            "evidence_required": {"type": "boolean"},
        },
    },
    "close_no_action": {
        "type": "object",
        "additionalProperties": False,
        "required": ["reason"],
        "properties": {
            "reason": {
                "type": "string",
                "enum": ["already_resolved", "duplicate_or_spam", "no_consumer_harm_detected"],
            }
        },
    },
}


def _tool_variant(name: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "arguments"],
        "properties": {
            "name": {"const": name},
            "arguments": TOOL_ARGUMENT_SCHEMAS[name],
        },
    }


TASK_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://frontier-forge.local/schemas/task-output-v1.json",
    "title": "CFPB structured triage ticket v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "product",
        "issue",
        "company",
        "urgency",
        "ambiguity_flag",
        "tool_call",
    ],
    "properties": {
        "product": {"type": "string", "enum": list(PRODUCTS)},
        "issue": {"type": "string", "minLength": 1},
        "company": {
            "oneOf": [
                {"type": "string", "minLength": 1},
                {"type": "null"},
            ]
        },
        "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
        "ambiguity_flag": {"type": "boolean"},
        "tool_call": {"oneOf": [_tool_variant(name) for name in TOOL_NAMES]},
    },
    "allOf": [
        {
            "if": {
                "required": ["ambiguity_flag"],
                "properties": {"ambiguity_flag": {"const": True}},
            },
            "then": {
                "properties": {
                    "tool_call": {"properties": {"name": {"const": "request_more_info"}}}
                }
            },
        },
        {
            "if": {
                "required": ["tool_call"],
                "properties": {
                    "tool_call": {
                        "required": ["name"],
                        "properties": {"name": {"const": "request_more_info"}},
                    }
                },
            },
            "then": {"properties": {"ambiguity_flag": {"const": True}}},
        },
    ],
}

_VALIDATOR = Draft202012Validator(TASK_SCHEMA)


def schema_errors(value: object) -> tuple[str, ...]:
    """Return deterministic, human-readable validation errors for ``value``."""

    errors: list[str] = []
    for error in sorted(_VALIDATOR.iter_errors(value), key=lambda item: list(item.absolute_path)):
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"{path}: {error.message}")
    return tuple(errors)


def is_schema_valid(value: object) -> bool:
    """Return whether ``value`` exactly satisfies the v1 task schema."""

    return not schema_errors(value)


def export_schema(path: Path) -> None:
    """Write the canonical schema as stable, reviewable JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(TASK_SCHEMA, indent=2, ensure_ascii=False) + "\n")
