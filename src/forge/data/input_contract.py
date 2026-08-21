"""Canonical model input contract for CFPB structured triage.

Input contract v2 is locked by docs/engineering-log/DECISIONS.md D3.1.  Every
model-facing producer must call :func:`build_model_input` rather than selecting
fields ad hoc.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

INPUT_CONTRACT_VERSION = 2
MODEL_INPUT_FIELDS: tuple[str, ...] = (
    "complaint_id",
    "narrative",
    "source_product",
    "source_issue",
    "source_company",
)


def _first_present(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    joined = ", ".join(keys)
    raise KeyError(f"model input row is missing required field (accepted keys: {joined})")


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"model input {field} must be a non-empty string")
    return value


def build_model_input(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build the only supported model-visible input shape.

    Source metadata aliases are accepted at ingestion boundaries, but the
    returned keys are always the stable v2 names.  Narrative text and metadata
    are preserved verbatim so prompt construction cannot silently alter evidence.
    """

    complaint_id = _first_present(row, "complaint_id")
    if isinstance(complaint_id, bool):
        raise ValueError("model input complaint_id must be an integer")
    try:
        complaint_id = int(complaint_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("model input complaint_id must be an integer") from exc
    if complaint_id < 1:
        raise ValueError("model input complaint_id must be positive")

    narrative = _required_text(_first_present(row, "narrative", "complaint_narrative"), "narrative")
    source_product = _required_text(
        _first_present(row, "source_product", "product"), "source_product"
    )
    source_issue = _first_present(row, "source_issue", "issue")
    if source_issue is not None and not isinstance(source_issue, str):
        raise ValueError("model input source_issue must be a string or null")
    if isinstance(source_issue, str) and not source_issue.strip():
        source_issue = None
    source_company = _first_present(row, "source_company", "company")
    if source_company is not None and not isinstance(source_company, str):
        raise ValueError("model input source_company must be a string or null")
    if isinstance(source_company, str) and not source_company.strip():
        source_company = None

    return {
        "complaint_id": complaint_id,
        "narrative": narrative,
        "source_product": source_product,
        "source_issue": source_issue,
        "source_company": source_company,
    }


def model_input_json(row: Mapping[str, Any]) -> str:
    """Serialize input v2 deterministically for prompts, hashes, and datasets."""

    return json.dumps(
        build_model_input(row),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
