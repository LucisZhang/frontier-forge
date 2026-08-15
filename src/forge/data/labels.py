"""Deterministic Phase-1 label derivation for CFPB complaint rows."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from forge.verify.schema import MISSING_FIELDS, PRODUCTS, is_schema_valid, schema_errors

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RULES_PATH = REPO_ROOT / "configs" / "label_rules.yaml"
_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    """Normalize user/source text without erasing meaningful punctuation."""

    if value is None:
        return ""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", str(value))).strip()


def comparison_text(value: object) -> str:
    """Return the canonical comparison form used by rules and the verifier."""

    return normalize_text(value).casefold()


@dataclass(frozen=True)
class LabelRules:
    version: int
    product_map: Mapping[str, str]
    min_narrative_chars: int
    ambiguity_phrases: tuple[str, ...]
    missing_field_order: tuple[str, ...]
    high_keywords: tuple[str, ...]
    medium_keywords: tuple[str, ...]
    default_urgency: str
    no_action_phrases: tuple[str, ...]
    refund_phrases: tuple[str, ...]


def _normalized_terms(values: list[object]) -> tuple[str, ...]:
    return tuple(comparison_text(value) for value in values)


@lru_cache(maxsize=8)
def load_rules(path: Path = DEFAULT_RULES_PATH) -> LabelRules:
    """Load and validate the documented rule table."""

    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    product_map = dict(raw["taxonomy"]["product_map"])
    unknown_targets = set(product_map.values()) - set(PRODUCTS)
    if unknown_targets:
        raise ValueError(f"label rules map to unknown product enum(s): {sorted(unknown_targets)}")
    if set(product_map.values()) != set(PRODUCTS):
        missing = set(PRODUCTS) - set(product_map.values())
        raise ValueError(f"label rules do not cover product enum(s): {sorted(missing)}")

    ambiguity = raw["ambiguity"]
    urgency = raw["urgency"]
    tools = raw["tools"]
    field_order = tuple(ambiguity["missing_field_order"])
    if set(field_order) != set(MISSING_FIELDS):
        raise ValueError("ambiguity.missing_field_order must contain each schema field once")
    default_urgency = str(urgency["default"])
    if default_urgency not in {"low", "medium", "high"}:
        raise ValueError(f"invalid default urgency: {default_urgency!r}")

    return LabelRules(
        version=int(raw["version"]),
        product_map=product_map,
        min_narrative_chars=int(ambiguity["min_narrative_chars"]),
        ambiguity_phrases=_normalized_terms(ambiguity["phrases"]),
        missing_field_order=field_order,
        high_keywords=_normalized_terms(urgency["high_keywords"]),
        medium_keywords=_normalized_terms(urgency["medium_keywords"]),
        default_urgency=default_urgency,
        no_action_phrases=_normalized_terms(tools["no_action_phrases"]),
        refund_phrases=_normalized_terms(tools["refund_phrases"]),
    )


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(needle in haystack for needle in needles)


def _ordered_missing(fields: set[str], rules: LabelRules) -> list[str]:
    return [field for field in rules.missing_field_order if field in fields]


def derive_label(row: Mapping[str, Any], rules: LabelRules | None = None) -> dict[str, Any]:
    """Derive one v1 task label from an upstream CFPB row.

    The function has no I/O and no wall-clock dependency. Unknown source products
    fail closed because silently inventing a taxonomy mapping would invalidate the
    frozen split contract.
    """

    rules = load_rules() if rules is None else rules
    source_product = normalize_text(row.get("product"))
    try:
        product = rules.product_map[source_product]
    except KeyError as exc:
        raise ValueError(f"unmapped CFPB product: {source_product!r}") from exc

    complaint_id = int(row["complaint_id"])
    issue = normalize_text(row.get("issue"))
    company_text = normalize_text(row.get("company"))
    company: str | None = company_text or None
    narrative = normalize_text(row.get("narrative"))
    rule_text = comparison_text(f"{issue} {narrative}")

    missing: set[str] = set()
    if not issue:
        missing.add("issue")
        issue = "unspecified"
    if company is None:
        missing.add("company")
    if len(narrative) < rules.min_narrative_chars:
        missing.add("details")
    if _contains_any(rule_text, rules.ambiguity_phrases):
        missing.add("details")

    ambiguity = bool(missing)
    if _contains_any(rule_text, rules.high_keywords):
        urgency = "high"
    elif _contains_any(rule_text, rules.medium_keywords):
        urgency = "medium"
    else:
        urgency = rules.default_urgency

    if ambiguity:
        missing_fields = _ordered_missing(missing, rules)
        joined = ", ".join(missing_fields)
        tool_call: dict[str, Any] = {
            "name": "request_more_info",
            "arguments": {
                "missing_fields": missing_fields,
                "question": f"Please provide the missing {joined} for complaint {complaint_id}.",
            },
        }
    elif _contains_any(rule_text, rules.no_action_phrases):
        tool_call = {
            "name": "close_no_action",
            "arguments": {"reason": "already_resolved"},
        }
    elif urgency == "high":
        tool_call = {
            "name": "escalate_to_regulator",
            "arguments": {"complaint_id": complaint_id, "reason": issue},
        }
    elif _contains_any(rule_text, rules.refund_phrases):
        # company cannot be None here: that condition takes the abstention path.
        tool_call = {
            "name": "start_refund_workflow",
            "arguments": {"company": company, "issue": issue, "evidence_required": True},
        }
    elif company is not None:
        tool_call = {
            "name": "route_to_company",
            "arguments": {"company": company, "issue": issue},
        }
    else:  # Defensive fallback; current ambiguity rules make this branch unreachable.
        tool_call = {
            "name": "close_no_action",
            "arguments": {"reason": "no_consumer_harm_detected"},
        }

    label = {
        "product": product,
        "issue": issue,
        "company": company,
        "urgency": urgency,
        "ambiguity_flag": ambiguity,
        "tool_call": tool_call,
    }
    if not is_schema_valid(label):
        raise AssertionError(f"derived label violates task schema: {schema_errors(label)}")
    return label


def canonical_label_json(row: Mapping[str, Any], rules: LabelRules | None = None) -> str:
    """Return a stable compact JSON encoding of :func:`derive_label`."""

    return json.dumps(
        derive_label(row, rules),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
