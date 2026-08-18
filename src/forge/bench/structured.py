"""Structured-output, tool-choice constraint-tax, and two-pass benchmark."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

import httpx

from forge.teacher.filters import breakdown_dict
from forge.verify.verifier import score

from .loadgen import VramSampler, normalize_verifier_input
from .metrics import parse_prometheus, percentile, prometheus_delta, summarize_vllm_metrics
from .workload import load_workload


def schema_for_field_count(field_count: int) -> dict[str, Any]:
    """A backend-portable schema whose complexity scales by required fields."""

    properties = {
        f"field_{index:02d}": {
            "type": "string",
            "enum": [f"value_{index:02d}_a", f"value_{index:02d}_b"],
        }
        for index in range(field_count)
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def decoder_compatible_triage_schema(*, tool_name: str | None = None) -> dict[str, Any]:
    """Keep decoder constraints portable; the strict verifier remains authoritative."""

    tool_name_schema: dict[str, Any] = (
        {"const": tool_name}
        if tool_name is not None
        else {
            "type": "string",
            "enum": [
                "request_more_info",
                "close_no_action",
                "escalate_to_regulator",
                "start_refund_workflow",
                "route_to_company",
            ],
        }
    )
    return {
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
            "product": {
                "type": "string",
                "enum": [
                    "card",
                    "credit_reporting",
                    "debt_collection",
                    "deposit_account",
                    "money_service",
                    "mortgage",
                    "payday_personal_loan",
                    "student_loan",
                    "vehicle_loan",
                ],
            },
            "issue": {"type": "string"},
            "company": {"type": ["string", "null"]},
            "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
            "ambiguity_flag": {"type": "boolean"},
            "tool_call": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "arguments"],
                "properties": {
                    "name": tool_name_schema,
                    "arguments": {"type": "object"},
                },
            },
        },
    }


def tool_registry() -> list[dict[str, Any]]:
    definitions = {
        "request_more_info": {
            "type": "object",
            "additionalProperties": False,
            "required": ["missing_fields", "question"],
            "properties": {
                "missing_fields": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["product", "issue", "company", "details"],
                    },
                },
                "question": {"type": "string"},
            },
        },
        "close_no_action": {
            "type": "object",
            "additionalProperties": False,
            "required": ["reason"],
            "properties": {
                "reason": {
                    "type": "string",
                    "enum": [
                        "already_resolved",
                        "duplicate_or_spam",
                        "no_consumer_harm_detected",
                    ],
                }
            },
        },
        "escalate_to_regulator": {
            "type": "object",
            "additionalProperties": False,
            "required": ["complaint_id", "reason"],
            "properties": {
                "complaint_id": {"type": "integer"},
                "reason": {"type": "string"},
            },
        },
        "start_refund_workflow": {
            "type": "object",
            "additionalProperties": False,
            "required": ["company", "issue", "evidence_required"],
            "properties": {
                "company": {"type": "string"},
                "issue": {"type": "string"},
                "evidence_required": {"type": "boolean"},
            },
        },
        "route_to_company": {
            "type": "object",
            "additionalProperties": False,
            "required": ["company", "issue"],
            "properties": {
                "company": {"type": "string"},
                "issue": {"type": "string"},
            },
        },
    }
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"CFPB triage action: {name}",
                "strict": True,
                "parameters": parameters,
            },
        }
        for name, parameters in definitions.items()
    ]


async def _post_chat(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if extra:
        payload.update(extra)
    started = time.perf_counter()
    try:
        response = await client.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=payload)
        elapsed = time.perf_counter() - started
        if response.is_error:
            return {
                "status": response.status_code,
                "latency_s": elapsed,
                "error": response.text[:1000],
                "content": "",
                "tool_calls": [],
                "usage": None,
            }
        value = response.json()
        choices = value.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        return {
            "status": response.status_code,
            "latency_s": elapsed,
            "error": None,
            "content": message.get("content") or "",
            "tool_calls": message.get("tool_calls") or [],
            "usage": value.get("usage"),
            "finish_reason": choices[0].get("finish_reason") if choices else None,
            "request_id": response.headers.get("x-request-id"),
        }
    except httpx.HTTPError as exc:
        return {
            "status": None,
            "latency_s": time.perf_counter() - started,
            "error": f"{type(exc).__name__}:{exc}",
            "content": "",
            "tool_calls": [],
            "usage": None,
        }


def _tool_name(response: Mapping[str, Any]) -> str | None:
    calls = response.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    function = calls[0].get("function") if isinstance(calls[0], Mapping) else None
    if not isinstance(function, Mapping):
        return None
    value = function.get("name")
    return value if isinstance(value, str) else None


def _safe_score(label: Mapping[str, Any], content: object) -> dict[str, Any]:
    return breakdown_dict(score({"label": label}, normalize_verifier_input(content)))


async def _compile_bench(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    field_counts: list[int],
    steady_repetitions: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    results = []
    for count in field_counts:
        schema = schema_for_field_count(count)
        keys = ", ".join(schema["properties"])
        messages = [
            {"role": "system", "content": "Return only the requested JSON object."},
            {
                "role": "user",
                "content": f"Return all of these fields using their first allowed value: {keys}.",
            },
        ]
        extra = {"structured_outputs": {"json": schema}}
        cold = await _post_chat(
            client,
            base_url=base_url,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            extra=extra,
        )
        steady = [
            await _post_chat(
                client,
                base_url=base_url,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                extra=extra,
            )
            for _ in range(steady_repetitions)
        ]
        control = await _post_chat(
            client,
            base_url=base_url,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        steady_latencies = [item["latency_s"] for item in steady if item["error"] is None]
        steady_median = percentile(steady_latencies, 0.5)
        results.append(
            {
                "schema_required_fields": count,
                "cold": cold,
                "steady": steady,
                "unconstrained_control": control,
                "cold_compile_overhead_s": (
                    cold["latency_s"] - steady_median if steady_median is not None else None
                ),
                "steady_constraint_overhead_s": (
                    steady_median - control["latency_s"]
                    if steady_median is not None and control["error"] is None
                    else None
                ),
            }
        )
    return results


async def _constraint_tax_bench(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tools = tool_registry()
    schema = decoder_compatible_triage_schema()
    records: list[dict[str, Any]] = []
    for row in rows:
        common = {"tools": tools, "tool_choice": "required"}
        unconstrained = await _post_chat(
            client,
            base_url=base_url,
            model=model,
            messages=row["messages"],
            max_tokens=int(row["max_tokens"]),
            extra=common,
        )
        simultaneous = await _post_chat(
            client,
            base_url=base_url,
            model=model,
            messages=row["messages"],
            max_tokens=int(row["max_tokens"]),
            extra={**common, "structured_outputs": {"json": schema}},
        )
        simultaneous_verifier_input = normalize_verifier_input(simultaneous["content"])
        before_score = _safe_score(row["label"], simultaneous_verifier_input)
        pass1 = unconstrained
        selected_tool = _tool_name(pass1)
        pass2: dict[str, Any] | None = None
        pass2_verifier_input: str | None = None
        after_score: dict[str, Any] | None = None
        if selected_tool is not None:
            messages = [
                *row["messages"],
                {
                    "role": "user",
                    "content": (
                        "The routing pass selected tool "
                        f"{selected_tool}. Return the complete six-field ticket JSON now; "
                        f"tool_call.name must be {selected_tool}."
                    ),
                },
            ]
            pass2 = await _post_chat(
                client,
                base_url=base_url,
                model=model,
                messages=messages,
                max_tokens=int(row["max_tokens"]),
                extra={
                    "structured_outputs": {
                        "json": decoder_compatible_triage_schema(tool_name=selected_tool)
                    }
                },
            )
            pass2_verifier_input = normalize_verifier_input(pass2["content"])
            after_score = _safe_score(row["label"], pass2_verifier_input)
        expected_tool = str(row["label"]["tool_call"]["name"])
        records.append(
            {
                "request_id": row["request_id"],
                "complaint_id": row["complaint_id"],
                "expected_tool": expected_tool,
                "unconstrained_tool_name": _tool_name(unconstrained),
                "simultaneous_tool_name": _tool_name(simultaneous),
                "unconstrained": unconstrained,
                "simultaneous_constraint_and_tool": simultaneous,
                "simultaneous_verifier_input": simultaneous_verifier_input,
                "simultaneous_score": before_score,
                "two_pass_selected_tool": selected_tool,
                "two_pass_second": pass2,
                "two_pass_verifier_input": pass2_verifier_input,
                "two_pass_score": after_score,
            }
        )
    total = len(records)
    before_successes = sum(item["simultaneous_score"]["task_success"] for item in records)
    after_eligible = [item for item in records if item["two_pass_score"] is not None]
    after_successes = sum(item["two_pass_score"]["task_success"] for item in after_eligible)
    unconstrained_tool_calls = sum(item["unconstrained_tool_name"] is not None for item in records)
    simultaneous_tool_calls = sum(item["simultaneous_tool_name"] is not None for item in records)
    before_latencies = [item["simultaneous_constraint_and_tool"]["latency_s"] for item in records]
    after_latencies = [
        item["unconstrained"]["latency_s"] + item["two_pass_second"]["latency_s"]
        for item in after_eligible
        if item["two_pass_second"] is not None
    ]
    summary = {
        "requests": total,
        "unconstrained_tool_call_rate": unconstrained_tool_calls / total if total else 0.0,
        "simultaneous_schema_tool_call_rate": simultaneous_tool_calls / total if total else 0.0,
        "constraint_tax_tool_call_rate_delta": (simultaneous_tool_calls - unconstrained_tool_calls)
        / total
        if total
        else 0.0,
        "simultaneous_task_success": before_successes / total if total else 0.0,
        "two_pass_task_success": after_successes / total if total else 0.0,
        "two_pass_coverage": len(after_eligible) / total if total else 0.0,
        "mitigation_task_success_delta": (after_successes - before_successes) / total
        if total
        else 0.0,
        "simultaneous_latency_p50_s": percentile(before_latencies, 0.5),
        "simultaneous_latency_p95_s": percentile(before_latencies, 0.95),
        "two_pass_latency_p50_s": percentile(after_latencies, 0.5),
        "two_pass_latency_p95_s": percentile(after_latencies, 0.95),
        "latency_delta_p50_s": (
            percentile(after_latencies, 0.5) - percentile(before_latencies, 0.5)
            if after_latencies and before_latencies
            else None
        ),
        "task_success_denominator": (
            "all original requests; uncovered two-pass rows count as failure"
        ),
    }
    return summary, records


async def run_structured_benchmark(
    config: Mapping[str, Any], *, base_url: str, smoke: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    workload = load_workload(config, smoke=smoke)
    settings = config["structured"]
    count = (
        min(2, len(workload))
        if smoke
        else min(int(settings["measurement_requests"]), len(workload))
    )
    rows = workload[:count]
    repetitions = 1 if smoke else int(settings["steady_repetitions"])
    timeout = httpx.Timeout(float(config["workload"]["deadline_s"]))
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        health = await client.get(f"{base_url.rstrip('/')}/health")
        health.raise_for_status()
        before_text = (await client.get(f"{base_url.rstrip('/')}/metrics")).text
        sampler = VramSampler(enabled=not smoke)
        sampler_task = asyncio.create_task(sampler.run())
        compile_results = await _compile_bench(
            client,
            base_url=base_url,
            model=str(config["model"]["served_name"]),
            field_counts=[int(item) for item in settings["schema_field_counts"]],
            steady_repetitions=repetitions,
            max_tokens=int(settings["schema_max_tokens"]),
        )
        constraint_summary, constraint_records = await _constraint_tax_bench(
            client,
            base_url=base_url,
            model=str(config["model"]["served_name"]),
            rows=rows,
        )
        sampler.stop()
        await sampler_task
        after_text = (await client.get(f"{base_url.rstrip('/')}/metrics")).text
    server_metrics = summarize_vllm_metrics(
        prometheus_delta(parse_prometheus(before_text), parse_prometheus(after_text))
    )
    return (
        {
            "backend": settings["backend"],
            "schema_compile": compile_results,
            "constraint_tax_and_mitigation": constraint_summary,
            "server": server_metrics,
            "vram": sampler.summary(),
        },
        constraint_records,
    )
