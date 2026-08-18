"""Metric helpers that keep client observations separate from vLLM metrics."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

_SAMPLE = re.compile(
    r"^(?P<name>[^\s{]+)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]Inf|NaN)$"
)


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_latencies(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean_s": sum(values) / len(values) if values else None,
        "p50_s": percentile(values, 0.50),
        "p95_s": percentile(values, 0.95),
    }


def parse_prometheus(text: str) -> dict[str, dict[tuple[tuple[str, str], ...], float]]:
    """Parse numeric Prometheus samples without importing the server library."""

    parsed: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.fullmatch(line)
        if not match:
            continue
        value_text = match.group("value")
        value = float(value_text)
        labels_text = match.group("labels") or ""
        labels: list[tuple[str, str]] = []
        if labels_text:
            for item in re.finditer(r'(\w+)="((?:\\.|[^"\\])*)"(?:,|$)', labels_text):
                labels.append(
                    (item.group(1), bytes(item.group(2), "utf-8").decode("unicode_escape"))
                )
        parsed.setdefault(match.group("name"), {})[tuple(sorted(labels))] = value
    return parsed


def prometheus_delta(
    before: Mapping[str, Mapping[tuple[tuple[str, str], ...], float]],
    after: Mapping[str, Mapping[tuple[tuple[str, str], ...], float]],
) -> dict[str, dict[tuple[tuple[str, str], ...], float]]:
    result: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
    for name, samples in after.items():
        for labels, value in samples.items():
            old = before.get(name, {}).get(labels, 0.0)
            delta = value - old
            if delta != 0:
                result.setdefault(name, {})[labels] = delta
    return result


def _sum_metric(
    metrics: Mapping[str, Mapping[tuple[tuple[str, str], ...], float]], name: str
) -> float:
    return sum(metrics.get(name, {}).values())


def _histogram_quantile(
    metrics: Mapping[str, Mapping[tuple[tuple[str, str], ...], float]],
    base_name: str,
    quantile: float,
) -> float | None:
    buckets: dict[float, float] = {}
    for labels, value in metrics.get(f"{base_name}_bucket", {}).items():
        label_map = dict(labels)
        if "le" not in label_map:
            continue
        buckets[float(label_map["le"])] = buckets.get(float(label_map["le"]), 0.0) + value
    if not buckets:
        return None
    total = max(buckets.values())
    if total <= 0:
        return None
    target = total * quantile
    previous_bound = 0.0
    previous_count = 0.0
    for bound in sorted(buckets):
        count = buckets[bound]
        if count >= target:
            if math.isinf(bound) or count <= previous_count:
                return previous_bound
            fraction = (target - previous_count) / (count - previous_count)
            return previous_bound + (bound - previous_bound) * fraction
        previous_bound = bound
        previous_count = count
    return None


def summarize_vllm_metrics(
    delta: Mapping[str, Mapping[tuple[tuple[str, str], ...], float]],
) -> dict[str, Any]:
    """Summarize vLLM-owned histograms and speculative counters."""

    aliases = {
        "ttft": ("vllm:time_to_first_token_seconds",),
        "itl": ("vllm:time_per_output_token_seconds",),
        "e2e": ("vllm:e2e_request_latency_seconds",),
    }
    result: dict[str, Any] = {"measurement_side": "server_prometheus"}
    for label, candidates in aliases.items():
        base = next(
            (
                candidate
                for candidate in candidates
                if f"{candidate}_count" in delta or f"{candidate}_bucket" in delta
            ),
            candidates[0],
        )
        count = _sum_metric(delta, f"{base}_count")
        total = _sum_metric(delta, f"{base}_sum")
        result[label] = {
            "count": int(round(count)),
            "mean_s": total / count if count > 0 else None,
            "p50_s": _histogram_quantile(delta, base, 0.50),
            "p95_s": _histogram_quantile(delta, base, 0.95),
        }
    accepted = _sum_metric(delta, "vllm:spec_decode_num_accepted_tokens_total")
    drafted = _sum_metric(delta, "vllm:spec_decode_num_draft_tokens_total")
    drafts = _sum_metric(delta, "vllm:spec_decode_num_drafts_total")
    result["speculative"] = {
        "accepted_tokens": int(round(accepted)),
        "draft_tokens": int(round(drafted)),
        "drafts": int(round(drafts)),
        "acceptance_rate": accepted / drafted if drafted > 0 else None,
        "mean_acceptance_length": 1 + accepted / drafts if drafts > 0 else None,
    }
    return result
