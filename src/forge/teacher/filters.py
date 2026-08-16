"""Deterministic Phase 2 deduplication, contamination, and DPO perturbations."""

from __future__ import annotations

import copy
import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import duckdb

from forge.verify.schema import is_schema_valid
from forge.verify.verifier import ScoreBreakdown, score

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def normalized_tokens(value: object) -> tuple[str, ...]:
    """Tokenize with the normalization used by both contamination sides."""

    return tuple(token.casefold() for token in _TOKEN.findall(str(value or "")))


def token_ngrams(value: object, size: int) -> tuple[tuple[str, ...], ...]:
    """Return unique, sorted normalized token n-grams."""

    if size < 1:
        raise ValueError("token n-gram size must be positive")
    tokens = normalized_tokens(value)
    if len(tokens) < size:
        return ()
    return tuple(sorted({tokens[index : index + size] for index in range(len(tokens) - size + 1)}))


def _shingle_bytes(value: object, size: int) -> tuple[bytes, ...]:
    shingles = token_ngrams(value, size)
    if not shingles:
        fallback = " ".join(normalized_tokens(value)).encode()
        return (fallback,)
    return tuple(" ".join(shingle).encode() for shingle in shingles)


def minhash_signature(value: object, *, token_ngram: int, permutations: int) -> tuple[int, ...]:
    """Compute a stable MinHash signature without process-randomized hashes."""

    if permutations < 1:
        raise ValueError("MinHash permutations must be positive")
    shingles = _shingle_bytes(value, token_ngram)
    signature: list[int] = []
    for permutation in range(permutations):
        salt = permutation.to_bytes(4, "big")
        signature.append(
            min(
                int.from_bytes(hashlib.blake2b(salt + shingle, digest_size=8).digest(), "big")
                for shingle in shingles
            )
        )
    return tuple(signature)


def minhash_deduplicate(
    records: list[dict[str, Any]],
    *,
    token_ngram: int,
    permutations: int,
    bands: int,
    similarity_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the first deterministic record from each MinHash near-duplicate cluster."""

    if permutations % bands:
        raise ValueError("MinHash permutations must divide evenly into bands")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("MinHash similarity threshold must be in [0, 1]")
    rows_per_band = permutations // bands
    kept: list[dict[str, Any]] = []
    signatures: list[tuple[int, ...]] = []
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    exact: dict[tuple[str, ...], int] = {}
    duplicates: list[dict[str, Any]] = []

    for record in records:
        complaint_id = int(record["complaint_id"])
        narrative = record["model_input"]["narrative"]
        normalized = normalized_tokens(narrative)
        if normalized in exact:
            duplicate_index = exact[normalized]
            duplicates.append(
                {
                    "complaint_id": complaint_id,
                    "duplicate_of": int(kept[duplicate_index]["complaint_id"]),
                    "estimated_similarity": 1.0,
                    "reason": "exact_normalized_text",
                }
            )
            continue

        signature = minhash_signature(
            narrative,
            token_ngram=token_ngram,
            permutations=permutations,
        )
        candidates: set[int] = set()
        for band in range(bands):
            start = band * rows_per_band
            key = (band, signature[start : start + rows_per_band])
            candidates.update(buckets.get(key, ()))
        duplicate_index: int | None = None
        duplicate_similarity = 0.0
        for candidate in sorted(candidates):
            similarity = (
                sum(
                    left == right
                    for left, right in zip(signature, signatures[candidate], strict=True)
                )
                / permutations
            )
            if similarity >= similarity_threshold:
                duplicate_index = candidate
                duplicate_similarity = similarity
                break
        if duplicate_index is not None:
            duplicates.append(
                {
                    "complaint_id": complaint_id,
                    "duplicate_of": int(kept[duplicate_index]["complaint_id"]),
                    "estimated_similarity": duplicate_similarity,
                    "reason": "minhash_near_duplicate",
                }
            )
            continue

        kept_index = len(kept)
        kept.append(record)
        signatures.append(signature)
        exact[normalized] = kept_index
        for band in range(bands):
            start = band * rows_per_band
            key = (band, signature[start : start + rows_per_band])
            buckets[key].append(kept_index)
    return kept, duplicates


def _ngram_digest(ngram: tuple[str, ...]) -> str:
    return hashlib.sha256(" ".join(ngram).encode()).hexdigest()


def contamination_audit(
    records: list[dict[str, Any]],
    *,
    test_paths: Mapping[str, Path],
    token_ngram: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Quarantine any TRAIN record sharing an exact long n-gram with either TEST split."""

    index: dict[str, set[int]] = defaultdict(set)
    by_id = {int(record["complaint_id"]): record for record in records}
    for complaint_id, record in by_id.items():
        for ngram in token_ngrams(record["model_input"]["narrative"], token_ngram):
            index[_ngram_digest(ngram)].add(complaint_id)

    quarantined: dict[int, dict[str, Any]] = {}
    scanned: dict[str, int] = {}
    con = duckdb.connect()
    try:
        for split, path in test_paths.items():
            cursor = con.execute(
                "SELECT complaint_id, narrative FROM read_parquet(?) ORDER BY complaint_id",
                [str(path)],
            )
            count = 0
            while True:
                rows = cursor.fetchmany(1000)
                if not rows:
                    break
                count += len(rows)
                for test_id, narrative in rows:
                    digests = {
                        _ngram_digest(ngram) for ngram in token_ngrams(narrative, token_ngram)
                    }
                    for digest in sorted(digests & index.keys()):
                        for complaint_id in sorted(index[digest]):
                            if complaint_id in quarantined:
                                continue
                            quarantined[complaint_id] = {
                                "complaint_id": complaint_id,
                                "test_split": split,
                                "test_complaint_id": int(test_id),
                                "token_ngram": token_ngram,
                                "ngram_sha256": digest,
                                "policy": "any exact normalized n-gram overlap",
                            }
            scanned[split] = count
    finally:
        con.close()

    clean = [record for record in records if int(record["complaint_id"]) not in quarantined]
    quarantine = [quarantined[key] for key in sorted(quarantined)]
    return clean, quarantine, scanned


def breakdown_dict(item: ScoreBreakdown) -> dict[str, Any]:
    """Serialize the stable scorer-v2 audit surface."""

    return {
        "scorer_version": item.scorer_version,
        "json_valid": item.json_valid,
        "schema_valid": item.schema_valid,
        "decision_checks": item.decision_checks,
        "secondary_metrics": item.secondary_metrics,
        "task_success": item.task_success,
        "reward": item.reward,
        "errors": list(item.errors),
    }


def _alternative_tool_call(
    name: str,
    *,
    model_input: Mapping[str, Any],
    output: Mapping[str, Any],
) -> dict[str, Any] | None:
    company = output.get("company") or model_input.get("source_company")
    issue = output.get("issue") or model_input.get("source_issue") or "unspecified"
    if name == "request_more_info":
        return {
            "name": name,
            "arguments": {
                "missing_fields": ["details"],
                "question": "Please provide additional action-critical complaint details.",
            },
        }
    if name == "close_no_action":
        return {"name": name, "arguments": {"reason": "duplicate_or_spam"}}
    if name == "escalate_to_regulator":
        return {
            "name": name,
            "arguments": {
                "complaint_id": int(model_input["complaint_id"]),
                "reason": "Perturbed near-miss escalation reason.",
            },
        }
    if name == "start_refund_workflow" and company:
        return {
            "name": name,
            "arguments": {"company": company, "issue": issue, "evidence_required": True},
        }
    if name == "route_to_company" and company:
        return {"name": name, "arguments": {"company": company, "issue": issue}}
    return None


def _apply_perturbation(
    name: str,
    *,
    chosen: Mapping[str, Any],
    model_input: Mapping[str, Any],
) -> dict[str, Any] | None:
    rejected = copy.deepcopy(dict(chosen))
    if name == "urgency_cycle":
        values = ("low", "medium", "high")
        current = str(rejected["urgency"])
        rejected["urgency"] = values[(values.index(current) + 1) % len(values)]
        return rejected
    if name == "ambiguity_flip":
        flipped = not bool(rejected["ambiguity_flag"])
        rejected["ambiguity_flag"] = flipped
        target = "request_more_info" if flipped else "close_no_action"
        rejected["tool_call"] = _alternative_tool_call(
            target,
            model_input=model_input,
            output=rejected,
        )
        return rejected
    if name == "tool_choice_substitution":
        names = (
            "request_more_info",
            "close_no_action",
            "escalate_to_regulator",
            "start_refund_workflow",
            "route_to_company",
        )
        current = chosen["tool_call"]["name"]
        offset = int(hashlib.sha256(str(model_input["complaint_id"]).encode()).hexdigest(), 16)
        for index in range(len(names)):
            candidate = names[(offset + index) % len(names)]
            if candidate == current:
                continue
            tool_call = _alternative_tool_call(
                candidate,
                model_input=model_input,
                output=rejected,
            )
            if tool_call is None:
                continue
            rejected["tool_call"] = tool_call
            rejected["ambiguity_flag"] = candidate == "request_more_info"
            return rejected
    return None


def perturb_near_miss(
    *,
    chosen: Mapping[str, Any],
    model_input: Mapping[str, Any],
    rule_label: Mapping[str, Any],
    taxonomy: Iterable[str],
) -> tuple[dict[str, Any], str, ScoreBreakdown, ScoreBreakdown]:
    """Create a schema-valid rejected answer with a strictly lower verifier reward."""

    chosen_score = score({"label": rule_label}, chosen)
    names = tuple(taxonomy)
    if not names:
        raise ValueError("DPO perturbation taxonomy cannot be empty")
    start = int(
        hashlib.blake2b(str(model_input["complaint_id"]).encode(), digest_size=2).hexdigest(),
        16,
    )
    for offset in range(len(names)):
        name = names[(start + offset) % len(names)]
        rejected = _apply_perturbation(name, chosen=chosen, model_input=model_input)
        if rejected is None or not is_schema_valid(rejected):
            continue
        rejected_score = score({"label": rule_label}, rejected)
        if rejected_score.reward < chosen_score.reward:
            return rejected, name, chosen_score, rejected_score
    raise RuntimeError("perturbation taxonomy could not produce a lower-scoring near miss")
