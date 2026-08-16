"""Receipt-backed Phase 2 teacher generation and corpus materialization."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from forge.data.ingest import sha256_file
from forge.data.input_contract import INPUT_CONTRACT_VERSION, build_model_input
from forge.data.spot_label import (
    OPENROUTER_URL,
    _content_text,
    _load_key,
    _parse_teacher_content,
    _reported_cost,
)
from forge.teacher.filters import (
    breakdown_dict,
    contamination_audit,
    minhash_deduplicate,
    perturb_near_miss,
)
from forge.teacher.freeze import (
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    load_teacher_config,
    resolve_path,
    verify_frozen_source,
)
from forge.verify.schema import TASK_SCHEMA
from forge.verify.verifier import SCORER_VERSION, score

DEFAULT_API_ENV_PATH = REPO_ROOT / ".env"
RAW_LOG_NAME = "raw_teacher_generations.jsonl"
LEDGER_NAME = "generation_ledger.json"
FILTER_LOG_NAME = "filter_funnel.jsonl"
SFT_RULE_NAME = "sft_rule.jsonl"
SFT_DISTILLED_NAME = "sft_distilled.jsonl"
DPO_NAME = "dpo_pairs.jsonl"
MANIFEST_NAME = "manifest.json"


def _rank(seed: int, complaint_id: int) -> bytes:
    return hashlib.blake2b(f"{seed}:{complaint_id}".encode(), digest_size=16).digest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _new_temp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    return Path(name)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = _new_temp_path(path)
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, value: object) -> None:
    _write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    text = "".join(_canonical_json(record) + "\n" for record in records)
    _write_text_atomic(path, text)


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _select_rows(train_path: Path, *, limit: int, seed: int) -> list[dict[str, Any]]:
    con = duckdb.connect()
    try:
        ids = [
            int(row[0])
            for row in con.execute(
                "SELECT complaint_id FROM read_parquet(?)", [str(train_path)]
            ).fetchall()
        ]
        selected_ids = sorted(ids, key=lambda value: (_rank(seed, value), value))[:limit]
        placeholders = ",".join("?" for _ in selected_ids)
        rows = con.execute(
            f"""
            SELECT complaint_id, narrative, source_product, source_issue,
                   source_company, label_json
            FROM read_parquet(?)
            WHERE complaint_id IN ({placeholders})
            """,
            [str(train_path), *selected_ids],
        ).fetchall()
    finally:
        con.close()
    by_id = {
        int(row[0]): {
            "complaint_id": int(row[0]),
            "narrative": row[1],
            "source_product": row[2],
            "source_issue": row[3],
            "source_company": row[4],
            "rule_label": json.loads(row[5]),
        }
        for row in rows
    }
    if len(by_id) != limit:
        raise RuntimeError(f"selected {limit} TRAIN ids but loaded {len(by_id)} rows")
    return [by_id[complaint_id] for complaint_id in selected_ids]


def _request(
    *,
    api_key: str,
    model: str,
    prompt: str,
    model_input: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": _canonical_json(model_input)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "phase2_triage_teacher",
                "strict": True,
                "schema": TASK_SCHEMA,
            },
        },
        "usage": {"include": True},
    }
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "frontier-forge Phase 2 teacher factory",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _call_with_retry(
    *, max_retries: int, **kwargs: Any
) -> tuple[dict[str, Any] | None, str | None]:
    last_error: str | None = None
    for attempt in range(max_retries):
        try:
            return _request(**kwargs), None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except (TimeoutError, urllib.error.URLError, http.client.IncompleteRead) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < max_retries:
            time.sleep(0.5 * (2**attempt))
    return None, last_error


def _mock_output(row: Mapping[str, Any], sequence: int) -> object:
    output = json.loads(_canonical_json(row["rule_label"]))
    mode = sequence % 5
    if mode == 2:
        urgency = ("low", "medium", "high")
        output["urgency"] = urgency[(urgency.index(output["urgency"]) + 1) % len(urgency)]
    elif mode == 3:
        return "mock response that is intentionally not JSON"
    return output


def _base_record(
    *,
    row: Mapping[str, Any],
    sequence: int,
    attempt: int,
    fingerprint: str,
    model: str,
    prompt_sha256: str,
) -> dict[str, Any]:
    model_input = build_model_input(row)
    return {
        "sequence": sequence,
        "attempt": attempt,
        "run_fingerprint": fingerprint,
        "complaint_id": int(row["complaint_id"]),
        "teacher_model_id": model,
        "prompt_sha256": prompt_sha256,
        "input_contract_version": INPUT_CONTRACT_VERSION,
        "scorer_version": SCORER_VERSION,
        "input_sha256": _canonical_hash(model_input),
        "model_input": model_input,
        "rule_label": row["rule_label"],
    }


def _score_record(record: dict[str, Any], parsed_output: object) -> None:
    item = score({"label": record["rule_label"]}, parsed_output)
    record["parsed_output"] = parsed_output
    record["score"] = breakdown_dict(item)


def _mock_record(
    *,
    row: Mapping[str, Any],
    sequence: int,
    fingerprint: str,
    model: str,
    prompt_sha256: str,
) -> dict[str, Any]:
    record = _base_record(
        row=row,
        sequence=sequence,
        attempt=1,
        fingerprint=fingerprint,
        model=model,
        prompt_sha256=prompt_sha256,
    )
    output = _mock_output(row, sequence)
    raw_content = _canonical_json(output) if not isinstance(output, str) else output
    record.update(
        {
            "status": "ok",
            "provider": "local-mock",
            "response_id": f"mock-{row['complaint_id']}",
            "raw_response": {
                "model": model,
                "choices": [{"message": {"content": raw_content}}],
                "usage": {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
            },
            "raw_content": raw_content,
            "usage": {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0},
            "reported_cost_usd": 0.0,
        }
    )
    _score_record(record, output)
    return record


def _live_record(
    *,
    row: Mapping[str, Any],
    sequence: int,
    attempt: int,
    fingerprint: str,
    model: str,
    prompt: str,
    prompt_sha256: str,
    api_key: str,
    teacher_config: Mapping[str, Any],
) -> dict[str, Any]:
    record = _base_record(
        row=row,
        sequence=sequence,
        attempt=attempt,
        fingerprint=fingerprint,
        model=model,
        prompt_sha256=prompt_sha256,
    )
    response, error = _call_with_retry(
        max_retries=int(teacher_config["max_retries"]),
        api_key=api_key,
        model=model,
        prompt=prompt,
        model_input=record["model_input"],
        temperature=float(teacher_config["temperature"]),
        max_tokens=int(teacher_config["max_tokens"]),
        timeout_seconds=float(teacher_config["timeout_seconds"]),
    )
    if response is None:
        record.update(
            {
                "status": "request_failed",
                "error": error,
                "reported_cost_usd": 0.0,
            }
        )
        return record

    content = response.get("choices", [{}])[0].get("message", {}).get("content")
    try:
        raw_content = _content_text(content)
    except ValueError:
        raw_content = str(content)
    try:
        parsed_output = _parse_teacher_content(raw_content)
    except (json.JSONDecodeError, ValueError):
        parsed_output = raw_content
    record.update(
        {
            "status": "ok",
            "response_id": response.get("id"),
            "provider": response.get("provider"),
            "raw_response": response,
            "raw_content": raw_content,
            "usage": response.get("usage"),
            "reported_cost_usd": _reported_cost(response),
        }
    )
    _score_record(record, parsed_output)
    return record


def _load_raw_records(path: Path, fingerprint: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if any(record.get("run_fingerprint") != fingerprint for record in records):
        raise RuntimeError("existing Phase 2 raw log belongs to another run fingerprint")
    attempts = [(int(record["complaint_id"]), int(record["attempt"])) for record in records]
    if len(attempts) != len(set(attempts)):
        raise RuntimeError("duplicate complaint_id/attempt pair in Phase 2 raw log")
    return records


def _latest_records(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for record in records:
        complaint_id = int(record["complaint_id"])
        if complaint_id not in latest or int(record["attempt"]) > int(
            latest[complaint_id]["attempt"]
        ):
            latest[complaint_id] = record
    return latest


def _successful_records(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    successful: dict[int, dict[str, Any]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        complaint_id = int(record["complaint_id"])
        if complaint_id not in successful or int(record["attempt"]) > int(
            successful[complaint_id]["attempt"]
        ):
            successful[complaint_id] = record
    return successful


def _generation_ledger(
    *,
    fingerprint: str,
    mode: str,
    model: str,
    selected: list[dict[str, Any]],
    records: list[dict[str, Any]],
    raw_path: Path,
    budget: float,
    prompt_path: Path,
    prompt_sha256: str,
    started_at: str,
    status: str,
) -> dict[str, Any]:
    successful = _successful_records(records)
    latest = _latest_records(records)
    costs = [
        float(record["reported_cost_usd"])
        for record in successful.values()
        if record.get("reported_cost_usd") is not None
    ]
    missing_cost = sorted(
        complaint_id
        for complaint_id, record in successful.items()
        if record.get("reported_cost_usd") is None
    )
    return {
        "version": 1,
        "status": status,
        "mode": mode,
        "run_fingerprint": fingerprint,
        "started_at": started_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "teacher_model_id": model,
        "selected_rows": len(selected),
        "selected_complaint_ids_sha256": _canonical_hash(
            [int(row["complaint_id"]) for row in selected]
        ),
        "attempt_records": len(records),
        "latest_records": len(latest),
        "successful_responses": len(successful),
        "failed_latest_responses": sum(record.get("status") != "ok" for record in latest.values()),
        "reported_api_usd": sum(costs),
        "missing_cost_complaint_ids": missing_cost,
        "budget_usd": budget,
        "within_budget": not missing_cost and sum(costs) <= budget,
        "prompt_path": str(prompt_path.relative_to(REPO_ROOT)),
        "prompt_sha256": prompt_sha256,
        "raw_log_path": str(raw_path),
        "raw_log_sha256": sha256_file(raw_path) if raw_path.is_file() else None,
    }


def _execute_generation(
    *,
    selected: list[dict[str, Any]],
    output_dir: Path,
    config: Mapping[str, Any],
    fingerprint: str,
    prompt: str,
    prompt_path: Path,
    smoke: bool,
    api_env_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    teacher = config["teacher"]
    model = str(teacher["mock_model"] if smoke else teacher["model"])
    prompt_sha256 = sha256_file(prompt_path)
    raw_path = output_dir / RAW_LOG_NAME
    ledger_path = output_dir / LEDGER_NAME
    records = _load_raw_records(raw_path, fingerprint)
    successful = _successful_records(records)
    attempt_counts = Counter(int(record["complaint_id"]) for record in records)
    budget = 0.0 if smoke else float(teacher["max_budget_usd"])
    started_at = (
        json.loads(ledger_path.read_text()).get("started_at")
        if ledger_path.exists()
        else datetime.now(UTC).isoformat()
    )

    if smoke:
        new_records = [
            _mock_record(
                row=row,
                sequence=sequence,
                fingerprint=fingerprint,
                model=model,
                prompt_sha256=prompt_sha256,
            )
            for sequence, row in enumerate(selected)
            if int(row["complaint_id"]) not in successful
        ]
        _append_jsonl(raw_path, new_records)
        records.extend(new_records)
    else:
        api_key = _load_key(api_env_path)
        concurrency = int(teacher["concurrency"])
        planning_cost = float(teacher["planning_cost_per_call_usd"])
        missing = [
            (sequence, row)
            for sequence, row in enumerate(selected)
            if int(row["complaint_id"]) not in successful
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            for start in range(0, len(missing), concurrency):
                batch = missing[start : start + concurrency]
                current_cost = sum(
                    float(record.get("reported_cost_usd") or 0.0)
                    for record in _successful_records(records).values()
                )
                if current_cost + len(batch) * planning_cost > budget:
                    raise RuntimeError(
                        "conservative budget guard stopped teacher calls before the configured cap"
                    )
                futures = [
                    executor.submit(
                        _live_record,
                        row=row,
                        sequence=sequence,
                        attempt=attempt_counts[int(row["complaint_id"])] + 1,
                        fingerprint=fingerprint,
                        model=model,
                        prompt=prompt,
                        prompt_sha256=prompt_sha256,
                        api_key=api_key,
                        teacher_config=teacher,
                    )
                    for sequence, row in batch
                ]
                batch_records = [future.result() for future in futures]
                batch_records.sort(key=lambda record: int(record["sequence"]))
                _append_jsonl(raw_path, batch_records)
                records.extend(batch_records)
                for record in batch_records:
                    attempt_counts[int(record["complaint_id"])] += 1
                current_cost = sum(
                    float(record.get("reported_cost_usd") or 0.0)
                    for record in _successful_records(records).values()
                )
                if current_cost > budget:
                    raise RuntimeError(f"teacher API cost ${current_cost:.6f} exceeded cap")
                if len(records) % 160 < concurrency:
                    print(
                        f"teacher progress: successful={len(_successful_records(records))}/"
                        f"{len(selected)} api_usd={current_cost:.6f}",
                        flush=True,
                    )
                checkpoint = _generation_ledger(
                    fingerprint=fingerprint,
                    mode="live",
                    model=model,
                    selected=selected,
                    records=records,
                    raw_path=raw_path,
                    budget=budget,
                    prompt_path=prompt_path,
                    prompt_sha256=prompt_sha256,
                    started_at=str(started_at),
                    status="in_progress",
                )
                _write_json_atomic(ledger_path, checkpoint)

    successful = _successful_records(records)
    missing_ids = [
        int(row["complaint_id"]) for row in selected if int(row["complaint_id"]) not in successful
    ]
    status = "complete" if not missing_ids else "incomplete"
    ledger = _generation_ledger(
        fingerprint=fingerprint,
        mode="smoke" if smoke else "live",
        model=model,
        selected=selected,
        records=records,
        raw_path=raw_path,
        budget=budget,
        prompt_path=prompt_path,
        prompt_sha256=prompt_sha256,
        started_at=str(started_at),
        status=status,
    )
    ledger["finished_at"] = datetime.now(UTC).isoformat() if status == "complete" else None
    ledger["missing_complaint_ids"] = missing_ids
    _write_json_atomic(ledger_path, ledger)
    if status != "complete" or ledger["within_budget"] is not True:
        raise RuntimeError(
            f"teacher generation incomplete: missing={len(missing_ids)} "
            f"missing_cost={len(ledger['missing_cost_complaint_ids'])}"
        )
    return records, ledger


def _funnel_stage(name: str, count: int, previous: int, initial: int) -> dict[str, Any]:
    return {
        "stage": name,
        "count": count,
        "retention_from_previous": count / previous if previous else 0.0,
        "retention_from_selected": count / initial if initial else 0.0,
    }


def _teacher_disagreements(records: list[dict[str, Any]]) -> dict[str, Any]:
    schema_valid = [record for record in records if record["score"]["schema_valid"]]
    total = len(schema_valid)
    decision_names = (
        "urgency",
        "ambiguity_flag",
        "tool_choice",
        "tool_arguments_structural",
    )
    secondary_names = (
        "product_match",
        "issue_normalized_match",
        "company_normalized_match",
        "tool_arguments_semantic_valid",
    )
    decision = {
        name: {
            "matches": sum(record["score"]["decision_checks"][name] for record in schema_valid),
            "disagreements": sum(
                not record["score"]["decision_checks"][name] for record in schema_valid
            ),
        }
        for name in decision_names
    }
    secondary = {
        name: {
            "matches": sum(record["score"]["secondary_metrics"][name] for record in schema_valid),
            "disagreements": sum(
                not record["score"]["secondary_metrics"][name] for record in schema_valid
            ),
        }
        for name in secondary_names
    }
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for record in schema_valid:
        rule_tool = str(record["rule_label"]["tool_call"]["name"])
        predicted = record["parsed_output"]
        teacher_tool = (
            str(predicted.get("tool_call", {}).get("name"))
            if isinstance(predicted, dict)
            else "invalid"
        )
        matrix[rule_tool][teacher_tool] += 1
    return {
        "denominator": total,
        "population": "schema-valid teacher outputs before verifier rejection",
        "decision_fields": decision,
        "secondary_fields": secondary,
        "tool_choice_matrix": {
            expected: dict(sorted(observed.items()))
            for expected, observed in sorted(matrix.items())
        },
    }


def _messages(
    prompt: str,
    model_input: Mapping[str, Any],
    output: Mapping[str, Any],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": _canonical_json(model_input)},
        {"role": "assistant", "content": _canonical_json(output)},
    ]


def _artifact_entry(path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "rows": rows,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _data_card(manifest: Mapping[str, Any]) -> str:
    artifacts = manifest["artifacts"]
    cost = manifest["cost"]
    contamination = manifest["contamination"]
    disagreement = manifest["teacher_vs_rule_disagreement"]
    lines = [
        "# Phase 2 teacher-data card",
        "",
        f"Status: **{str(manifest['status']).upper()}**.",
        "",
        "## Frozen source",
        "",
        f"- Label-rules version: {manifest['source']['label_rules_version']}",
        f"- Dataset hash: `{manifest['source']['dataset_hash']}`",
        f"- Label-rules SHA-256: `{manifest['source']['label_rules_sha256']}`",
        f"- TRAIN payload SHA-256: `{manifest['source']['train_payload_sha256']}`",
        (
            "- TEST-IID and TEST-DRIFT were read only by the contamination auditor "
            "and were never sent to the teacher."
        ),
        "",
        "## Filter funnel",
        "",
        "| Stage | Rows | Retained from previous | Retained from selected |",
        "|---|---:|---:|---:|",
    ]
    for stage in manifest["filter_funnel"]:
        lines.append(
            f"| {stage['stage']} | {stage['count']} | "
            f"{stage['retention_from_previous']:.1%} | "
            f"{stage['retention_from_selected']:.1%} |"
        )
    lines.extend(
        [
            "",
            (
                "Verifier rejection requires schema validity, scorer-v2 task success, "
                "and semantically meaningful tool arguments. MinHash then removes "
                "near-duplicate TRAIN narratives. The contamination stage quarantines "
                "any survivor with an exact normalized 13-token n-gram found in either "
                "frozen TEST split."
            ),
            "",
            "## Materialized corpora",
            "",
            "| Artifact | Rows | SHA-256 |",
            "|---|---:|---|",
            (
                f"| Rule-label SFT | {artifacts['sft_rule']['rows']} | "
                f"`{artifacts['sft_rule']['sha256']}` |"
            ),
            (
                f"| Distilled SFT | {artifacts['sft_distilled']['rows']} | "
                f"`{artifacts['sft_distilled']['sha256']}` |"
            ),
            (
                f"| DPO pairs | {artifacts['dpo_pairs']['rows']} | "
                f"`{artifacts['dpo_pairs']['sha256']}` |"
            ),
            "",
            (
                "Rule and distilled SFT coverage is identical over "
                f"{manifest['coverage']['shared_complaint_ids']} complaint IDs; their "
                "ordered complaint-ID SHA-256 is "
                f"`{manifest['coverage']['complaint_ids_sha256']}`. DPO chosen responses "
                "are surviving high-scoring teacher outputs. Rejected responses are "
                "deterministic lower-scoring near misses from the documented "
                "perturbation taxonomy."
            ),
            "",
            "## Contamination",
            "",
            f"- Quarantined TRAIN samples: {contamination['quarantined_rows']}",
            f"- TEST-IID rows scanned: {contamination['test_rows_scanned']['test_iid']}",
            f"- TEST-DRIFT rows scanned: {contamination['test_rows_scanned']['test_drift']}",
            f"- Quarantine receipt: `{contamination['quarantine_sha256']}`",
            "",
            (
                "A zero quarantine count is a clean audit under this exact 13-token "
                "policy. A nonzero count is also gate-complete because every hit is "
                "excluded and preserved in the committed quarantine receipt."
            ),
            "",
            "## Teacher versus frozen rule policy",
            "",
            (
                f"Denominator: {disagreement['denominator']} schema-valid teacher "
                "outputs before verifier rejection."
            ),
            "",
            "| Field | Matches | Disagreements | Match rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, item in disagreement["decision_fields"].items():
        denominator = disagreement["denominator"]
        rate = item["matches"] / denominator if denominator else 0.0
        lines.append(f"| {name} | {item['matches']} | {item['disagreements']} | {rate:.1%} |")
    urgency = disagreement["decision_fields"]["urgency"]
    urgency_rate = (
        urgency["matches"] / disagreement["denominator"] if disagreement["denominator"] else 0
    )
    lines.extend(
        [
            "",
            (
                f"Urgency agreement is {urgency_rate:.1%}. This measures teacher "
                "compliance with the frozen keyword policy, not human semantic "
                "correctness. The Phase 1.2 review established known rule false "
                "negatives, so downstream reports must describe urgency ground truth "
                "as rule-policy ground truth."
            ),
            "",
            "## Cost ledger",
            "",
            f"- Teacher model: `{manifest['generation']['teacher_model_id']}`",
            f"- Provider-reported Phase 2 API cost: **${cost['api_usd']:.6f}**",
            f"- Run-specific hard cap: ${cost['run_cap_usd']:.2f}",
            (
                "- Project teacher-API envelope: "
                f"${cost['project_envelope_usd'][0]:.0f}–"
                f"${cost['project_envelope_usd'][1]:.0f}"
            ),
            f"- Within run cap: {'yes' if cost['within_run_cap'] else 'no'}",
            "- GPU cost: $0.00; Phase 2 is local plus API only.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "make teacher-data SMOKE=1",
            "make teacher-audit SMOKE=1",
            "make teacher-audit",
            "```",
            "",
            (
                "The live raw response log is receipt-backed and resumable. "
                "Re-auditing makes no network request."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _materialize(
    *,
    selected: list[dict[str, Any]],
    records: list[dict[str, Any]],
    ledger: Mapping[str, Any],
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    prompt: str,
    prompt_path: Path,
    frozen: Mapping[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    successful = _successful_records(records)
    latest = [successful[int(row["complaint_id"])] for row in selected]
    initial = len(selected)
    schema_valid = [record for record in latest if record["score"]["schema_valid"]]
    verifier_valid = [
        record
        for record in schema_valid
        if record["score"]["task_success"]
        and record["score"]["secondary_metrics"]["tool_arguments_semantic_valid"]
    ]
    filter_config = config["filter"]
    minhash = filter_config["minhash"]
    deduplicated, duplicate_rows = minhash_deduplicate(
        verifier_valid,
        token_ngram=int(minhash["token_ngram"]),
        permutations=int(minhash["permutations"]),
        bands=int(minhash["bands"]),
        similarity_threshold=float(minhash["similarity_threshold"]),
    )
    test_paths = {
        name: resolve_path(item["path"]) for name, item in config["source"]["test_splits"].items()
    }
    clean, quarantine, scanned = contamination_audit(
        deduplicated,
        test_paths=test_paths,
        token_ngram=int(filter_config["contamination"]["token_ngram"]),
    )

    funnel = [
        _funnel_stage("selected TRAIN inputs", initial, initial, initial),
        _funnel_stage("teacher response recorded", len(latest), initial, initial),
        _funnel_stage("schema valid", len(schema_valid), len(latest), initial),
        _funnel_stage("verifier accepted", len(verifier_valid), len(schema_valid), initial),
        _funnel_stage("MinHash unique", len(deduplicated), len(verifier_valid), initial),
        _funnel_stage("contamination clean", len(clean), len(deduplicated), initial),
    ]
    clean_ids = [int(record["complaint_id"]) for record in clean]
    filter_log: list[dict[str, Any]] = []
    duplicate_by_id = {int(item["complaint_id"]): item for item in duplicate_rows}
    quarantine_by_id = {int(item["complaint_id"]): item for item in quarantine}
    verifier_ids = {int(record["complaint_id"]) for record in verifier_valid}
    schema_ids = {int(record["complaint_id"]) for record in schema_valid}
    clean_id_set = set(clean_ids)
    for record in latest:
        complaint_id = int(record["complaint_id"])
        if complaint_id not in schema_ids:
            outcome, reason = "rejected", "schema_invalid"
        elif complaint_id not in verifier_ids:
            outcome, reason = "rejected", "verifier_score"
        elif complaint_id in duplicate_by_id:
            outcome, reason = "rejected", "minhash_duplicate"
        elif complaint_id in quarantine_by_id:
            outcome, reason = "quarantined", "test_ngram_overlap"
        elif complaint_id in clean_id_set:
            outcome, reason = "accepted", None
        else:
            raise AssertionError(f"unclassified Phase 2 sample {complaint_id}")
        filter_log.append(
            {
                "complaint_id": complaint_id,
                "outcome": outcome,
                "reason": reason,
                "teacher_reward": record["score"]["reward"],
                "task_success": record["score"]["task_success"],
                "duplicate": duplicate_by_id.get(complaint_id),
                "contamination": quarantine_by_id.get(complaint_id),
            }
        )

    sft_rule: list[dict[str, Any]] = []
    sft_distilled: list[dict[str, Any]] = []
    dpo_pairs: list[dict[str, Any]] = []
    perturbation_counts: Counter[str] = Counter()
    taxonomy = tuple(config["dpo"]["perturbation_taxonomy"])
    for record in clean:
        complaint_id = int(record["complaint_id"])
        model_input = record["model_input"]
        rule_label = record["rule_label"]
        teacher_output = record["parsed_output"]
        if not isinstance(teacher_output, dict):
            raise AssertionError("accepted teacher output must be an object")
        common = {
            "complaint_id": complaint_id,
            "input_contract_version": INPUT_CONTRACT_VERSION,
            "model_input": model_input,
        }
        sft_rule.append(
            {
                **common,
                "messages": _messages(prompt, model_input, rule_label),
                "target": rule_label,
                "target_source": "frozen_label_rules_v3",
                "dataset_hash": frozen["dataset_hash"],
                "label_rules_sha256": frozen["label_rules_sha256"],
            }
        )
        sft_distilled.append(
            {
                **common,
                "messages": _messages(prompt, model_input, teacher_output),
                "target": teacher_output,
                "target_source": "filtered_teacher",
                "teacher_model_id": record["teacher_model_id"],
                "prompt_sha256": record["prompt_sha256"],
                "raw_response": record["raw_response"],
                "teacher_score": record["score"],
            }
        )
        rejected, perturbation, chosen_score, rejected_score = perturb_near_miss(
            chosen=teacher_output,
            model_input=model_input,
            rule_label=rule_label,
            taxonomy=taxonomy,
        )
        perturbation_counts[perturbation] += 1
        dpo_pairs.append(
            {
                **common,
                "prompt": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": _canonical_json(model_input)},
                ],
                "chosen": _canonical_json(teacher_output),
                "rejected": _canonical_json(rejected),
                "chosen_score": breakdown_dict(chosen_score),
                "rejected_score": breakdown_dict(rejected_score),
                "rejected_source": config["dpo"]["rejected_source"],
                "perturbation_type": perturbation,
                "teacher_model_id": record["teacher_model_id"],
                "prompt_sha256": record["prompt_sha256"],
                "raw_response": record["raw_response"],
            }
        )

    rule_ids = [int(record["complaint_id"]) for record in sft_rule]
    distilled_ids = [int(record["complaint_id"]) for record in sft_distilled]
    dpo_ids = [int(record["complaint_id"]) for record in dpo_pairs]
    if not (rule_ids == distilled_ids == dpo_ids):
        raise AssertionError("Phase 2 corpora do not have identical input coverage")

    filter_path = output_dir / FILTER_LOG_NAME
    rule_path = output_dir / SFT_RULE_NAME
    distilled_path = output_dir / SFT_DISTILLED_NAME
    dpo_path = output_dir / DPO_NAME
    _write_jsonl_atomic(filter_path, filter_log)
    _write_jsonl_atomic(rule_path, sft_rule)
    _write_jsonl_atomic(distilled_path, sft_distilled)
    _write_jsonl_atomic(dpo_path, dpo_pairs)

    outputs = config["outputs"]
    if smoke:
        tracked_manifest_path = output_dir / "phase2_manifest.json"
        data_card_path = output_dir / "phase2_data_card.md"
        cost_ledger_path = output_dir / "phase2_cost_ledger.json"
        quarantine_path = output_dir / "phase2_contamination_quarantine.json"
    else:
        tracked_manifest_path = resolve_path(outputs["tracked_manifest"])
        data_card_path = resolve_path(outputs["data_card"])
        cost_ledger_path = resolve_path(outputs["cost_ledger"])
        quarantine_path = resolve_path(outputs["quarantine"])

    quarantine_receipt = {
        "version": 1,
        "dataset_hash": frozen["dataset_hash"],
        "token_ngram": int(filter_config["contamination"]["token_ngram"]),
        "policy": filter_config["contamination"]["policy"],
        "quarantined_rows": len(quarantine),
        "rows": quarantine,
    }
    _write_json_atomic(quarantine_path, quarantine_receipt)
    prompt_sha256 = sha256_file(prompt_path)
    cost_ledger = {
        "version": 1,
        "phase": 2,
        "teacher_model_id": ledger["teacher_model_id"],
        "provider_reported_api_usd": ledger["reported_api_usd"],
        "run_cap_usd": config["teacher"]["max_budget_usd"] if not smoke else 0.0,
        "project_teacher_api_envelope_usd": [20.0, 50.0],
        "within_run_cap": ledger["within_budget"],
        "missing_cost_complaint_ids": ledger["missing_cost_complaint_ids"],
        "gpu_type": "none",
        "gpu_hours": 0.0,
        "gpu_usd": 0.0,
        "raw_log_sha256": ledger["raw_log_sha256"],
    }
    _write_json_atomic(cost_ledger_path, cost_ledger)
    artifacts = {
        "raw_teacher_generations": _artifact_entry(output_dir / RAW_LOG_NAME, len(records)),
        "filter_funnel_log": _artifact_entry(filter_path, len(filter_log)),
        "sft_rule": _artifact_entry(rule_path, len(sft_rule)),
        "sft_distilled": _artifact_entry(distilled_path, len(sft_distilled)),
        "dpo_pairs": _artifact_entry(dpo_path, len(dpo_pairs)),
    }
    phase2_dataset_hash = _canonical_hash(
        {
            "source_dataset_hash": frozen["dataset_hash"],
            "config_sha256": sha256_file(config_path),
            "prompt_sha256": prompt_sha256,
            "artifact_sha256": {name: item["sha256"] for name, item in artifacts.items()},
        }
    )
    manifest = {
        "version": 1,
        "phase": 2,
        "status": "complete",
        "mode": "smoke" if smoke else "live",
        "phase2_dataset_hash": phase2_dataset_hash,
        "source": {
            "dataset_hash": frozen["dataset_hash"],
            "dataset_manifest_sha256": frozen["dataset_manifest_sha256"],
            "label_rules_version": frozen["label_rules_version"],
            "label_rules_sha256": frozen["label_rules_sha256"],
            "train_payload_sha256": frozen["splits"]["train"]["payload_sha256"],
            "input_contract_version": frozen["input_contract_version"],
            "scorer_version": frozen["scorer_version"],
        },
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(config_path),
        "generation": {
            "run_fingerprint": ledger["run_fingerprint"],
            "teacher_model_id": ledger["teacher_model_id"],
            "prompt_path": str(prompt_path.relative_to(REPO_ROOT)),
            "prompt_sha256": prompt_sha256,
            "selected_rows": initial,
            "selected_complaint_ids_sha256": ledger["selected_complaint_ids_sha256"],
            "attempt_records": ledger["attempt_records"],
            "started_at": ledger["started_at"],
            "finished_at": ledger["finished_at"],
            "raw_log_sha256": ledger["raw_log_sha256"],
        },
        "filter_funnel": funnel,
        "minhash": {
            **minhash,
            "duplicates_removed": len(duplicate_rows),
        },
        "contamination": {
            "token_ngram": filter_config["contamination"]["token_ngram"],
            "policy": filter_config["contamination"]["policy"],
            "test_rows_scanned": scanned,
            "quarantined_rows": len(quarantine),
            "quarantine_path": str(quarantine_path.relative_to(REPO_ROOT)),
            "quarantine_sha256": sha256_file(quarantine_path),
        },
        "coverage": {
            "shared_complaint_ids": len(rule_ids),
            "complaint_ids_sha256": _canonical_hash(rule_ids),
            "rule_equals_distilled_equals_dpo": True,
        },
        "dpo": {
            "pairs": len(dpo_pairs),
            "rejected_source": config["dpo"]["rejected_source"],
            "perturbation_taxonomy": list(taxonomy),
            "perturbation_counts": dict(sorted(perturbation_counts.items())),
        },
        "teacher_vs_rule_disagreement": _teacher_disagreements(latest),
        "cost": {
            "api_usd": ledger["reported_api_usd"],
            "run_cap_usd": cost_ledger["run_cap_usd"],
            "project_envelope_usd": cost_ledger["project_teacher_api_envelope_usd"],
            "within_run_cap": cost_ledger["within_run_cap"],
            "gpu_usd": 0.0,
        },
        "artifacts": artifacts,
        "audit_reproduction": {
            "command": "make teacher-audit" if not smoke else "make teacher-audit SMOKE=1",
            "network_calls": 0,
        },
    }
    local_manifest_path = output_dir / MANIFEST_NAME
    _write_text_atomic(data_card_path, _data_card(manifest))
    manifest["receipts"] = {
        "data_card_sha256": sha256_file(data_card_path),
        "cost_ledger_sha256": sha256_file(cost_ledger_path),
    }
    _write_json_atomic(local_manifest_path, manifest)
    _write_json_atomic(tracked_manifest_path, manifest)
    return manifest


def run_teacher_data(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    api_env_path: Path = DEFAULT_API_ENV_PATH,
    smoke: bool = False,
    live: bool = False,
    limit: int | None = None,
) -> tuple[dict[str, Any], bool]:
    """Generate teacher attempts and materialize all Phase 2 artifacts."""

    if smoke and live:
        raise ValueError("smoke mock mode and live API mode are mutually exclusive")
    config_path = Path(config_path)
    api_env_path = Path(api_env_path)
    config = load_teacher_config(config_path)
    frozen = verify_frozen_source(config_path)
    selection = config["selection"]
    configured_limit = int(
        selection["smoke_candidate_cap"] if smoke else selection["live_candidate_cap"]
    )
    selected_limit = configured_limit if limit is None else int(limit)
    if selected_limit < 1 or selected_limit > configured_limit:
        raise ValueError(f"requested limit must be between 1 and {configured_limit}")
    train_path = resolve_path(config["source"]["train"]["path"])
    prompt_path = resolve_path(config["teacher"]["prompt"])
    prompt = prompt_path.read_text()
    selected = _select_rows(
        train_path,
        limit=selected_limit,
        seed=int(selection["seed"]),
    )
    output_dir = resolve_path(
        config["outputs"]["smoke_dir"] if smoke else config["outputs"]["full_dir"]
    )
    mode = "smoke" if smoke else "live"
    fingerprint = _canonical_hash(
        {
            "mode": mode,
            "config_sha256": sha256_file(config_path),
            "dataset_hash": frozen["dataset_hash"],
            "train_payload_sha256": frozen["splits"]["train"]["payload_sha256"],
            "prompt_sha256": sha256_file(prompt_path),
            "teacher_model_id": config["teacher"]["mock_model" if smoke else "model"],
            "selection_seed": selection["seed"],
            "selected_ids": [int(row["complaint_id"]) for row in selected],
            "generator_sha256": sha256_file(Path(__file__)),
        }
    )
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        raw_path = output_dir / RAW_LOG_NAME
        if (
            manifest.get("status") == "complete"
            and manifest.get("mode") == mode
            and manifest.get("source", {}).get("dataset_hash") == frozen["dataset_hash"]
            and manifest.get("generation", {}).get("selected_complaint_ids_sha256")
            == _canonical_hash([int(row["complaint_id"]) for row in selected])
            and manifest.get("config_sha256") == sha256_file(config_path)
            and manifest.get("generation", {}).get("prompt_sha256") == sha256_file(prompt_path)
            and manifest.get("generation", {}).get("teacher_model_id")
            == config["teacher"]["mock_model" if smoke else "model"]
            and raw_path.is_file()
            and manifest.get("generation", {}).get("raw_log_sha256") == sha256_file(raw_path)
        ):
            return manifest, True
        raise RuntimeError("existing Phase 2 manifest does not match the requested frozen run")
    if not smoke and not live:
        return {
            "status": "dry-run",
            "mode": "live",
            "run_fingerprint": fingerprint,
            "selected_rows": selected_limit,
            "selected_complaint_ids_sha256": _canonical_hash(
                [int(row["complaint_id"]) for row in selected]
            ),
            "teacher_model_id": config["teacher"]["model"],
            "budget_usd": config["teacher"]["max_budget_usd"],
            "network_calls": 0,
        }, False

    records, ledger = _execute_generation(
        selected=selected,
        output_dir=output_dir,
        config=config,
        fingerprint=fingerprint,
        prompt=prompt,
        prompt_path=prompt_path,
        smoke=smoke,
        api_env_path=api_env_path,
    )
    manifest = _materialize(
        selected=selected,
        records=records,
        ledger=ledger,
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        prompt=prompt,
        prompt_path=prompt_path,
        frozen=frozen,
        smoke=smoke,
    )
    return manifest, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forge.teacher.generate")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--api-env", type=Path, default=DEFAULT_API_ENV_PATH)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    manifest, noop = run_teacher_data(
        config_path=args.config,
        api_env_path=args.api_env,
        smoke=args.smoke,
        live=args.live,
        limit=args.limit,
    )
    status = "frozen no-op" if noop else manifest["status"]
    if manifest["status"] == "dry-run":
        print(
            f"teacher data: dry-run; selected={manifest['selected_rows']}; "
            f"budget_usd={manifest['budget_usd']:.2f}; network_calls=0"
        )
    else:
        print(
            f"teacher data: {status}; mode={manifest['mode']}; "
            f"corpus_rows={manifest['coverage']['shared_complaint_ids']}; "
            f"api_usd={manifest['cost']['api_usd']:.6f}; "
            f"dataset_hash={manifest['phase2_dataset_hash']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
