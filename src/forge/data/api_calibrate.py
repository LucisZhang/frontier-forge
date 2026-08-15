"""Receipt-backed small-API stand-in calibration on frozen CAL only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import yaml

from forge.data.calibrate import CALIBRATION_SEED, _wilson
from forge.data.ingest import DEFAULT_CONFIG_PATH, load_data_source_config, sha256_file
from forge.data.splits import DEFAULT_OUTPUT_DIR
from forge.data.spot_label import (
    OPENROUTER_URL,
    _content_text,
    _load_key,
    _parse_teacher_content,
    _reported_cost,
)
from forge.verify.verifier import ScoreBreakdown, score

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAL_PATH = DEFAULT_OUTPUT_DIR / "cal.parquet"
DEFAULT_RULES_PATH = REPO_ROOT / "configs" / "label_rules.yaml"
DEFAULT_PROMPT_PATH = REPO_ROOT / "configs" / "teacher_prompts" / "phase1_api_calibration_v1.txt"
DEFAULT_RECEIPTS_PATH = REPO_ROOT / "results" / "phase1_api_calibration_receipts.jsonl"
DEFAULT_LEDGER_PATH = REPO_ROOT / "results" / "phase1_api_calibration_ledger.json"
DEFAULT_LIMIT = 25


def _rank(complaint_id: int) -> bytes:
    return hashlib.blake2b(f"{CALIBRATION_SEED}:{complaint_id}".encode(), digest_size=16).digest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _select_rows(cal_path: Path, limit: int) -> list[dict[str, Any]]:
    con = duckdb.connect()
    try:
        rows = con.execute(
            "SELECT complaint_id, narrative, label_json FROM read_parquet(?)",
            [str(cal_path)],
        ).fetchall()
    finally:
        con.close()
    selected = sorted(rows, key=lambda row: (_rank(int(row[0])), int(row[0])))[:limit]
    return [
        {
            "complaint_id": int(complaint_id),
            "narrative": narrative,
            "label": json.loads(label_json),
        }
        for complaint_id, narrative, label_json in selected
    ]


def _request(
    *,
    api_key: str,
    model: str,
    prompt: str,
    row: dict[str, Any],
    timeout_seconds: float = 90.0,
) -> dict[str, Any]:
    # Deliberately exclude source_product/source_issue/source_company: this is
    # zero-shot task calibration, not the separate teacher label audit.
    visible_input = {
        "complaint_id": row["complaint_id"],
        "narrative": row["narrative"],
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(visible_input, ensure_ascii=False, sort_keys=True),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "usage": {"include": True},
    }
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "frontier-forge Phase 1 API calibration",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _call_with_retry(**kwargs: Any) -> tuple[dict[str, Any] | None, str | None]:
    last_error: str | None = None
    for attempt in range(3):
        try:
            return _request(**kwargs), None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < 2:
            time.sleep(0.5 * (2**attempt))
    return None, last_error


def _breakdown_dict(item: ScoreBreakdown) -> dict[str, Any]:
    return {
        "json_valid": item.json_valid,
        "schema_valid": item.schema_valid,
        "field_matches": item.field_matches,
        "tool_choice_match": item.tool_choice_match,
        "tool_arguments_match": item.tool_arguments_match,
        "abstention_correct": item.abstention_correct,
        "task_success": item.task_success,
        "reward": item.reward,
        "errors": item.errors,
    }


def _write_receipts(path: Path, records: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _existing_records(path: Path, fingerprint: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if any(record.get("run_fingerprint") != fingerprint for record in records):
        raise RuntimeError("existing API calibration receipts belong to another run")
    ids = [record["complaint_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate complaint_id in API calibration receipts")
    return records


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [record["score"] for record in records if "score" in record]
    total = len(scored)

    def rate(key: str) -> float:
        return sum(bool(item[key]) for item in scored) / total if total else 0.0

    def field_rate(key: str) -> float:
        return sum(bool(item["field_matches"][key]) for item in scored) / total if total else 0.0

    successes = sum(bool(item["task_success"]) for item in scored)
    low, high = _wilson(successes, total)
    return {
        "samples": total,
        "task_success_count": successes,
        "task_success": successes / total if total else 0.0,
        "task_success_ci95_wilson": [low, high],
        "schema_valid": rate("schema_valid"),
        "product_match": field_rate("product"),
        "issue_match": field_rate("issue"),
        "company_match": field_rate("company"),
        "urgency_match": field_rate("urgency"),
        "ambiguity_flag_match": field_rate("ambiguity_flag"),
        "tool_choice_match": rate("tool_choice_match"),
        "tool_arguments_match": rate("tool_arguments_match"),
        "abstention_correct": rate("abstention_correct"),
        "mean_reward": sum(float(item["reward"]) for item in scored) / total if total else 0.0,
    }


def run_api_calibration(
    *,
    cal_path: Path = DEFAULT_CAL_PATH,
    data_config_path: Path = DEFAULT_CONFIG_PATH,
    rules_path: Path = DEFAULT_RULES_PATH,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    receipts_path: Path = DEFAULT_RECEIPTS_PATH,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    limit: int = DEFAULT_LIMIT,
    live: bool = False,
) -> tuple[dict[str, Any], bool]:
    cal_path = Path(cal_path)
    prompt_path = Path(prompt_path)
    receipts_path = Path(receipts_path)
    ledger_path = Path(ledger_path)
    if not cal_path.is_file():
        raise FileNotFoundError(f"CAL split is missing: {cal_path}")
    config = yaml.safe_load(Path(rules_path).read_text())["teacher_spot_labels"]
    model = str(config["model"])
    budget = float(config["max_budget_usd"])
    prompt = prompt_path.read_text()
    selected = _select_rows(cal_path, limit)
    selection_ids = [row["complaint_id"] for row in selected]
    fingerprint = _canonical_hash(
        {
            "cal_sha256": sha256_file(cal_path),
            "prompt_sha256": sha256_file(prompt_path),
            "runner_sha256": sha256_file(Path(__file__)),
            "model": model,
            "selection_ids": selection_ids,
            "visible_input_fields": ["complaint_id", "narrative"],
        }
    )
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger.get("run_fingerprint") == fingerprint and ledger.get("status") == "complete":
            return ledger, True
        raise RuntimeError("existing API calibration ledger is not this completed run")
    if not live:
        return {
            "status": "dry-run",
            "run_fingerprint": fingerprint,
            "model": model,
            "selected_complaint_ids": selection_ids,
            "network_calls": 0,
        }, False

    upstream = load_data_source_config(data_config_path).upstream_root
    api_key = _load_key(upstream / ".env")
    receipts_path.parent.mkdir(parents=True, exist_ok=True)
    records = _existing_records(receipts_path, fingerprint)
    completed_ids = {int(record["complaint_id"]) for record in records}
    total_cost = sum(float(record.get("reported_cost_usd") or 0.0) for record in records)
    started_at = datetime.now(UTC).isoformat()

    for row in selected:
        if row["complaint_id"] in completed_ids:
            continue
        if total_cost > budget:
            raise RuntimeError(f"API calibration cost {total_cost:.6f} exceeded cap")
        response, error = _call_with_retry(
            api_key=api_key,
            model=model,
            prompt=prompt,
            row=row,
        )
        record: dict[str, Any] = {
            "run_fingerprint": fingerprint,
            "complaint_id": row["complaint_id"],
            "model": model,
            "prompt_sha256": sha256_file(prompt_path),
            "input_sha256": _canonical_hash(
                {"complaint_id": row["complaint_id"], "narrative": row["narrative"]}
            ),
        }
        if response is None:
            record.update({"status": "request_failed", "error": error})
        else:
            raw_content = _content_text(
                response.get("choices", [{}])[0].get("message", {}).get("content")
            )
            try:
                output = _parse_teacher_content(raw_content)
            except json.JSONDecodeError:
                output = raw_content
            breakdown = score({"label": row["label"]}, output)
            cost = _reported_cost(response)
            record.update(
                {
                    "status": "ok",
                    "response_id": response.get("id"),
                    "provider": response.get("provider"),
                    "raw_content": raw_content,
                    "parsed_output": output,
                    "score": _breakdown_dict(breakdown),
                    "usage": response.get("usage"),
                    "reported_cost_usd": cost,
                }
            )
            if cost is not None:
                total_cost += cost
        records.append(record)
        _write_receipts(receipts_path, records)

    response_records = [record for record in records if record["status"] == "ok"]
    missing_cost = [
        record["complaint_id"]
        for record in response_records
        if record.get("reported_cost_usd") is None
    ]
    metrics = _metrics(response_records)
    ledger = {
        "version": 1,
        "status": (
            "complete"
            if len(response_records) == limit and len(records) == limit and not missing_cost
            else "incomplete"
        ),
        "run_fingerprint": fingerprint,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "model": model,
        "input_fields_visible_to_model": ["complaint_id", "narrative"],
        "cal_path": str(cal_path.resolve()),
        "cal_sha256": sha256_file(cal_path),
        "selection_seed": CALIBRATION_SEED,
        "selected_rows": limit,
        "calls_recorded": len(records),
        "metrics": metrics,
        "reported_api_usd": total_cost,
        "missing_cost_complaint_ids": missing_cost,
        "budget_usd": budget,
        "within_budget": not missing_cost and total_cost <= budget,
        "prompt_path": str(prompt_path.resolve()),
        "prompt_sha256": sha256_file(prompt_path),
        "receipts_path": str(receipts_path.resolve()),
        "receipts_sha256": sha256_file(receipts_path),
    }
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    if ledger["status"] != "complete" or not ledger["within_budget"]:
        raise RuntimeError("API calibration ledger is incomplete or outside budget")
    return ledger, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forge.data.api_calibrate")
    parser.add_argument("--cal", type=Path, default=DEFAULT_CAL_PATH)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS_PATH)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    ledger, noop = run_api_calibration(
        cal_path=args.cal,
        data_config_path=args.data_config,
        rules_path=args.rules,
        prompt_path=args.prompt,
        receipts_path=args.receipts,
        ledger_path=args.ledger,
        limit=args.limit,
        live=args.live,
    )
    status = "frozen no-op" if noop else ledger["status"]
    metrics = ledger.get("metrics", {})
    print(
        f"API calibration: {status}; calls={ledger.get('calls_recorded', 0)}; "
        f"task_success={metrics.get('task_success', 0.0):.3f}; "
        f"api_usd={ledger.get('reported_api_usd', 0.0):.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
