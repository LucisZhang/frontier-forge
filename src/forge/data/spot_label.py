"""Budget-capped OpenRouter spot labeling for ambiguous Phase-1 CAL rows."""

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

from forge.data.ingest import DEFAULT_CONFIG_PATH, load_data_source_config, sha256_file
from forge.data.splits import DEFAULT_OUTPUT_DIR
from forge.verify.schema import TASK_SCHEMA, is_schema_valid, schema_errors
from forge.verify.verifier import score

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAL_PATH = DEFAULT_OUTPUT_DIR / "cal.parquet"
DEFAULT_RULES_PATH = REPO_ROOT / "configs" / "label_rules.yaml"
DEFAULT_PROMPT_PATH = REPO_ROOT / "configs" / "teacher_prompts" / "phase1_spot_label_v1.txt"
DEFAULT_RECEIPTS_PATH = REPO_ROOT / "results" / "phase1_teacher_spot_labels.jsonl"
DEFAULT_LEDGER_PATH = REPO_ROOT / "results" / "phase1_teacher_spot_label_ledger.json"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SELECTION_SEED = 20260815
DEFAULT_LIMIT = 20


def _rank(complaint_id: int) -> bytes:
    return hashlib.blake2b(f"{SELECTION_SEED}:{complaint_id}".encode(), digest_size=16).digest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_key(upstream_env_path: Path) -> str:
    from_environment = os.environ.get("OPENROUTER_API_KEY")
    if from_environment:
        return from_environment
    if upstream_env_path.is_file():
        for raw_line in upstream_env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip('"').strip("'")
    raise RuntimeError(
        "OPENROUTER_API_KEY is absent from the environment and the configured nlp-eval-lab .env"
    )


def _select_rows(cal_path: Path, limit: int) -> list[dict[str, Any]]:
    con = duckdb.connect()
    try:
        rows = con.execute(
            """
            SELECT complaint_id, narrative, source_product, source_issue,
                   source_company, label_json
            FROM read_parquet(?)
            WHERE CAST(json_extract(label_json, '$.ambiguity_flag') AS BOOLEAN)
            ORDER BY complaint_id
            """,
            [str(cal_path)],
        ).fetchall()
    finally:
        con.close()
    selected = sorted(rows, key=lambda row: (_rank(int(row[0])), int(row[0])))[:limit]
    return [
        {
            "complaint_id": int(row[0]),
            "narrative": row[1],
            "source_product": row[2],
            "source_issue": row[3],
            "source_company": row[4],
            "rule_label": json.loads(row[5]),
        }
        for row in selected
    ]


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "".join(str(piece) for piece in pieces)
    raise ValueError("OpenRouter response content is not text")


def _reported_cost(response: dict[str, Any]) -> float | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    for key in ("cost", "total_cost"):
        value = usage.get(key)
        if value is not None:
            return float(value)
    return None


def _parse_teacher_content(raw_content: str) -> object:
    """Parse JSON, tolerating a provider-added Markdown fence for audit purposes."""

    text = raw_content.strip()
    if text.startswith("```") and text.endswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 : -3].strip()
    return json.loads(text)


def _request(
    *,
    api_key: str,
    model: str,
    prompt: str,
    row: dict[str, Any],
    timeout_seconds: float = 90.0,
) -> dict[str, Any]:
    record = {
        "complaint_id": row["complaint_id"],
        "product": row["source_product"],
        "issue": row["source_issue"],
        "company": row["source_company"],
        "narrative": row["narrative"],
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(record, ensure_ascii=False, sort_keys=True),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "phase1_triage_label",
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
            "X-Title": "frontier-forge Phase 1 spot-label audit",
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
        raise RuntimeError("existing spot-label receipts belong to a different frozen run")
    ids = [record["complaint_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate complaint_id in existing spot-label receipts")
    return records


def run_spot_labels(
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
    """Select ambiguous rows and optionally execute the budget-capped live audit."""

    cal_path = Path(cal_path)
    prompt_path = Path(prompt_path)
    receipts_path = Path(receipts_path)
    ledger_path = Path(ledger_path)
    if not cal_path.is_file():
        raise FileNotFoundError(f"CAL split is missing: {cal_path}")
    rule_config = yaml.safe_load(Path(rules_path).read_text())
    spot_config = rule_config["teacher_spot_labels"]
    model = str(spot_config["model"])
    budget = float(spot_config["max_budget_usd"])
    prompt = prompt_path.read_text()
    selected = _select_rows(cal_path, limit)
    if len(selected) != limit:
        raise RuntimeError(f"requested {limit} ambiguous rows but only found {len(selected)}")
    selection_ids = [row["complaint_id"] for row in selected]
    fingerprint = _canonical_hash(
        {
            "cal_sha256": sha256_file(cal_path),
            "rules_sha256": sha256_file(Path(rules_path)),
            "prompt_sha256": sha256_file(prompt_path),
            "model": model,
            "limit": limit,
            "selection_ids": selection_ids,
            "runner_sha256": sha256_file(Path(__file__)),
        }
    )
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger.get("run_fingerprint") == fingerprint and ledger.get("status") == "complete":
            return ledger, True
        raise RuntimeError("existing spot-label ledger is not the same completed frozen run")
    if not live:
        return {
            "status": "dry-run",
            "run_fingerprint": fingerprint,
            "model": model,
            "selected_complaint_ids": selection_ids,
            "budget_usd": budget,
            "network_calls": 0,
        }, False

    config = load_data_source_config(data_config_path)
    api_key = _load_key(config.upstream_root / ".env")
    receipts_path.parent.mkdir(parents=True, exist_ok=True)
    records = _existing_records(receipts_path, fingerprint)
    completed_ids = {int(record["complaint_id"]) for record in records}
    total_cost = sum(float(record.get("reported_cost_usd") or 0.0) for record in records)
    started_at = datetime.now(UTC).isoformat()

    for row in selected:
        if row["complaint_id"] in completed_ids:
            continue
        if total_cost > budget:
            raise RuntimeError(f"spot-label cost {total_cost:.6f} exceeded ${budget:.2f} cap")
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
            "sample_sha256": _canonical_hash(
                {
                    "complaint_id": row["complaint_id"],
                    "narrative": row["narrative"],
                    "source_product": row["source_product"],
                    "source_issue": row["source_issue"],
                    "source_company": row["source_company"],
                }
            ),
            "rule_label": row["rule_label"],
            "application": "audit-only",
        }
        if response is None:
            record.update({"status": "request_failed", "error": error, "reported_cost_usd": 0.0})
        else:
            message = response.get("choices", [{}])[0].get("message", {})
            raw_content = _content_text(message.get("content"))
            try:
                teacher_label = _parse_teacher_content(raw_content)
            except json.JSONDecodeError:
                teacher_label = None
            valid = is_schema_valid(teacher_label)
            cost = _reported_cost(response)
            record.update(
                {
                    "status": "ok" if valid else "invalid_teacher_output",
                    "response_id": response.get("id"),
                    "provider": response.get("provider"),
                    "raw_content": raw_content,
                    "teacher_label": teacher_label,
                    "schema_valid": valid,
                    "schema_errors": schema_errors(teacher_label),
                    "usage": response.get("usage"),
                    "reported_cost_usd": cost,
                }
            )
            if valid:
                breakdown = score({"label": row["rule_label"]}, teacher_label)
                record["agreement"] = {
                    "scorer_version": breakdown.scorer_version,
                    "task_success": breakdown.task_success,
                    "reward": breakdown.reward,
                    "field_matches": breakdown.field_matches,
                    "tool_choice_match": breakdown.tool_choice_match,
                    "tool_arguments_structural_valid": breakdown.tool_arguments_valid,
                    "secondary_metrics": breakdown.secondary_metrics,
                }
            if cost is not None:
                total_cost += cost
        records.append(record)
        _write_receipts(receipts_path, records)

    successful = [record for record in records if record["status"] == "ok"]
    missing_cost = [
        record["complaint_id"] for record in records if record.get("reported_cost_usd") is None
    ]
    ledger = {
        "version": 1,
        "status": "complete" if len(records) == limit and not missing_cost else "incomplete",
        "run_fingerprint": fingerprint,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "model": model,
        "prompt_path": str(prompt_path.resolve()),
        "prompt_sha256": sha256_file(prompt_path),
        "cal_path": str(cal_path.resolve()),
        "cal_sha256": sha256_file(cal_path),
        "selection_seed": SELECTION_SEED,
        "selected_rows": limit,
        "calls_recorded": len(records),
        "schema_valid_outputs": len(successful),
        "exact_rule_agreements": sum(
            bool(record.get("agreement", {}).get("task_success")) for record in successful
        ),
        "reported_api_usd": total_cost,
        "missing_cost_complaint_ids": missing_cost,
        "budget_usd": budget,
        "within_budget": not missing_cost and total_cost <= budget,
        "receipts_path": str(receipts_path.resolve()),
        "receipts_sha256": sha256_file(receipts_path),
        "application": "audit-only; no frozen labels overwritten",
    }
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    if ledger["status"] != "complete" or not ledger["within_budget"]:
        raise RuntimeError(
            "teacher spot-label ledger is incomplete or cannot prove the budget gate"
        )
    return ledger, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forge.data.spot_label")
    parser.add_argument("--cal", type=Path, default=DEFAULT_CAL_PATH)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS_PATH)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    ledger, noop = run_spot_labels(
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
    print(
        f"teacher spot labels: {status}; calls={ledger.get('calls_recorded', 0)}; "
        f"api_usd={ledger.get('reported_api_usd', 0.0):.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
