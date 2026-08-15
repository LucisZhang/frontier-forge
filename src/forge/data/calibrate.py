"""Phase-1 difficulty calibration plumbing and evidence report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import duckdb
import yaml

from forge.data.ingest import sha256_file
from forge.data.labels import comparison_text
from forge.data.splits import DEFAULT_OUTPUT_DIR, SMOKE_OUTPUT_DIR
from forge.verify.verifier import score

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "difficulty_candidates.yaml"
DEFAULT_REPORT_PATH = REPO_ROOT / "results" / "phase1_calibration_report.md"
DEFAULT_API_LEDGER_PATH = REPO_ROOT / "results" / "phase1_api_calibration_ledger.json"
SMOKE_REPORT_PATH = REPO_ROOT / "data" / "smoke" / "phase1_calibration_report.md"
CALIBRATION_SEED = 20260815
FULL_SAMPLE_CAP = 200
SMOKE_SAMPLE_CAP = 20


def _rank(complaint_id: int) -> bytes:
    return hashlib.blake2b(f"{CALIBRATION_SEED}:{complaint_id}".encode(), digest_size=16).digest()


def _sample_rows(path: Path, cap: int) -> list[dict[str, Any]]:
    con = duckdb.connect()
    try:
        raw = con.execute(
            """
            SELECT complaint_id, narrative, label_json
            FROM read_parquet(?) ORDER BY complaint_id
            """,
            [str(path)],
        ).fetchall()
    finally:
        con.close()
    selected = sorted(raw, key=lambda row: (_rank(int(row[0])), int(row[0])))[:cap]
    return [
        {"complaint_id": int(cid), "narrative": narrative, "label": json.loads(label_json)}
        for cid, narrative, label_json in selected
    ]


_PRODUCT_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mortgage", ("mortgage", "foreclosure", "escrow")),
    ("student_loan", ("student loan",)),
    ("vehicle_loan", ("vehicle", "auto loan", "car loan", "repossession")),
    ("debt_collection", ("debt collector", "collection agency", "collect a debt")),
    ("credit_reporting", ("credit report", "credit bureau", "credit score")),
    ("deposit_account", ("checking account", "savings account", "bank account")),
    ("money_service", ("money transfer", "wire transfer", "virtual currency")),
    ("payday_personal_loan", ("payday", "title loan", "personal loan")),
    ("card", ("credit card", "prepaid card", "card charge")),
)

_HIGH_HINTS = (
    "foreclosure",
    "eviction",
    "identity theft",
    "garnishment",
    "repossession",
    "account takeover",
)
_MEDIUM_HINTS = ("fraud", "unauthorized", "dispute", "collection", "charged", "denied")


def stand_in_prediction(narrative: str) -> dict[str, Any]:
    """Rule-blind deterministic stand-in used only to exercise local calibration."""

    text = comparison_text(narrative)
    product = "credit_reporting"
    for candidate, hints in _PRODUCT_HINTS:
        if any(hint in text for hint in hints):
            product = candidate
            break
    if any(hint in text for hint in _HIGH_HINTS):
        urgency = "high"
    elif any(hint in text for hint in _MEDIUM_HINTS):
        urgency = "medium"
    else:
        urgency = "low"
    return {
        "product": product,
        "issue": "unspecified",
        "company": None,
        "urgency": urgency,
        "ambiguity_flag": True,
        "tool_call": {
            "name": "request_more_info",
            "arguments": {
                "missing_fields": ["issue", "company"],
                "question": "Please provide the issue and company so this complaint can be routed.",
            },
        },
    }


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def evaluate_stand_in(rows: list[dict[str, Any]]) -> dict[str, Any]:
    breakdowns = [score(row, stand_in_prediction(str(row["narrative"]))) for row in rows]
    total = len(breakdowns)

    def rate(attribute: str) -> float:
        return sum(bool(getattr(item, attribute)) for item in breakdowns) / total if total else 0.0

    successes = sum(item.task_success for item in breakdowns)
    low, high = _wilson(successes, total)
    return {
        "samples": total,
        "task_success_count": successes,
        "task_success": successes / total if total else 0.0,
        "task_success_ci95_wilson": [low, high],
        "schema_valid": rate("schema_valid"),
        "product_match": rate("product_match"),
        "issue_match": rate("issue_match"),
        "company_match": rate("company_match"),
        "urgency_match": rate("urgency_match"),
        "ambiguity_flag_match": rate("ambiguity_flag_match"),
        "tool_choice_match": rate("tool_choice_match"),
        "tool_arguments_match": rate("tool_arguments_match"),
        "abstention_correct": rate("abstention_correct"),
        "mean_reward": sum(item.reward for item in breakdowns) / total if total else 0.0,
    }


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_report(
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    cal_path: Path,
    smoke: bool,
    api_ledger: dict[str, Any] | None,
) -> str:
    target_low, target_high = (float(value) for value in config["target_task_success_band"])
    primary = api_ledger["metrics"] if api_ledger is not None else metrics
    success = float(primary["task_success"])
    if target_low <= success <= target_high:
        comparison = "inside"
    elif success < target_low:
        comparison = "below"
    else:
        comparison = "above"
    candidate = config["evaluated_candidate"]
    low, high = primary["task_success_ci95_wilson"]
    mode = "SMOKE" if smoke else "local Phase-1"
    lines = [
        "# Phase 1 difficulty calibration report",
        "",
        "## Decision status",
        "",
        (
            "**HUMAN DECISION REQUIRED.** No final difficulty setting or D3 task "
            "fallback is selected here."
        ),
        "",
    ]
    if api_ledger is not None:
        lines.extend(
            [
                (
                    f"The receipt-backed API stand-in scored **{_percent(success)}** "
                    f"task success on {primary['samples']} frozen CAL rows (95% Wilson "
                    f"CI {_percent(low)}–{_percent(high)}), which is **{comparison}** "
                    f"the D3 target band of {_percent(target_low)}–"
                    f"{_percent(target_high)}."
                ),
                "",
                (
                    f"This is zero-shot evidence from `{api_ledger['model']}`, with only "
                    "complaint ID and narrative visible to the model. It is a small API "
                    "stand-in, not the Qwen base-model result. The first valid full "
                    "calibration remains the human-launched Qwen base run in Phase 3."
                ),
            ]
        )
    else:
        lines.extend(
            [
                (
                    f"The {mode} stand-in scored **{_percent(success)}** task success "
                    f"(95% Wilson CI {_percent(low)}–{_percent(high)}), which is "
                    f"**{comparison}** the D3 target band of "
                    f"{_percent(target_low)}–{_percent(target_high)}."
                ),
                "",
                (
                    "This is plumbing evidence from "
                    "`deterministic-rule-blind-stand-in-v1`, not a Qwen base-model "
                    "result. It makes no model or API calls."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Evaluated difficulty knobs",
            "",
            f"- Candidate: `{candidate['name']}` (proposed for human review)",
            f"- Schema fields: {candidate['schema_fields']}",
            f"- Product classes: {candidate['product_classes']}",
            (
                f"- Tools: {candidate['tool_count']} plus "
                f"{candidate['distractor_tool_count']} distractor slots"
            ),
            f"- Ambiguous-sample ratio: `{candidate['ambiguous_sample_ratio']}`",
            f"- Bilingual instructions: `{str(candidate['bilingual_instructions']).lower()}`",
            f"- Issue scoring: `{candidate['issue_match']}`",
            "",
            (
                "Alternatives retained for the owner: `c0_reduced_ticket` (easier) and "
                "`c2_distractor_heavy` (harder). Changing to either requires an explicit "
                "human decision before any new split materialization; frozen artifacts "
                "are never edited in place."
            ),
            "",
            "## API stand-in metrics" if api_ledger is not None else "## Stand-in metrics",
            "",
            "| Metric | Result |",
            "|---|---:|",
        ]
    )
    ordered = (
        ("Samples", "samples", False),
        ("Task success", "task_success", True),
        ("Schema valid", "schema_valid", True),
        ("Product match", "product_match", True),
        ("Issue match", "issue_match", True),
        ("Company match", "company_match", True),
        ("Urgency match", "urgency_match", True),
        ("Ambiguity match", "ambiguity_flag_match", True),
        ("Tool choice", "tool_choice_match", True),
        ("Tool arguments", "tool_arguments_match", True),
        ("Abstention correctness", "abstention_correct", True),
        ("Mean verifier reward", "mean_reward", True),
    )
    for label, key, as_rate in ordered:
        value = _percent(float(primary[key])) if as_rate else str(primary[key])
        lines.append(f"| {label} | {value} |")
    if api_ledger is not None:
        lines.extend(
            [
                "",
                "## Deterministic plumbing baseline",
                "",
                (
                    "`deterministic-rule-blind-stand-in-v1` made zero network calls and "
                    f"scored {_percent(float(metrics['task_success']))} task success on "
                    f"{metrics['samples']} CAL rows. It remains a pipeline check only."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- CAL artifact: `{cal_path}`",
            f"- CAL SHA-256: `{sha256_file(cal_path)}`",
            "- Command: `make calibrate-difficulty` (or `SMOKE=1 make calibrate-difficulty`)",
        ]
    )
    if api_ledger is not None:
        lines.extend(
            [
                (
                    "- API stand-in command: "
                    "`python -m forge.data.api_calibrate --live`; subsequent report "
                    "builds consume the frozen receipts offline"
                ),
                f"- API calls: {api_ledger['calls_recorded']}",
                (
                    f"- API selection: smallest "
                    f"`blake2b('{CALIBRATION_SEED}:<complaint_id>')`, "
                    f"cap {api_ledger['selected_rows']}"
                ),
                f"- Reported API cost: ${api_ledger['reported_api_usd']:.6f}",
                f"- API receipts SHA-256: `{api_ledger['receipts_sha256']}`",
                (f"- Deterministic selection: same rank, cap {metrics['samples']}"),
                "",
            ]
        )
    else:
        lines.extend(["- Network calls: 0 for this stand-in run", ""])
    return "\n".join(lines)


def _load_api_ledger(path: Path, cal_path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    ledger = json.loads(path.read_text())
    receipts_path = Path(ledger["receipts_path"])
    if not receipts_path.is_file():
        receipts_path = REPO_ROOT / "results" / receipts_path.name
    if ledger.get("status") != "complete" or not ledger.get("within_budget"):
        raise ValueError("API calibration ledger is not complete and within budget")
    if ledger.get("cal_sha256") != sha256_file(cal_path):
        raise ValueError("API calibration ledger targets a different CAL artifact")
    if not receipts_path.is_file() or sha256_file(receipts_path) != ledger.get("receipts_sha256"):
        raise ValueError("API calibration receipt hash mismatch")
    return ledger


def run_calibration(
    *,
    cal_path: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    api_ledger_path: Path = DEFAULT_API_LEDGER_PATH,
    smoke: bool = False,
) -> dict[str, Any]:
    cal_path = Path(cal_path)
    if not cal_path.is_file():
        raise FileNotFoundError(f"CAL split is missing: {cal_path}")
    config = yaml.safe_load(Path(config_path).read_text())
    rows = _sample_rows(cal_path, SMOKE_SAMPLE_CAP if smoke else FULL_SAMPLE_CAP)
    metrics = evaluate_stand_in(rows)
    api_ledger = None if smoke else _load_api_ledger(Path(api_ledger_path), cal_path)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(
            config=config,
            metrics=metrics,
            cal_path=cal_path,
            smoke=smoke,
            api_ledger=api_ledger,
        )
    )
    return api_ledger["metrics"] if api_ledger is not None else metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forge.data.calibrate")
    parser.add_argument("--cal", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--api-ledger", type=Path, default=DEFAULT_API_LEDGER_PATH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    smoke = args.smoke or os.environ.get("SMOKE") == "1"
    cal_path = args.cal or (SMOKE_OUTPUT_DIR if smoke else DEFAULT_OUTPUT_DIR) / "cal.parquet"
    report_path = args.report or (SMOKE_REPORT_PATH if smoke else DEFAULT_REPORT_PATH)
    metrics = run_calibration(
        cal_path=cal_path,
        config_path=args.config,
        report_path=report_path,
        api_ledger_path=args.api_ledger,
        smoke=smoke,
    )
    print(
        f"calibration: samples={metrics['samples']}; "
        f"task_success={metrics['task_success']:.3f}; report={report_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
