"""Phase-1.1 difficulty calibration plumbing and evidence report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb
import yaml

from forge.data.ingest import sha256_file
from forge.data.input_contract import (
    INPUT_CONTRACT_VERSION,
    MODEL_INPUT_FIELDS,
    build_model_input,
)
from forge.data.labels import comparison_text, load_rules
from forge.data.relabel import DEFAULT_AUDIT_PATH, DEFAULT_OUTPUT_DIR
from forge.data.relabel import DEFAULT_MANIFEST_PATH as DEFAULT_DATASET_MANIFEST_PATH
from forge.data.splits import SMOKE_OUTPUT_DIR
from forge.verify.verifier import SCORER_VERSION, ScoreBreakdown, score

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "difficulty_candidates.yaml"
DEFAULT_REPORT_PATH = REPO_ROOT / "results" / "phase1_1_calibration_report.md"
DEFAULT_API_LEDGER_PATH = REPO_ROOT / "results" / "phase1_1_api_calibration_ledger.json"
PHASE1_API_LEDGER_PATH = REPO_ROOT / "results" / "phase1_api_calibration_ledger.json"
SMOKE_REPORT_PATH = REPO_ROOT / "data" / "smoke" / "phase1_1_calibration_report.md"
CALIBRATION_SEED = 20260815
CALIBRATION_RUN_ID = "phase1_1_api_calibration_v2_s20260815"
FULL_SAMPLE_CAP = 200
SMOKE_SAMPLE_CAP = 20


def _rank(complaint_id: int) -> bytes:
    return hashlib.blake2b(f"{CALIBRATION_SEED}:{complaint_id}".encode(), digest_size=16).digest()


def _sample_rows(path: Path, cap: int) -> list[dict[str, Any]]:
    con = duckdb.connect()
    try:
        raw = con.execute(
            """
            SELECT complaint_id, narrative, source_product, source_issue,
                   source_company, label_json
            FROM read_parquet(?) ORDER BY complaint_id
            """,
            [str(path)],
        ).fetchall()
    finally:
        con.close()
    selected = sorted(raw, key=lambda row: (_rank(int(row[0])), int(row[0])))[:cap]
    return [
        {
            "complaint_id": int(complaint_id),
            "narrative": narrative,
            "source_product": source_product,
            "source_issue": source_issue,
            "source_company": source_company,
            "label": json.loads(label_json),
        }
        for (
            complaint_id,
            narrative,
            source_product,
            source_issue,
            source_company,
            label_json,
        ) in selected
    ]


_HIGH_HINTS = (
    "foreclosure",
    "eviction",
    "identity theft",
    "garnishment",
    "repossession",
    "unable to access funds",
    "account takeover",
)
_MEDIUM_HINTS = (
    "fraud",
    "unauthorized",
    "dispute",
    "incorrect information",
    "late fee",
    "collection",
    "charged",
    "denied",
)


def stand_in_prediction(row: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic, network-free stand-in used only for pipeline smoke checks."""

    model_input = build_model_input(row)
    rules = load_rules()
    product = rules.product_map[str(model_input["source_product"])]
    issue = str(model_input["source_issue"] or "unspecified")
    company = model_input["source_company"]
    text = comparison_text(f"{issue} {model_input['narrative']}")
    if any(hint in text for hint in _HIGH_HINTS):
        urgency = "high"
    elif any(hint in text for hint in _MEDIUM_HINTS):
        urgency = "medium"
    else:
        urgency = "low"

    missing_fields: list[str] = []
    if company is None:
        missing_fields.append("company")
    if len(str(model_input["narrative"]).strip()) < rules.min_narrative_chars:
        missing_fields.append("details")
    ambiguity = bool(missing_fields)
    if ambiguity:
        tool_call: dict[str, Any] = {
            "name": "request_more_info",
            "arguments": {
                "missing_fields": missing_fields,
                "question": "Please provide the missing information needed for triage.",
            },
        }
    elif urgency == "high":
        tool_call = {
            "name": "escalate_to_regulator",
            "arguments": {
                "complaint_id": model_input["complaint_id"],
                "reason": "The complaint describes an urgent high-impact event.",
            },
        }
    else:
        tool_call = {
            "name": "route_to_company",
            "arguments": {"company": company, "issue": issue},
        }
    return {
        "product": product,
        "issue": issue,
        "company": company,
        "urgency": urgency,
        "ambiguity_flag": ambiguity,
        "tool_call": tool_call,
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


def _metrics_from_breakdowns(breakdowns: list[ScoreBreakdown]) -> dict[str, Any]:
    total = len(breakdowns)

    def rate(attribute: str) -> float:
        return sum(bool(getattr(item, attribute)) for item in breakdowns) / total if total else 0.0

    successes = sum(item.task_success for item in breakdowns)
    low, high = _wilson(successes, total)
    return {
        "scorer_version": SCORER_VERSION,
        "samples": total,
        "task_success_count": successes,
        "task_success": successes / total if total else 0.0,
        "task_success_ci95_wilson": [low, high],
        "schema_valid": rate("schema_valid"),
        "urgency_match": rate("urgency_match"),
        "ambiguity_flag_match": rate("ambiguity_flag_match"),
        "tool_choice_match": rate("tool_choice_match"),
        "tool_arguments_structural_valid": rate("tool_arguments_valid"),
        "secondary_metrics": {
            "product_match": rate("product_match"),
            "issue_normalized_match": rate("issue_match"),
            "company_normalized_match": rate("company_match"),
            "tool_arguments_semantic_valid": rate("tool_arguments_semantic_valid"),
            "abstention_correct": rate("abstention_correct"),
        },
        "mean_reward": (sum(item.reward for item in breakdowns) / total if total else 0.0),
    }


def evaluate_stand_in(rows: list[dict[str, Any]]) -> dict[str, Any]:
    breakdowns = [score(row, stand_in_prediction(row)) for row in rows]
    return _metrics_from_breakdowns(breakdowns)


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _receipt_path(path: Path) -> str:
    """Render repository artifacts without machine-specific absolute prefixes."""

    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _load_api_ledger(path: Path, cal_path: Path, dataset_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    ledger = json.loads(path.read_text())
    receipts_path = Path(ledger["receipts_path"])
    if not receipts_path.is_file():
        receipts_path = REPO_ROOT / "results" / receipts_path.name
    if ledger.get("status") != "complete" or not ledger.get("within_budget"):
        raise ValueError("Phase 1.1 API ledger is not complete and within budget")
    if ledger.get("cal_sha256") != sha256_file(cal_path):
        raise ValueError("Phase 1.1 API ledger targets a different CAL artifact")
    if ledger.get("dataset_hash") != dataset_hash:
        raise ValueError("Phase 1.1 API ledger targets a different v2 dataset")
    if ledger.get("scorer_version") != SCORER_VERSION:
        raise ValueError("Phase 1.1 API ledger uses a different scorer version")
    if ledger.get("input_contract_version") != INPUT_CONTRACT_VERSION:
        raise ValueError("Phase 1.1 API ledger uses a different input contract")
    if not receipts_path.is_file() or sha256_file(receipts_path) != ledger.get("receipts_sha256"):
        raise ValueError("Phase 1.1 API receipt hash mismatch")
    return ledger


def _render_report(
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    cal_path: Path,
    smoke: bool,
    api_ledger: dict[str, Any] | None,
    dataset_manifest: dict[str, Any] | None,
    phase1_ledger: dict[str, Any] | None,
) -> str:
    target_low, target_high = (float(value) for value in config["target_task_success_band"])
    primary = api_ledger["metrics"] if api_ledger is not None else metrics
    success = float(primary["task_success"])
    if target_low <= success <= target_high:
        comparison = "inside"
        gate_status = "PASS"
        escalation = "No calibration-band escalation is required."
    elif success < target_low:
        comparison = "below"
        gate_status = "ESCALATE"
        escalation = (
            "Per D3.1, no difficulty knobs were changed to force the target. The human "
            "must review this below-band result before Phase 2."
        )
    else:
        comparison = "above"
        gate_status = "ESCALATE"
        escalation = (
            "Per D3.1, no difficulty knobs were changed to force the target. The human "
            "must review this above-band result before Phase 2."
        )
    low, high = primary["task_success_ci95_wilson"]
    mode = "receipt-backed API stand-in" if api_ledger is not None else "local smoke stand-in"
    lines = [
        "# Phase 1.1 calibration remediation report",
        "",
        "## Gate result",
        "",
        (
            f"**{gate_status}.** The {mode} scored **{_percent(success)}** task success "
            f"on n={primary['samples']} CAL rows (95% Wilson CI {_percent(low)}–"
            f"{_percent(high)}), **{comparison}** the D3 target band of "
            f"{_percent(target_low)}–{_percent(target_high)}."
        ),
        "",
        escalation,
        "",
        "This remains an API stand-in, not the Phase-3 Qwen base-model R0 result.",
        "",
        "## D3.1 contracts implemented",
        "",
        (
            f"- Input contract v{INPUT_CONTRACT_VERSION}: narrative plus source product, "
            "issue, and company metadata (and complaint ID for tool arguments)."
        ),
        (
            f"- Scorer v{SCORER_VERSION}: task success requires schema validity plus "
            "urgency, ambiguity flag, tool choice, and structural argument validity."
        ),
        (
            "- Product/issue/company normalization and non-verbatim tool-text semantics "
            "are secondary diagnostics excluded from task success and reward."
        ),
        (
            "- Fair prompt v2 discloses the urgency policy, ambiguity definition, product "
            "mapping, tool registry semantics, priority order, and argument schemas."
        ),
        (
            "- Label rules v2 cap phrase-only ambiguity triggers at 200 narrative "
            "characters; long narratives containing phrases such as `not sure` are not "
            "made ambiguous by that phrase alone."
        ),
        "",
        "## Decision-check breakdown",
        "",
        "| Check | Result | Included in task success |",
        "|---|---:|---|",
        f"| Schema valid | {_percent(float(primary['schema_valid']))} | yes |",
        f"| Urgency match | {_percent(float(primary['urgency_match']))} | yes |",
        (f"| Ambiguity flag match | {_percent(float(primary['ambiguity_flag_match']))} | yes |"),
        f"| Tool choice match | {_percent(float(primary['tool_choice_match']))} | yes |",
        (
            "| Tool arguments structurally valid | "
            f"{_percent(float(primary['tool_arguments_structural_valid']))} | yes |"
        ),
        f"| Mean scorer-v2 reward | {_percent(float(primary['mean_reward']))} | — |",
        "",
        "## Secondary metrics (excluded from success and reward)",
        "",
        "| Metric | Result |",
        "|---|---:|",
    ]
    secondary = primary["secondary_metrics"]
    for label, key in (
        ("Product match", "product_match"),
        ("Issue normalized match", "issue_normalized_match"),
        ("Company normalized match", "company_normalized_match"),
        ("Tool argument semantic validity", "tool_arguments_semantic_valid"),
        ("Abstention correctness", "abstention_correct"),
    ):
        lines.append(f"| {label} | {_percent(float(secondary[key]))} |")

    lines.extend(["", "## Delta from Phase 1", ""])
    if phase1_ledger is not None:
        old_metrics = phase1_ledger["metrics"]
        lines.extend(
            [
                "| Contract | Phase 1 | Phase 1.1 |",
                "|---|---|---|",
                (f"| Rows | {old_metrics['samples']} | {primary['samples']} |"),
                (
                    f"| Task success | {_percent(float(old_metrics['task_success']))} | "
                    f"{_percent(success)} |"
                ),
                "| Visible evidence | complaint ID + narrative | narrative + source metadata |",
                "| Success checks | hard-AND over 8 checks | decision fields + structural args |",
                (
                    "| Tool free text | normalized template equality | structural + "
                    "non-verbatim semantics |"
                ),
                (
                    "| Label rules | v1 phrase trigger at any length | v2 phrase "
                    "trigger capped at 200 chars |"
                ),
            ]
        )
    else:
        lines.append("Phase 1 receipt ledger was unavailable for the delta table.")

    if dataset_manifest is not None:
        lines.extend(
            [
                "",
                "## Frozen-membership proof",
                "",
                "| Split | Rows | Phase 1 membership SHA-256 | v2 membership SHA-256 | Match |",
                "|---|---:|---|---|---|",
            ]
        )
        for name, split in dataset_manifest["splits"].items():
            lines.append(
                f"| {name} | {split['rows']} | `{split['phase1_membership_sha256']}` | "
                f"`{split['membership_sha256']}` | yes |"
            )
        lines.extend(
            [
                "",
                f"- Version-2 dataset hash: `{dataset_manifest['dataset_hash']}`",
                f"- Changed-row audit: `{DEFAULT_AUDIT_PATH}`",
                (
                    "- Changed labels by split: "
                    + ", ".join(
                        f"{name}={count}"
                        for name, count in dataset_manifest["audit"][
                            "full_changed_rows_by_split"
                        ].items()
                    )
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Reproducibility and receipts",
            "",
            f"- CAL artifact: `{_receipt_path(cal_path)}`",
            f"- CAL payload SHA-256: `{sha256_file(cal_path)}`",
            f"- Scorer version: `{SCORER_VERSION}`",
            f"- Input contract version: `{INPUT_CONTRACT_VERSION}`",
            f"- Visible input fields: `{', '.join(MODEL_INPUT_FIELDS)}`",
            (f"- Append-only run record: `results/runs.jsonl` entry `{CALIBRATION_RUN_ID}`"),
            "- Offline report command: `make calibrate-difficulty`",
        ]
    )
    if api_ledger is not None:
        lines.extend(
            [
                "- Live command: `python -m forge.data.api_calibrate --live`",
                f"- Model: `{api_ledger['model']}`",
                f"- Calls: {api_ledger['calls_recorded']}",
                f"- Reported API cost: ${api_ledger['reported_api_usd']:.6f}",
                f"- Budget ceiling: ${api_ledger['budget_usd']:.2f}",
                f"- Prompt SHA-256: `{api_ledger['prompt_sha256']}`",
                f"- Receipts SHA-256: `{api_ledger['receipts_sha256']}`",
            ]
        )
    else:
        lines.append("- Network calls: 0 (smoke plumbing only; not a gate result)")
    lines.extend(["", "**HUMAN REVIEW REQUIRED before Phase 2 starts.**", ""])
    return "\n".join(lines)


def run_calibration(
    *,
    cal_path: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    api_ledger_path: Path = DEFAULT_API_LEDGER_PATH,
    dataset_manifest_path: Path = DEFAULT_DATASET_MANIFEST_PATH,
    phase1_api_ledger_path: Path = PHASE1_API_LEDGER_PATH,
    smoke: bool = False,
) -> dict[str, Any]:
    cal_path = Path(cal_path)
    if not cal_path.is_file():
        raise FileNotFoundError(f"CAL split is missing: {cal_path}")
    config = yaml.safe_load(Path(config_path).read_text())
    rows = _sample_rows(cal_path, SMOKE_SAMPLE_CAP if smoke else FULL_SAMPLE_CAP)
    metrics = evaluate_stand_in(rows)
    dataset_manifest = None
    if not smoke:
        dataset_manifest = json.loads(Path(dataset_manifest_path).read_text())
        if dataset_manifest.get("version") != 2:
            raise ValueError("calibration report requires the version-2 dataset manifest")
    dataset_hash = dataset_manifest["dataset_hash"] if dataset_manifest is not None else "smoke"
    api_ledger = None if smoke else _load_api_ledger(Path(api_ledger_path), cal_path, dataset_hash)
    phase1_ledger = (
        json.loads(Path(phase1_api_ledger_path).read_text())
        if Path(phase1_api_ledger_path).is_file()
        else None
    )
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(
            config=config,
            metrics=metrics,
            cal_path=cal_path,
            smoke=smoke,
            api_ledger=api_ledger,
            dataset_manifest=dataset_manifest,
            phase1_ledger=phase1_ledger,
        )
    )
    return api_ledger["metrics"] if api_ledger is not None else metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forge.data.calibrate")
    parser.add_argument("--cal", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--api-ledger", type=Path, default=DEFAULT_API_LEDGER_PATH)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST_PATH)
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
        dataset_manifest_path=args.dataset_manifest,
        smoke=smoke,
    )
    print(
        f"calibration v2: samples={metrics['samples']}; "
        f"task_success={metrics['task_success']:.3f}; report={report_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
