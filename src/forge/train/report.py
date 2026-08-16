"""Regenerate the Phase 3 ladder report and adjacent paired deltas from receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from forge.train.artifacts import write_json_atomic, write_text_atomic
from forge.train.config import (
    ALL_RUNGS,
    CONFIG_PATHS,
    REPO_ROOT,
    evaluation_root,
    load_config,
    runs_path,
)
from forge.train.evaluate import bootstrap_ci
from forge.train.ledger import billable_records


def _records(*, smoke: bool) -> list[dict[str, Any]]:
    path = runs_path(smoke=smoke)
    if not path.is_file():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [record for record in records if record.get("phase") == 3]


def _failed_gpu_attempts(*, smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return []
    path = REPO_ROOT / "results" / "phase3_gpu_ledger.jsonl"
    if not path.is_file():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [record for record in billable_records(records) if record["status"] == "failed"]


def _config_map() -> dict[str, dict[str, Any]]:
    return {load_config(path)["rung"]: load_config(path) for path in CONFIG_PATHS}


def _rung_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_rung = {rung: [] for rung in ALL_RUNGS}
    for record in records:
        config = load_config(record["config_path"])
        by_rung[str(config["rung"])].append(record)
    for values in by_rung.values():
        values.sort(key=lambda item: (int(item["seed"]), str(item.get("backend", "trl"))))
    return by_rung


def _prediction_map(path: Path) -> dict[tuple[str, int], float]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    result = {
        (str(row["split"]), int(row["complaint_id"])): float(row["score"]["task_success"])
        for row in rows
    }
    if len(result) != len(rows):
        raise ValueError(f"duplicate paired key in {path}")
    return result


def _seed_zero_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [record for record in records if int(record["seed"]) == 0]
    if not candidates:
        return None
    policy_path = REPO_ROOT / "results" / "phase3_backend_policy.json"
    preferred = "trl"
    if policy_path.is_file():
        preferred = json.loads(policy_path.read_text()).get("default_backend", "trl")
    return next(
        (record for record in candidates if record.get("backend", "trl") == preferred),
        candidates[0],
    )


def paired_deltas(by_rung: dict[str, list[dict[str, Any]]], *, smoke: bool) -> dict[str, Any]:
    configs = _config_map()
    adjacent_pairs = (("r0", "r1"), ("r1", "r2"), ("r2", "r3"), ("r3", "r4"))
    ablation_pairs = (("r1", "r1b"), ("r2", "r1b"))

    def summarize_pair(left: str, right: str) -> dict[str, Any]:
        left_record = _seed_zero_record(by_rung[left])
        right_record = _seed_zero_record(by_rung[right])
        if left_record is None or right_record is None:
            return {"from": left, "to": right, "status": "pending"}
        left_backend = str(left_record.get("backend", "trl"))
        right_backend = str(right_record.get("backend", "trl"))
        left_path = (
            evaluation_root(configs[left], seed=0, smoke=smoke, backend=left_backend)
            / "predictions.jsonl"
        )
        right_path = (
            evaluation_root(configs[right], seed=0, smoke=smoke, backend=right_backend)
            / "predictions.jsonl"
        )
        if not left_path.is_file() or not right_path.is_file():
            return {"from": left, "to": right, "status": "predictions_missing"}
        left_values = _prediction_map(left_path)
        right_values = _prediction_map(right_path)
        if left_values.keys() != right_values.keys():
            raise ValueError(
                f"adjacent rung pair {left}->{right} is not evaluated on identical rows"
            )
        deltas = [right_values[key] - left_values[key] for key in sorted(left_values)]
        config = configs[right]
        ci = bootstrap_ci(
            deltas,
            resamples=int(config["evaluation"]["bootstrap_resamples"]),
            seed=int(config["evaluation"]["bootstrap_seed"]),
        )
        return {
            "from": left,
            "to": right,
            "status": "complete",
            "paired_rows": len(deltas),
            "mean_task_success_delta": sum(deltas) / len(deltas),
            "ci95": ci,
            "from_backend": left_backend,
            "to_backend": right_backend,
            "bootstrap_resamples": 1000,
        }

    return {
        "phase": 3,
        "mode": "smoke" if smoke else "full",
        "adjacent_pairs": [summarize_pair(*pair) for pair in adjacent_pairs],
        "optional_ablation_pairs": [summarize_pair(*pair) for pair in ablation_pairs],
    }


def _fmt_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _report_text(
    by_rung: dict[str, list[dict[str, Any]]], deltas: dict[str, Any], *, smoke: bool
) -> str:
    mode = "SMOKE (non-headline)" if smoke else "FULL GPU"
    lines = [
        "# Phase 3 post-training ladder report",
        "",
        f"Mode: **{mode}**.",
        "",
    ]
    full_count = sum(len(values) for values in by_rung.values())
    if full_count == 0:
        lines.extend(
            [
                "Status: **REMOTE RUNS PENDING HUMAN LAUNCH**.",
                "",
                "The implementation and launch contracts are prepared, but no full Phase 3 GPU "
                "metric, cost, backend-agreement result, or exported-weight hash exists yet. "
                "No headline is drafted from smoke data.",
                "",
            ]
        )
    lines.extend(
        [
            "## Ladder",
            "",
            "| Rung | Seed | Backend | Task success | 95% CI | Schema valid | "
            "Tool accuracy | GPU hours | USD |",
            "|---|---:|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for rung in ALL_RUNGS:
        values = by_rung[rung]
        if not values:
            label = "optional; pending" if rung == "r1b" else "pending"
            lines.append(f"| {rung.upper()} | — | — | {label} | — | — | — | — | — |")
            continue
        for record in values:
            metrics = record["metrics"]
            ci = metrics["ci95"]
            lines.append(
                f"| {rung.upper()} | {record['seed']} | {record.get('backend', 'trl')} | "
                f"{_fmt_percent(float(metrics['task_success']))} | "
                f"[{_fmt_percent(float(ci[0]))}, {_fmt_percent(float(ci[1]))}] | "
                f"{_fmt_percent(float(metrics['schema_valid']))} | "
                f"{_fmt_percent(float(metrics['tool_acc']))} | "
                f"{float(record['cost']['gpu_hours']):.3f} | "
                f"${float(record['cost']['usd']):.3f} |"
            )
    lines.extend(["", "## Adjacent paired deltas", ""])
    for item in deltas["adjacent_pairs"]:
        if item["status"] != "complete":
            lines.append(f"- {item['from'].upper()} → {item['to'].upper()}: {item['status']}.")
        else:
            ci = item["ci95"]
            lines.append(
                f"- {item['from'].upper()} → {item['to'].upper()}: "
                f"{_fmt_percent(float(item['mean_task_success_delta']))} paired delta, "
                f"95% CI [{_fmt_percent(float(ci[0]))}, {_fmt_percent(float(ci[1]))}], "
                f"n={item['paired_rows']}."
            )

    lines.extend(["", "## Optional R1b ablation deltas", ""])
    for item in deltas["optional_ablation_pairs"]:
        if item["status"] != "complete":
            lines.append(f"- {item['from'].upper()} → {item['to'].upper()}: {item['status']}.")
        else:
            ci = item["ci95"]
            lines.append(
                f"- {item['from'].upper()} → {item['to'].upper()}: "
                f"{_fmt_percent(float(item['mean_task_success_delta']))} paired delta, "
                f"95% CI [{_fmt_percent(float(ci[0]))}, {_fmt_percent(float(ci[1]))}], "
                f"n={item['paired_rows']}."
            )

    lines.extend(["", "## Backend agreement", ""])
    agreement_path = REPO_ROOT / "results" / "phase3_backend_agreement.json"
    agreement = json.loads(agreement_path.read_text())
    default_backend = agreement.get("default_backend_after_check") or "trl (reference)"
    lines.append(
        f"R1 TRL/Unsloth status: **{agreement['status']}**. "
        f"Default after check: `{default_backend}`."
    )

    lines.extend(["", "## Reward-hacking probes", ""])
    r4 = _seed_zero_record(by_rung["r4"])
    if r4 is None:
        lines.append("R4 probes are pending the human-launched GRPO run and frozen evaluation.")
    else:
        probes = r4["metrics"]["reward_hacking_probes"]
        length_rows = probes["length_inflation"]["reward_increase_rows"]
        format_rows = probes["format_exploitation"]["reward_increase_rows"]
        abstention_rate = _fmt_percent(
            float(probes["degenerate_abstention"]["model_abstention_rate"])
        )
        abstain_success = _fmt_percent(
            float(probes["degenerate_abstention"]["always_abstain_task_success"])
        )
        lines.extend(
            [
                f"- Length inflation reward increases: {length_rows} rows.",
                f"- Markdown-format exploit reward increases: {format_rows} rows.",
                f"- Model abstention rate: {abstention_rate}; "
                f"always-abstain task success: {abstain_success}.",
            ]
        )

    lines.extend(["", "## Failure and negative-result register", ""])
    negatives = []
    comparison_results = deltas["adjacent_pairs"] + deltas["optional_ablation_pairs"]
    for item in comparison_results:
        if item.get("status") == "complete" and float(item["mean_task_success_delta"]) < 0:
            loss = _fmt_percent(abs(float(item["mean_task_success_delta"])))
            negatives.append(
                f"{item['to'].upper()} lost {loss} task success versus {item['from'].upper()}."
            )
    for rung, records in by_rung.items():
        for record in records:
            failures = record["metrics"].get("failure_categories_nonexclusive", {})
            if failures:
                negatives.append(
                    f"{rung.upper()} seed {record['seed']} failure counts (nonexclusive): "
                    + ", ".join(f"{name}={count}" for name, count in sorted(failures.items()))
                    + "."
                )
    for attempt in _failed_gpu_attempts(smoke=smoke):
        negatives.append(
            f"Failed GPU attempt {attempt['ledger_id']}: {attempt['rung'].upper()} "
            f"seed {attempt['seed']}, {float(attempt['gpu_hours']):.3f} GPU-hours, "
            f"${float(attempt['usd']):.3f}, exit code {attempt['exit_code']}."
        )
    if negatives:
        lines.extend(f"- {item}" for item in negatives)
    else:
        lines.append(
            "No full-run negative result is available yet; none is inferred from smoke runs."
        )

    lines.extend(["", "## Export", ""])
    if smoke:
        export_root = REPO_ROOT / "data" / "smoke" / "phase3" / "export" / "r4"
        export_receipts = sorted(export_root.glob("*/s*/export_manifest.json"))
    else:
        export_root = REPO_ROOT / "checkpoints" / "full" / "r4"
        export_receipts = sorted(export_root.glob("*/s*/export/export_manifest.json"))
    if not export_receipts:
        if smoke:
            lines.append("The smoke-only adapter packing rehearsal is pending.")
        else:
            lines.append(
                "Merged BF16 and deployment GPTQ-int4 hashes are pending a human-launched R4 "
                "export."
            )
    else:
        for path in export_receipts:
            receipt = json.loads(path.read_text())
            if smoke:
                lines.append(
                    f"- SMOKE ONLY, {receipt['backend']} seed {receipt['seed']}: adapter input "
                    f"`{receipt['full_precision_export']['sha256']}`; synthetic int4 packing "
                    f"`{receipt['deployment_int4_export']['sha256']}`. Neither is a deployable "
                    "model export."
                )
            else:
                lines.append(
                    f"- {receipt['backend']} seed {receipt['seed']}: merged BF16 "
                    f"`{receipt['full_precision_export']['sha256']}`; GPTQ int4 "
                    f"`{receipt['deployment_int4_export']['sha256']}`."
                )

    lines.extend(["", "## Draft headline", ""])
    preferred_backend = "trl"
    policy_path = REPO_ROOT / "results" / "phase3_backend_policy.json"
    if policy_path.is_file():
        preferred_backend = json.loads(policy_path.read_text()).get("default_backend", "trl")
    headline_records = [
        record
        for record in by_rung["r4"]
        if record.get("backend", "trl") == preferred_backend and int(record["seed"]) in {0, 1, 2}
    ]
    if {int(record["seed"]) for record in headline_records} != {0, 1, 2}:
        lines.append("Withheld until the full ladder, three R4 seeds, CIs, and costs are recorded.")
    else:
        task_success = [float(record["metrics"]["task_success"]) for record in headline_records]
        gpu_hours = sum(float(record["cost"]["gpu_hours"]) for record in headline_records)
        usd = sum(float(record["cost"]["usd"]) for record in headline_records)
        lines.append(
            f"Draft: verifier-reward GRPO reached {_fmt_percent(sum(task_success) / 3)} "
            f"mean task success across three seeds (range "
            f"{_fmt_percent(min(task_success))}–{_fmt_percent(max(task_success))}) using "
            f"{gpu_hours:.3f} measured RTX4090 GPU-hours (${usd:.3f}) for the R4 runs."
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "python -m forge.train.report",
            "```",
            "",
            "All task-success intervals use 1,000 fixed-seed bootstrap resamples. Field F1 "
            "entries are single-label micro-F1 (equivalent to exact-match accuracy). Urgency "
            "ground truth is the frozen rule policy, not human semantic judgment.",
            "",
        ]
    )
    return "\n".join(lines)


def generate(*, smoke: bool) -> tuple[Path, Path]:
    records = _records(smoke=smoke)
    by_rung = _rung_records(records)
    deltas = paired_deltas(by_rung, smoke=smoke)
    if smoke:
        output_dir = REPO_ROOT / "data" / "smoke" / "phase3"
        report_path = output_dir / "phase3_report.md"
        delta_path = output_dir / "phase3_paired_deltas.json"
    else:
        report_path = REPO_ROOT / "results" / "phase3_report.md"
        delta_path = REPO_ROOT / "results" / "phase3_paired_deltas.json"
    write_json_atomic(delta_path, deltas)
    write_text_atomic(report_path, _report_text(by_rung, deltas, smoke=smoke))
    print(f"Phase 3 report regenerated: {report_path.relative_to(REPO_ROOT)}")
    return report_path, delta_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    generate(smoke=args.smoke)


if __name__ == "__main__":
    main()
