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
    checkpoint_root,
    evaluation_root,
    load_config,
    runs_path,
    smoke_output_root,
)
from forge.train.evaluate import bootstrap_ci
from forge.train.export import durable_export_manifest_path
from forge.train.ledger import billable_records


def _run_statuses() -> dict[str, dict[str, Any]]:
    path = REPO_ROOT / "results" / "phase3_run_status.jsonl"
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    statuses = {str(row["run_id"]): row for row in rows}
    if len(statuses) != len(rows):
        raise ValueError("duplicate Phase 3 run status entry")
    return statuses


def _records(*, smoke: bool) -> list[dict[str, Any]]:
    path = runs_path(smoke=smoke)
    if not path.is_file():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    statuses = _run_statuses() if not smoke else {}
    return [
        {
            **record,
            "_interpretation_status": statuses.get(str(record.get("run_id")), {}).get(
                "status", "active"
            ),
        }
        for record in records
        if record.get("phase") == 3
    ]


def _active_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("_interpretation_status") == "active"]


def _failed_gpu_attempts(*, smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return []
    path = REPO_ROOT / "results" / "phase3_gpu_ledger.jsonl"
    if not path.is_file():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [record for record in billable_records(records) if record["status"] == "failed"]


def _r4_reward_signal_diagnostic(*, smoke: bool) -> dict[str, Any] | None:
    if smoke:
        return None
    path = REPO_ROOT / "results" / "phase3_r4_reward_signal_diagnostic.json"
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "aborted-zero-reward-variance":
        raise ValueError(f"unexpected R4 reward-signal diagnostic status in {path}")
    return receipt


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
    candidates = [record for record in _active_records(records) if int(record["seed"]) == 0]
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

    def summarize_records(
        left: str,
        right: str,
        *,
        left_record: dict[str, Any] | None,
        right_record: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if left_record is None or right_record is None:
            return {"from": left, "to": right, "status": "pending"}
        left_backend = str(left_record.get("backend", "trl"))
        right_backend = str(right_record.get("backend", "trl"))
        left_seed = int(left_record["seed"])
        right_seed = int(right_record["seed"])
        left_path = (
            evaluation_root(configs[left], seed=left_seed, smoke=smoke, backend=left_backend)
            / "predictions.jsonl"
        )
        right_path = (
            evaluation_root(configs[right], seed=right_seed, smoke=smoke, backend=right_backend)
            / "predictions.jsonl"
        )
        if not left_path.is_file() or not right_path.is_file():
            return {
                "from": left,
                "to": right,
                "to_seed": right_seed,
                "status": "predictions_missing",
            }
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
            "from_seed": left_seed,
            "to_seed": right_seed,
            "status": "complete",
            "paired_rows": len(deltas),
            "mean_task_success_delta": sum(deltas) / len(deltas),
            "ci95": ci,
            "from_backend": left_backend,
            "to_backend": right_backend,
            "bootstrap_resamples": 1000,
        }

    def summarize_pair(left: str, right: str) -> dict[str, Any]:
        return summarize_records(
            left,
            right,
            left_record=_seed_zero_record(by_rung[left]),
            right_record=_seed_zero_record(by_rung[right]),
        )

    current_r4_hash = configs["r4"]["_config_hash"]
    current_r4 = {
        int(record["seed"]): record
        for record in _active_records(by_rung["r4"])
        if record.get("backend", "trl") == "trl" and record.get("config_hash") == current_r4_hash
    }
    r3 = _seed_zero_record(by_rung["r3"])
    signal_receipts = _r4_v2_signal_receipts(smoke=smoke)
    r4_seed_deltas = []
    for seed in (0, 1, 2):
        delta = {
            **summarize_records(
                "r3",
                "r4",
                left_record=r3,
                right_record=current_r4.get(seed),
            ),
            "to_seed": seed,
        }
        receipt = signal_receipts.get(seed)
        if (
            delta["status"] == "pending"
            and receipt is not None
            and receipt.get("status") == "aborted-zero-reward-variance"
        ):
            delta = {
                "from": "r3",
                "to": "r4",
                "to_seed": seed,
                "status": "aborted-zero-reward-variance",
                "paired_rows": 0,
                "guard_status": receipt["status"],
                "reason": receipt.get("error", {}).get("message"),
            }
        r4_seed_deltas.append(delta)
    aggregate: dict[str, Any] = {"from": "r3", "to": "r4", "status": "pending"}
    aborted_seeds = [
        int(item["to_seed"])
        for item in r4_seed_deltas
        if item["status"] == "aborted-zero-reward-variance"
    ]
    if aborted_seeds:
        aggregate = {
            "from": "r3",
            "to": "r4",
            "status": "aborted-zero-reward-variance",
            "verdict": "aborted",
            "aborted_seeds": aborted_seeds,
            "completed_seeds": [
                int(item["to_seed"]) for item in r4_seed_deltas if item["status"] == "complete"
            ],
            "reason": (
                "The unchanged ten-step reward-variance guard stopped R4 v2; no "
                "three-seed aggregate or missing paired delta is fabricated."
            ),
        }
    elif all(item["status"] == "complete" for item in r4_seed_deltas) and r3 is not None:
        r3_backend = str(r3.get("backend", "trl"))
        r3_values = _prediction_map(
            evaluation_root(configs["r3"], seed=int(r3["seed"]), smoke=smoke, backend=r3_backend)
            / "predictions.jsonl"
        )
        seed_values = []
        for seed in (0, 1, 2):
            path = (
                evaluation_root(configs["r4"], seed=seed, smoke=smoke, backend="trl")
                / "predictions.jsonl"
            )
            values = _prediction_map(path)
            if values.keys() != r3_values.keys():
                raise ValueError(f"R3->R4 seed {seed} does not share the frozen eval rows")
            seed_values.append(values)
        deltas = [
            sum(values[key] for values in seed_values) / len(seed_values) - r3_values[key]
            for key in sorted(r3_values)
        ]
        ci = bootstrap_ci(
            deltas,
            resamples=int(configs["r4"]["evaluation"]["bootstrap_resamples"]),
            seed=int(configs["r4"]["evaluation"]["bootstrap_seed"]),
        )
        verdict = "win" if ci[0] > 0.0 else "loss" if ci[1] < 0.0 else "tie"
        aggregate = {
            "from": "r3",
            "to": "r4",
            "status": "complete",
            "seeds": [0, 1, 2],
            "paired_rows": len(deltas),
            "mean_task_success_delta": sum(deltas) / len(deltas),
            "ci95": ci,
            "bootstrap_resamples": 1000,
            "verdict": verdict,
            "verdict_rule": (
                "win if CI lower bound > 0; loss if CI upper bound < 0; otherwise tie"
            ),
        }

    return {
        "phase": 3,
        "mode": "smoke" if smoke else "full",
        "adjacent_pairs": [summarize_pair(*pair) for pair in adjacent_pairs],
        "optional_ablation_pairs": [summarize_pair(*pair) for pair in ablation_pairs],
        "r4_v2_seed_deltas": r4_seed_deltas,
        "r4_v2_aggregate": aggregate,
    }


def _fmt_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _export_manifest_path(record: dict[str, Any], *, smoke: bool) -> Path:
    config = load_config(record["config_path"])
    backend = str(record.get("backend", "trl"))
    seed = int(record["seed"])
    if smoke:
        root = smoke_output_root() / "export" / str(config["rung"])
        if config.get("run_revision"):
            root /= str(config["run_revision"])
        return root / backend / f"s{seed}" / "export_manifest.json"
    durable = durable_export_manifest_path(config, seed=seed, backend=backend)
    if durable.is_file():
        return durable
    return (
        checkpoint_root(config, seed=seed, smoke=False, backend=backend)
        / "export"
        / "export_manifest.json"
    )


def _best_r4_record(
    records: list[dict[str, Any]], *, eligible_seeds: set[int] | None = None
) -> dict[str, Any] | None:
    current = load_config("configs/r4_grpo.yaml")
    candidates = [
        record
        for record in _active_records(records)
        if record.get("backend", "trl") == "trl"
        and int(record["seed"]) in {0, 1, 2}
        and record.get("config_hash") == current["_config_hash"]
        and (eligible_seeds is None or int(record["seed"]) in eligible_seeds)
    ]
    required = {0, 1, 2} if eligible_seeds is None else eligible_seeds
    if {int(record["seed"]) for record in candidates} != required:
        return None
    return max(
        candidates,
        key=lambda record: (
            float(record["metrics"]["task_success"]),
            float(record["metrics"]["mean_reward"]),
            -int(record["seed"]),
        ),
    )


def _r4_v2_signal_receipts(*, smoke: bool) -> dict[int, dict[str, Any]]:
    if smoke:
        return {}
    config = load_config("configs/r4_grpo.yaml")
    revision = str(config.get("run_revision", "unversioned"))
    receipts = {}
    for seed in (0, 1, 2):
        path = REPO_ROOT / "results" / f"phase3_r4_reward_signal_{revision}_s{seed}.json"
        if path.is_file():
            receipts[seed] = json.loads(path.read_text())
    return receipts


def _export_selection(
    by_rung: dict[str, list[dict[str, Any]]],
    deltas: dict[str, Any],
    *,
    smoke: bool,
) -> dict[str, Any]:
    r1b = _seed_zero_record(by_rung["r1b"])
    serving: dict[str, Any] = {"status": "pending", "rung": "r1b", "seed": 0}
    if r1b is not None:
        path = _export_manifest_path(r1b, smoke=smoke)
        serving.update({"run_id": r1b["run_id"], "manifest_path": str(path.relative_to(REPO_ROOT))})
        if path.is_file():
            receipt = json.loads(path.read_text())
            serving.update(
                {
                    "status": "complete",
                    "merged_bf16_sha256": receipt["full_precision_export"]["sha256"],
                    "gptq_int4_sha256": receipt["deployment_int4_export"]["sha256"],
                }
            )
    r4_contract: dict[str, Any] = {
        "status": "pending-r4-v2",
        "selection_metric": "task_success_then_mean_reward_then_lowest_seed",
        "eligibility_rule": "paired delta versus R3 has CI95 lower bound greater than zero",
        "required_seeds": [0, 1, 2],
        "paired_deltas": deltas["r4_v2_seed_deltas"],
    }
    complete_deltas = [item for item in deltas["r4_v2_seed_deltas"] if item["status"] == "complete"]
    signal_receipts = _r4_v2_signal_receipts(smoke=smoke)
    failed_signals = {
        seed: receipt
        for seed, receipt in signal_receipts.items()
        if receipt.get("status") != "passed-nonzero-reward-variance"
    }
    if failed_signals:
        r4_contract.update(
            {
                "status": "aborted-r4-v2-guard",
                "failed_signal_seeds": sorted(failed_signals),
            }
        )
    elif len(complete_deltas) == 3:
        eligible_seeds = {
            int(item["to_seed"]) for item in complete_deltas if float(item["ci95"][0]) > 0.0
        }
        if eligible_seeds:
            r4_best = _best_r4_record(by_rung["r4"], eligible_seeds=eligible_seeds)
            if r4_best is None:
                raise RuntimeError("R4 v2 eligible seeds do not have matching active records")
            selected_delta = next(
                item for item in complete_deltas if int(item["to_seed"]) == int(r4_best["seed"])
            )
            r4_contract.update(
                {
                    "status": "eligible-ci-significant-win",
                    "run_id": r4_best["run_id"],
                    "seed": int(r4_best["seed"]),
                    "task_success": float(r4_best["metrics"]["task_success"]),
                    "mean_reward": float(r4_best["metrics"]["mean_reward"]),
                    "paired_delta": float(selected_delta["mean_task_success_delta"]),
                    "paired_ci95": list(selected_delta["ci95"]),
                }
            )
        else:
            r4_contract["status"] = "not-eligible-no-ci-significant-win"
    diagnostic = _r4_reward_signal_diagnostic(smoke=smoke)
    if diagnostic is not None:
        r4_contract["historical_phase3_1_diagnostic"] = (
            "results/phase3_r4_reward_signal_diagnostic.json"
        )
    return {
        "version": 1,
        "phase": 3,
        "mode": "smoke" if smoke else "full",
        "phase4_serving_artifact": serving,
        "r4_best_seed_export_contract": r4_contract,
    }


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
            "| Rung | Seed | Backend | Status | Task success | 95% CI | Schema valid | "
            "Tool accuracy | GPU hours | USD |",
            "|---|---:|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for rung in ALL_RUNGS:
        values = by_rung[rung]
        if not values:
            label = "optional; pending" if rung == "r1b" else "pending"
            lines.append(f"| {rung.upper()} | — | — | {label} | — | — | — | — | — | — |")
            continue
        for record in values:
            metrics = record["metrics"]
            ci = metrics["ci95"]
            lines.append(
                f"| {rung.upper()} | {record['seed']} | {record.get('backend', 'trl')} | "
                f"{record.get('_interpretation_status', 'active')} | "
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

    lines.extend(["", "## R4 v2 fresh-pool verdict", ""])
    for item in deltas["r4_v2_seed_deltas"]:
        seed = int(item["to_seed"])
        if item["status"] == "aborted-zero-reward-variance":
            lines.append(
                f"- Seed {seed}: aborted by the locked ten-step reward-variance guard; "
                "no frozen evaluation or paired delta exists."
            )
        elif item["status"] != "complete":
            lines.append(f"- Seed {seed}: {item['status']}.")
        else:
            ci = item["ci95"]
            lines.append(
                f"- Seed {seed} vs R3: "
                f"{_fmt_percent(float(item['mean_task_success_delta']))} paired delta, "
                f"95% CI [{_fmt_percent(float(ci[0]))}, {_fmt_percent(float(ci[1]))}], "
                f"n={item['paired_rows']}."
            )
    aggregate = deltas["r4_v2_aggregate"]
    if aggregate["status"] == "aborted-zero-reward-variance":
        seeds = ", ".join(str(seed) for seed in aggregate["aborted_seeds"])
        lines.append(
            "- Final R4 v2 verdict: **ABORTED BY LOCKED GUARD**; "
            f"seed(s) {seeds} stopped before frozen evaluation. The completed seed deltas "
            "are retained but not aggregated, and no retry or reward tuning is authorized."
        )
    elif aggregate["status"] != "complete":
        lines.append("- Final R4 v2 verdict: pending all three frozen-eval records.")
    else:
        ci = aggregate["ci95"]
        lines.append(
            f"- Final R4 v2 verdict: **{str(aggregate['verdict']).upper()}**; "
            f"three-seed mean paired delta "
            f"{_fmt_percent(float(aggregate['mean_task_success_delta']))}, 95% CI "
            f"[{_fmt_percent(float(ci[0]))}, {_fmt_percent(float(ci[1]))}]."
        )
    signal_receipts = _r4_v2_signal_receipts(smoke=smoke)
    if smoke:
        lines.append("- Ten-step reward-variance gate: full-run evidence only; smoke is exempt.")
    elif not signal_receipts:
        lines.append("- Ten-step reward-variance gate: pending the delegated GPU runs.")
    else:
        for seed in sorted(signal_receipts):
            receipt = signal_receipts[seed]
            observed = receipt.get("guard", {}).get("observed", {})
            lines.append(
                f"- Seed {seed} reward-variance gate: `{receipt.get('status')}`; "
                f"opening observations={len(observed)}, "
                f"passed={str(bool(receipt.get('passed_opening_steps'))).lower()}."
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
    reward_diagnostic = _r4_reward_signal_diagnostic(smoke=smoke)
    if r4 is None:
        lines.append(
            "R4 v2 probes are pending the fresh-pool GRPO runs and frozen evaluation. "
            "The Phase 3.1 saturation diagnostic remains historical incident evidence."
        )
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
    negatives = [
        "GRPO incident: R4 seeds 0/1/2 from the original config are "
        'superseded-inconclusive. The missing `chat_template_kwargs={"enable_thinking": '
        "False}` left a trailing `</think>` prefix in rollouts; bare `json.loads` then "
        "returned reward 0.0 for every completion, producing zero reward variance and zero "
        "gradient. The fixed rerun disables thinking at chat-template rendering, strips through "
        "a trailing think block defensively, archives an opening rollout, and aborts after ten "
        "all-zero-variance steps."
    ]
    if reward_diagnostic is not None:
        window = reward_diagnostic["logged_rollout_window"]
        negatives.append(
            "Phase 3.1 guard result: seed 0 proved the parser repair worked (clean bare JSON, "
            "no think marker, positive verifier reward), but the first ten optimizer steps "
            "all had mean reward 1.0, reward_std 0.0, frac_reward_zero_std 1.0, and grad_norm "
            "0.0. In the archived opening window, all "
            f"{window['completion_rows']} completions across {window['prompt_groups']} prompt "
            "groups received reward 1.0 and advantage 0.0; observed variation was confined "
            "to secondary fields excluded from scorer-v2 reward. The guard aborted the run, "
            "and seeds 1/2 were not launched rather than bypassing the locked reward/data "
            "contract."
        )
    for seed, receipt in sorted(_r4_v2_signal_receipts(smoke=smoke).items()):
        if receipt.get("status") != "aborted-zero-reward-variance":
            continue
        observed = receipt["guard"]["observed"]
        reward_audit = receipt["reward_audit"]
        negatives.append(
            f"Phase 3.2 guard result: seed {seed} stopped after "
            f"{len(observed)} opening steps all had zero within-group reward variance "
            f"(mean verifier reward {float(reward_audit['mean_reward']):.3f} across "
            f"{int(reward_audit['completions_scored'])} completions). Per the locked plan, "
            "there is no retry, frozen-eval record, paired delta, or three-seed aggregate."
        )
    comparison_results = deltas["adjacent_pairs"] + deltas["optional_ablation_pairs"]
    for item in comparison_results:
        if item.get("status") == "complete" and float(item["mean_task_success_delta"]) < 0:
            loss = _fmt_percent(abs(float(item["mean_task_success_delta"])))
            negatives.append(
                f"{item['to'].upper()} lost {loss} task success versus {item['from'].upper()}."
            )
    for rung, records in by_rung.items():
        for record in _active_records(records):
            failures = record["metrics"].get("failure_categories_nonexclusive", {})
            if failures:
                negatives.append(
                    f"{rung.upper()} {record.get('backend', 'trl')} seed {record['seed']} "
                    "failure counts (nonexclusive): "
                    + ", ".join(f"{name}={count}" for name, count in sorted(failures.items()))
                    + "."
                )
    for attempt in _failed_gpu_attempts(smoke=smoke):
        negatives.append(
            f"Failed GPU {attempt.get('operation', 'training')} attempt "
            f"{attempt['ledger_id']}: {attempt['rung'].upper()} "
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
    export_rows = []
    for rung in ("r1b", "r4"):
        for record in _active_records(by_rung[rung]):
            path = _export_manifest_path(record, smoke=smoke)
            if path.is_file():
                export_rows.append(json.loads(path.read_text()))
    if not export_rows:
        lines.append(
            "R1b merged BF16 and deployment GPTQ-int4 hashes are pending the authorized "
            "Phase 3.1 export."
        )
    for receipt in export_rows:
        if smoke:
            lines.append(
                f"- SMOKE ONLY, {str(receipt['rung']).upper()} {receipt['backend']} seed "
                f"{receipt['seed']}: adapter input "
                f"`{receipt['full_precision_export']['sha256']}`; synthetic int4 packing "
                f"`{receipt['deployment_int4_export']['sha256']}`. Neither is a deployable "
                "model export."
            )
        else:
            lines.append(
                f"- {str(receipt['rung']).upper()} {receipt['backend']} seed "
                f"{receipt['seed']}: merged BF16 "
                f"`{receipt['full_precision_export']['sha256']}`; GPTQ int4 "
                f"`{receipt['deployment_int4_export']['sha256']}`."
            )

    lines.extend(["", "## R4 best-seed export contract", ""])
    r4_contract = _export_selection(by_rung, deltas, smoke=smoke)["r4_best_seed_export_contract"]
    if r4_contract["status"] == "eligible-ci-significant-win":
        lines.append(
            f"Eligible: `{r4_contract['run_id']}` (seed {r4_contract['seed']}) beat R3 with "
            f"paired 95% CI [{_fmt_percent(float(r4_contract['paired_ci95'][0]))}, "
            f"{_fmt_percent(float(r4_contract['paired_ci95'][1]))}]. The contract permits a "
            "later export; Phase 3.2 does not perform it."
        )
    elif r4_contract["status"] == "not-eligible-no-ci-significant-win":
        lines.append(
            "No seed beat R3 with a paired 95% CI excluding zero; no R4 export contract is "
            "opened. R1b remains the Phase 4 serving artifact."
        )
    elif r4_contract["status"] == "aborted-r4-v2-guard":
        lines.append(
            "R4 v2 stopped at the unchanged reward-variance guard; no export contract is "
            "opened and no tuning is authorized."
        )
    else:
        lines.append(
            "Pending all three Phase 3.2 TRL seeds and their paired deltas versus R3. The "
            "Phase 3.1 saturation abort remains documented but is no longer the active contract."
        )

    lines.extend(["", "## Interpretation guards", ""])
    lines.extend(
        [
            "R1 and R2 use the same 1,450 examples and identical decision-field labels after "
            "rejection sampling. Their difference is output phrasing, not label coverage.",
            "",
            "The R2 loss therefore indicates that the teacher's semantic phrasing transferred a "
            "policy prior that diverged from the frozen keyword rules on new inputs; before "
            "filtering, teacher/rule urgency agreement was 36.8%.",
            "",
            "Boundary condition: this project shows distillation adds no value when perfect "
            "rule-generated labels are free and unlimited. It does not generalize to settings "
            "where gold labels are scarce and no executable labeling rules exist; there, teacher "
            "quality is decisive. A more semantic Sonnet-class teacher would be expected to "
            "diverge further from this keyword policy, not close the measured gap.",
        ]
    )

    lines.extend(["", "## Draft headline", ""])
    r1 = _seed_zero_record(by_rung["r1"])
    r1b = _seed_zero_record(by_rung["r1b"])
    r1b_delta = next(
        (
            item
            for item in deltas["optional_ablation_pairs"]
            if item["from"] == "r1" and item["to"] == "r1b" and item["status"] == "complete"
        ),
        None,
    )
    if r1 is None or r1b is None or r1b_delta is None:
        lines.append("Withheld until the R1b cost-quality comparison is fully recorded.")
    else:
        ci = r1b_delta["ci95"]
        lines.append(
            f"Draft: scaling free rule labels from 1,450 to 20,000 examples raised task "
            f"success from {_fmt_percent(float(r1['metrics']['task_success']))} to "
            f"{_fmt_percent(float(r1b['metrics']['task_success']))} "
            f"(+{_fmt_percent(float(r1b_delta['mean_task_success_delta']))}, paired 95% CI "
            f"[{_fmt_percent(float(ci[0]))}, {_fmt_percent(float(ci[1]))}]) for "
            f"{float(r1b['cost']['gpu_hours']):.3f} measured RTX4090 GPU-hours "
            f"(${float(r1b['cost']['usd']):.3f})."
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
        output_dir = smoke_output_root()
        report_path = output_dir / "phase3_report.md"
        delta_path = output_dir / "phase3_paired_deltas.json"
        selection_path = output_dir / "phase3_export_selection.json"
    else:
        report_path = REPO_ROOT / "results" / "phase3_report.md"
        delta_path = REPO_ROOT / "results" / "phase3_paired_deltas.json"
        selection_path = REPO_ROOT / "results" / "phase3_export_selection.json"
    write_json_atomic(delta_path, deltas)
    write_json_atomic(selection_path, _export_selection(by_rung, deltas, smoke=smoke))
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
