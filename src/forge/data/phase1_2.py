"""Phase 1.2 label-rules-v3 materialization and receipt-only calibration rescore.

This module is deliberately network-free. It reads the immutable Phase 1 and
Phase 1.1 artifacts, writes a new v3 dataset beside them, emits a reviewer audit,
and scores the already-recorded 100 model outputs against the new gold labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import yaml

from forge.data.ingest import ROW_GROUP_SIZE, sha256_file
from forge.data.labels import DEFAULT_RULES_PATH, canonical_label_json, load_rules
from forge.data.relabel import membership_sha256
from forge.data.splits import AUDIT_ROWS_PER_SPLIT, AUDIT_SEED, DISPLAY_NAMES, SPLITS
from forge.verify.schema import TASK_SCHEMA
from forge.verify.verifier import SCORER_VERSION, ScoreBreakdown, score

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE1_SPLIT_DIR = REPO_ROOT / "data" / "splits"
PHASE1_MANIFEST_PATH = PHASE1_SPLIT_DIR / "manifest.json"
PHASE1_1_SPLIT_DIR = REPO_ROOT / "data" / "phase1_1" / "splits"
PHASE1_1_MANIFEST_PATH = REPO_ROOT / "data" / "phase1_1" / "manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "phase1_2" / "splits"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "phase1_2" / "manifest.json"
DEFAULT_AUDIT_PATH = REPO_ROOT / "results" / "phase1_2_label_audit.md"
DEFAULT_RESCORE_REPORT_PATH = REPO_ROOT / "results" / "phase1_2_calibration_rescore.md"
DEFAULT_RECEIPTS_PATH = REPO_ROOT / "results" / "phase1_1_api_calibration_receipts.jsonl"
DEFAULT_CALIBRATION_LEDGER_PATH = REPO_ROOT / "results" / "phase1_1_api_calibration_ledger.json"
DEFAULT_CALIBRATION_CONFIG_PATH = REPO_ROOT / "configs" / "difficulty_candidates.yaml"
CALIBRATION_RUN_ID = "phase1_2_api_calibration_v3_offline_rescore_s20260815"
STRONG_ACTION_TOOLS = frozenset({"escalate_to_regulator", "start_refund_workflow"})
STRONG_ACTION_AUDIT_CAP = 50


@dataclass(frozen=True)
class ChangedRow:
    """One v2-to-v3 label change with its immutable source evidence."""

    split: str
    complaint_id: int
    date_received: Any
    source_product: str | None
    source_issue: str | None
    source_company: str | None
    narrative: str | None
    old_label: dict[str, Any]
    new_label: dict[str, Any]

    @property
    def old_tool(self) -> str:
        return str(self.old_label["tool_call"]["name"])

    @property
    def new_tool(self) -> str:
        return str(self.new_label["tool_call"]["name"])

    @property
    def transition(self) -> str:
        return f"{self.old_tool} -> {self.new_tool}"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _quoted(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _new_temp_path(parent: Path, suffix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".phase1-2-", suffix=suffix, dir=parent)
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = _new_temp_path(path.parent, path.suffix or ".tmp")
    try:
        temporary.write_text(value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, value: object) -> None:
    _write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _split_paths(split_dir: Path) -> dict[str, Path]:
    return {name: split_dir / f"{name}.parquet" for name in SPLITS}


def _verify_source_artifacts(
    *,
    phase1_split_dir: Path,
    phase1_manifest_path: Path,
    phase1_1_split_dir: Path,
    phase1_1_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, str]]:
    """Verify v1 payloads and v2 payload/membership before deriving v3."""

    if not phase1_manifest_path.is_file():
        raise FileNotFoundError(f"Phase 1 manifest is missing: {phase1_manifest_path}")
    if not phase1_1_manifest_path.is_file():
        raise FileNotFoundError(f"Phase 1.1 manifest is missing: {phase1_1_manifest_path}")
    phase1_manifest = json.loads(phase1_manifest_path.read_text())
    phase1_1_manifest = json.loads(phase1_1_manifest_path.read_text())
    if phase1_manifest.get("version") != 1 or phase1_manifest.get("frozen") is not True:
        raise ValueError("Phase 1 source manifest must be frozen version 1")
    if phase1_1_manifest.get("version") != 2:
        raise ValueError("Phase 1.2 requires the version-2 Phase 1.1 dataset")
    if phase1_1_manifest.get("label_rules", {}).get("version") != 2:
        raise ValueError("Phase 1.1 source must carry label rules version 2")
    if phase1_1_manifest.get("protocol", {}).get("phase2_data_generation_started") is not False:
        raise ValueError("label re-derivation is forbidden after Phase 2 data generation starts")
    expected_phase1_manifest_hash = phase1_1_manifest.get("phase1_source", {}).get(
        "manifest_sha256"
    )
    if sha256_file(phase1_manifest_path) != expected_phase1_manifest_hash:
        raise ValueError("Phase 1.1 manifest points to a different frozen Phase 1 manifest")

    phase1_paths = _split_paths(phase1_split_dir)
    source_paths = _split_paths(phase1_1_split_dir)
    phase1_membership: dict[str, str] = {}
    for name in SPLITS:
        phase1_path = phase1_paths[name]
        source_path = source_paths[name]
        if not phase1_path.is_file() or not source_path.is_file():
            raise FileNotFoundError(f"required split payload is missing for {name}")
        if sha256_file(phase1_path) != phase1_manifest["splits"][name]["sha256"]:
            raise ValueError(f"frozen Phase 1 payload changed for {name}")
        if sha256_file(source_path) != phase1_1_manifest["splits"][name]["payload_sha256"]:
            raise ValueError(f"immutable Phase 1.1 payload changed for {name}")
        v1_membership = membership_sha256(phase1_path)
        v2_membership = membership_sha256(source_path)
        declared_membership = phase1_1_manifest["splits"][name]["membership_sha256"]
        if not (v1_membership == v2_membership == declared_membership):
            raise ValueError(f"Phase 1.1 membership differs from frozen Phase 1 for {name}")
        phase1_membership[name] = v1_membership
    return phase1_1_manifest, source_paths, phase1_membership


def _materialize_split(source_path: Path, output_path: Path, rules_path: Path) -> None:
    rules = load_rules(rules_path)

    def label_json(
        product: str | None,
        issue: str | None,
        company: str | None,
        narrative: str | None,
        complaint_id: int,
    ) -> str:
        return canonical_label_json(
            {
                "complaint_id": complaint_id,
                "product": product,
                "issue": issue,
                "company": company,
                "narrative": narrative,
            },
            rules,
        )

    temporary = _new_temp_path(output_path.parent, ".parquet")
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")
        con.create_function(
            "phase1_2_label_json",
            label_json,
            return_type="VARCHAR",
            null_handling="special",
        )
        con.execute(
            f"""
            COPY (
                SELECT complaint_id, date_received, narrative, source_product,
                       source_issue, source_company, source_class, source_split,
                       phase1_2_label_json(
                           source_product, source_issue, source_company, narrative,
                           complaint_id
                       ) AS label_json
                FROM read_parquet({_quoted(source_path)})
                ORDER BY complaint_id, source_split
            ) TO {_quoted(temporary)}
            (FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE {ROW_GROUP_SIZE})
            """
        )
        os.replace(temporary, output_path)
    finally:
        con.close()
        if temporary.exists():
            temporary.unlink()


def _split_stats(source_path: Path, output_path: Path, phase1_membership: str) -> dict[str, Any]:
    source_membership = membership_sha256(source_path)
    output_membership = membership_sha256(output_path)
    con = duckdb.connect()
    try:
        rows, unique_ids = con.execute(
            "SELECT count(*), count(DISTINCT complaint_id) FROM read_parquet(?)",
            [str(output_path)],
        ).fetchone()
        changed = int(
            con.execute(
                """
                SELECT count(*)
                FROM read_parquet(?) AS old
                JOIN read_parquet(?) AS new USING (complaint_id)
                WHERE old.label_json <> new.label_json
                """,
                [str(source_path), str(output_path)],
            ).fetchone()[0]
        )
    finally:
        con.close()
    if rows != unique_ids:
        raise ValueError(f"duplicate complaint_id in v3 split: {output_path}")
    if not (source_membership == output_membership == phase1_membership):
        raise ValueError(f"split membership changed during v3 derivation: {output_path}")
    return {
        "path": str(output_path.resolve()),
        "rows": int(rows),
        "unique_complaint_ids": int(unique_ids),
        "payload_sha256": sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
        "membership_sha256": output_membership,
        "phase1_membership_sha256": phase1_membership,
        "phase1_1_membership_sha256": source_membership,
        "membership_matches_phase1": True,
        "labels_changed_from_v2": changed,
    }


def _changed_rows(split: str, source_path: Path, output_path: Path) -> list[ChangedRow]:
    con = duckdb.connect()
    try:
        rows = con.execute(
            """
            SELECT old.complaint_id, old.date_received, old.source_product,
                   old.source_issue, old.source_company, old.narrative,
                   old.label_json, new.label_json
            FROM read_parquet(?) AS old
            JOIN read_parquet(?) AS new USING (complaint_id)
            WHERE old.label_json <> new.label_json
            ORDER BY old.complaint_id
            """,
            [str(source_path), str(output_path)],
        ).fetchall()
    finally:
        con.close()
    return [
        ChangedRow(
            split=split,
            complaint_id=int(row[0]),
            date_received=row[1],
            source_product=row[2],
            source_issue=row[3],
            source_company=row[4],
            narrative=row[5],
            old_label=json.loads(row[6]),
            new_label=json.loads(row[7]),
        )
        for row in rows
    ]


def _rank(complaint_id: int) -> bytes:
    return hashlib.blake2b(f"{AUDIT_SEED}:{complaint_id}".encode(), digest_size=16).digest()


def _balanced_transition_quotas(grouped: dict[str, list[ChangedRow]], cap: int) -> dict[str, int]:
    """Allocate the cap evenly across action transitions, exhausting rare ones first."""

    quotas = {key: 0 for key in grouped}
    remaining = min(cap, sum(len(rows) for rows in grouped.values()))
    keys = sorted(grouped)
    while remaining:
        progressed = False
        for key in keys:
            if quotas[key] >= len(grouped[key]):
                continue
            quotas[key] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            break
    return quotas


def _round_robin_source_buckets(rows: list[ChangedRow], limit: int) -> list[ChangedRow]:
    """Cover frozen splits and source issues before taking a second row from either."""

    buckets: dict[tuple[str, str], list[ChangedRow]] = defaultdict(list)
    for row in rows:
        buckets[(row.split, row.source_issue or "<missing>")].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: row.complaint_id)
    selected: list[ChangedRow] = []
    offsets = {key: 0 for key in buckets}
    keys = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            offset = offsets[key]
            if offset >= len(buckets[key]):
                continue
            selected.append(buckets[key][offset])
            offsets[key] += 1
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    return selected


def _stratified_strong_action_sample(
    rows: list[ChangedRow], cap: int = STRONG_ACTION_AUDIT_CAP
) -> tuple[list[ChangedRow], list[ChangedRow]]:
    """Return the strong-action population and a transition/source-stratified sample."""

    population = [
        row
        for row in rows
        if row.old_tool in STRONG_ACTION_TOOLS or row.new_tool in STRONG_ACTION_TOOLS
    ]
    if len(population) <= cap:
        return population, sorted(population, key=lambda row: (row.transition, row.complaint_id))
    grouped: dict[str, list[ChangedRow]] = defaultdict(list)
    for row in population:
        grouped[row.transition].append(row)
    quotas = _balanced_transition_quotas(grouped, cap)
    selected: list[ChangedRow] = []
    for transition in sorted(grouped):
        selected.extend(_round_robin_source_buckets(grouped[transition], quotas[transition]))
    return population, selected


def _escape_cell(value: object, limit: int | None = None) -> str:
    text = "" if value is None else " ".join(str(value).split())
    if limit is not None and len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text.replace("|", "\\|")


def _date_text(value: object) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _audit_table(lines: list[str], rows: list[ChangedRow]) -> None:
    lines.extend(
        [
            (
                "| Split | ID | Date | Source issue | Old urgency | New urgency | "
                "Old tool | New tool | Narrative excerpt |"
            ),
            "|---|---:|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    DISPLAY_NAMES[row.split],
                    str(row.complaint_id),
                    _date_text(row.date_received),
                    _escape_cell(row.source_issue, 80),
                    str(row.old_label["urgency"]),
                    str(row.new_label["urgency"]),
                    row.old_tool,
                    row.new_tool,
                    _escape_cell(row.narrative, 190),
                )
            )
            + " |"
        )


def _write_changed_row_audit(
    source_paths: dict[str, Path], output_paths: dict[str, Path], audit_path: Path
) -> dict[str, Any]:
    all_changed = {
        name: _changed_rows(name, source_paths[name], output_paths[name]) for name in SPLITS
    }
    changed_rows = [row for name in SPLITS for row in all_changed[name]]
    totals = {name: len(rows) for name, rows in all_changed.items()}
    general_selected = {
        name: sorted(
            rows,
            key=lambda row: (_rank(row.complaint_id), row.complaint_id),
        )[:AUDIT_ROWS_PER_SPLIT]
        for name, rows in all_changed.items()
    }
    strong_population, strong_selected = _stratified_strong_action_sample(changed_rows)
    population_transitions = Counter(row.transition for row in strong_population)
    sample_transitions = Counter(row.transition for row in strong_selected)
    issue_population = Counter(row.source_issue or "<missing>" for row in strong_population)
    lines = [
        "# Phase 1.2 label-rules-v3 changed-row audit",
        "",
        (
            "Status: **HUMAN REVIEW REQUIRED**. Labels are compared against the immutable "
            "v2 dataset; split membership is unchanged."
        ),
        "",
        "## Scope and population",
        "",
        (
            "Label rules v3 match high-urgency escalation and refund keywords against the "
            "complaint narrative only. Source issue taxonomy strings no longer provide "
            "strong-action evidence."
        ),
        "",
        "Full changed-label counts: "
        + ", ".join(f"{DISPLAY_NAMES[name]}={totals[name]}" for name in SPLITS)
        + ".",
        "",
        "Strong-action transition population: "
        + ", ".join(f"{key}={value}" for key, value in sorted(population_transitions.items()))
        + ".",
        "",
        "Top source issues in the strong-action population: "
        + ", ".join(f"{issue}={count}" for issue, count in issue_population.most_common(12))
        + ".",
        "",
        "## Deterministic changed-row sample",
        "",
        (
            "Up to 50 changed rows per frozen split are ranked by "
            f"`blake2b('{AUDIT_SEED}:<complaint_id>')`, matching the prior audit protocol."
        ),
        "",
    ]
    for name in SPLITS:
        _audit_table(lines, general_selected[name])
        lines.append("")
    lines.extend(
        [
            "## Stratified strong-action sample",
            "",
            (
                f"Population={len(strong_population)}; emitted={len(strong_selected)}; "
                f"cap={STRONG_ACTION_AUDIT_CAP}. Every changed escalation/refund row is "
                "included when the population fits the cap. Otherwise the cap is balanced "
                "across action transitions, then round-robin across frozen split and source "
                "issue buckets. This section is not selected by global hash rank."
            ),
            "",
            "Sample transitions: "
            + ", ".join(f"{key}={value}" for key, value in sorted(sample_transitions.items()))
            + ".",
            "",
        ]
    )
    _audit_table(lines, strong_selected)
    lines.extend(
        [
            "",
            "## Known limitations intentionally left open",
            "",
            (
                "- Negation-blind matching remains. The delegated review estimated about 18 "
                "affected rows containing negated `identity theft`; v3 does not repair them."
            ),
            (
                "- The taxonomy remains single-action. Escalation outranks refund, so a "
                "dual-remedy narrative emits only `escalate_to_regulator`."
            ),
            "",
            "## Reviewer checklist",
            "",
            (
                "- [ ] Product/service taxonomy strings no longer cause escalation without "
                "narrative evidence."
            ),
            "- [ ] Every sampled refund action is supported by refund language in the narrative.",
            "- [ ] Every sampled post-fix tool choice follows the asserted priority order.",
            "- [ ] The two documented known limitations remain visible and unfixed.",
            "- [ ] Any further correction becomes a new label-rules version.",
            "",
        ]
    )
    _write_text_atomic(audit_path, "\n".join(lines))
    return {
        "selection_seed": AUDIT_SEED,
        "general_selection_cap": AUDIT_ROWS_PER_SPLIT * len(SPLITS),
        "general_rows_emitted": sum(len(rows) for rows in general_selected.values()),
        "full_changed_rows_by_split": totals,
        "strong_action_population": len(strong_population),
        "strong_action_cap": STRONG_ACTION_AUDIT_CAP,
        "strong_action_rows_emitted": len(strong_selected),
        "strong_action_population_by_transition": dict(sorted(population_transitions.items())),
        "strong_action_sample_by_transition": dict(sorted(sample_transitions.items())),
        "strong_action_selection": (
            "balanced action transitions; round-robin frozen split and source issue"
        ),
    }


def _verify_existing(
    manifest: dict[str, Any], fingerprint: str, output_dir: Path, audit_path: Path
) -> bool:
    if manifest.get("fingerprint") != fingerprint:
        return False
    for name in SPLITS:
        path = output_dir / f"{name}.parquet"
        expected = manifest.get("splits", {}).get(name, {}).get("payload_sha256")
        if not path.is_file() or sha256_file(path) != expected:
            return False
    audit = manifest.get("audit", {})
    return audit_path.is_file() and sha256_file(audit_path) == audit.get("sha256")


def run_phase1_2_labels(
    *,
    phase1_split_dir: Path = PHASE1_SPLIT_DIR,
    phase1_manifest_path: Path = PHASE1_MANIFEST_PATH,
    phase1_1_split_dir: Path = PHASE1_1_SPLIT_DIR,
    phase1_1_manifest_path: Path = PHASE1_1_MANIFEST_PATH,
    rules_path: Path = DEFAULT_RULES_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    audit_path: Path = DEFAULT_AUDIT_PATH,
) -> tuple[dict[str, Any], bool]:
    """Materialize label-rules v3 beside the immutable v1 and v2 datasets."""

    phase1_split_dir = Path(phase1_split_dir)
    phase1_manifest_path = Path(phase1_manifest_path)
    phase1_1_split_dir = Path(phase1_1_split_dir)
    phase1_1_manifest_path = Path(phase1_1_manifest_path)
    rules_path = Path(rules_path)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    audit_path = Path(audit_path)
    rules = load_rules(rules_path)
    if rules.version != 3:
        raise ValueError(f"Phase 1.2 requires label rules version 3, got {rules.version}")
    phase1_1_manifest, source_paths, phase1_membership = _verify_source_artifacts(
        phase1_split_dir=phase1_split_dir,
        phase1_manifest_path=phase1_manifest_path,
        phase1_1_split_dir=phase1_1_split_dir,
        phase1_1_manifest_path=phase1_1_manifest_path,
    )
    fingerprint = _canonical_hash(
        {
            "phase1_manifest_sha256": sha256_file(phase1_manifest_path),
            "phase1_1_manifest_sha256": sha256_file(phase1_1_manifest_path),
            "phase1_1_split_sha256": {
                name: phase1_1_manifest["splits"][name]["payload_sha256"] for name in SPLITS
            },
            "rules_sha256": sha256_file(rules_path),
            "labeler_sha256": sha256_file(Path(__file__).with_name("labels.py")),
            "runner_sha256": sha256_file(Path(__file__)),
            "task_schema": TASK_SCHEMA,
        }
    )
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if _verify_existing(existing, fingerprint, output_dir, audit_path):
            return existing, True
        raise RuntimeError("versioned v3 label artifact changed; refusing to overwrite")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = _split_paths(output_dir)
    if any(path.exists() for path in output_paths.values()):
        raise RuntimeError("v3 split payload exists without its manifest; refusing to overwrite")
    for name in SPLITS:
        _materialize_split(source_paths[name], output_paths[name], rules_path)
    stats = {
        name: _split_stats(source_paths[name], output_paths[name], phase1_membership[name])
        for name in SPLITS
    }
    audit = _write_changed_row_audit(source_paths, output_paths, audit_path)
    dataset_hash = _canonical_hash(
        {
            "label_rules_version": rules.version,
            "splits": {
                name: {
                    "membership_sha256": stats[name]["membership_sha256"],
                    "payload_sha256": stats[name]["payload_sha256"],
                }
                for name in SPLITS
            },
        }
    )
    if dataset_hash == phase1_1_manifest["dataset_hash"]:
        raise AssertionError("label rules v3 must produce a new dataset hash")
    manifest = {
        "version": 3,
        "fingerprint": fingerprint,
        "dataset_hash": dataset_hash,
        "phase1_source": {
            "manifest_path": str(phase1_manifest_path.resolve()),
            "manifest_sha256": sha256_file(phase1_manifest_path),
        },
        "phase1_1_source": {
            "manifest_path": str(phase1_1_manifest_path.resolve()),
            "manifest_sha256": sha256_file(phase1_1_manifest_path),
            "dataset_hash": phase1_1_manifest["dataset_hash"],
        },
        "label_rules": {
            "path": str(rules_path.resolve()),
            "version": rules.version,
            "sha256": sha256_file(rules_path),
            "strong_action_scope": "narrative_only",
            "tool_priority_asserted": list(rules.tool_priority),
        },
        "protocol": {
            "membership_frozen": True,
            "membership_hash_algorithm": "sha256(sorted decimal complaint_id + newline)",
            "label_derivation_versioned_under_d3_1": True,
            "phase2_data_generation_started": False,
            "network_calls": 0,
        },
        "splits": stats,
        "audit": {
            "path": str(audit_path.resolve()),
            **audit,
            "sha256": sha256_file(audit_path),
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest, False


def _score_dict(item: ScoreBreakdown) -> dict[str, Any]:
    return {
        "scorer_version": item.scorer_version,
        "json_valid": item.json_valid,
        "schema_valid": item.schema_valid,
        "decision_checks": item.decision_checks,
        "secondary_metrics": item.secondary_metrics,
        "task_success": item.task_success,
        "reward": item.reward,
        "errors": item.errors,
    }


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total**2)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _metrics(scores: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scores)

    def rate(key: str) -> float:
        return sum(bool(item[key]) for item in scores) / total if total else 0.0

    def decision_rate(key: str) -> float:
        return sum(bool(item["decision_checks"][key]) for item in scores) / total if total else 0.0

    def secondary_rate(key: str) -> float:
        return (
            sum(bool(item["secondary_metrics"][key]) for item in scores) / total if total else 0.0
        )

    successes = sum(bool(item["task_success"]) for item in scores)
    low, high = _wilson(successes, total)
    return {
        "scorer_version": SCORER_VERSION,
        "samples": total,
        "task_success_count": successes,
        "task_success": successes / total if total else 0.0,
        "task_success_ci95_wilson": [low, high],
        "schema_valid": rate("schema_valid"),
        "urgency_match": decision_rate("urgency"),
        "ambiguity_flag_match": decision_rate("ambiguity_flag"),
        "tool_choice_match": decision_rate("tool_choice"),
        "tool_arguments_structural_valid": decision_rate("tool_arguments_structural"),
        "secondary_metrics": {
            "product_match": secondary_rate("product_match"),
            "issue_normalized_match": secondary_rate("issue_normalized_match"),
            "company_normalized_match": secondary_rate("company_normalized_match"),
            "tool_arguments_semantic_valid": secondary_rate("tool_arguments_semantic_valid"),
            "abstention_correct": secondary_rate("abstention_correct"),
        },
        "mean_reward": (sum(float(item["reward"]) for item in scores) / total if total else 0.0),
    }


def _load_receipts(
    ledger_path: Path, receipts_path: Path, phase1_1_dataset_hash: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not ledger_path.is_file() or not receipts_path.is_file():
        raise FileNotFoundError("Phase 1.1 calibration ledger or receipts are missing")
    ledger = json.loads(ledger_path.read_text())
    if ledger.get("status") != "complete" or ledger.get("within_budget") is not True:
        raise ValueError("Phase 1.1 calibration ledger is not complete and within budget")
    if ledger.get("dataset_hash") != phase1_1_dataset_hash:
        raise ValueError("Phase 1.1 calibration ledger targets a different v2 dataset")
    if ledger.get("receipts_sha256") != sha256_file(receipts_path):
        raise ValueError("Phase 1.1 calibration receipt hash mismatch")
    records = [json.loads(line) for line in receipts_path.read_text().splitlines() if line]
    if len(records) != ledger.get("calls_recorded") or len(records) != 100:
        raise ValueError("Phase 1.2 requires exactly the existing 100 calibration receipts")
    if any(record.get("status") != "ok" or "parsed_output" not in record for record in records):
        raise ValueError("every calibration receipt must contain an existing parsed output")
    ids = [int(record["complaint_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate complaint_id in Phase 1.1 receipts")
    return ledger, records


def _labels_for_ids(path: Path, complaint_ids: list[int]) -> dict[int, dict[str, Any]]:
    placeholders = ", ".join("?" for _ in complaint_ids)
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"""
            SELECT complaint_id, label_json
            FROM read_parquet(?)
            WHERE complaint_id IN ({placeholders})
            """,
            [str(path), *complaint_ids],
        ).fetchall()
    finally:
        con.close()
    labels = {int(complaint_id): json.loads(label_json) for complaint_id, label_json in rows}
    if set(labels) != set(complaint_ids):
        raise ValueError("calibration receipt IDs do not exactly match the CAL split")
    return labels


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_rescore_report(
    *,
    manifest: dict[str, Any],
    ledger: dict[str, Any],
    receipts_path: Path,
    old_metrics: dict[str, Any],
    new_metrics: dict[str, Any],
    gold_changes: int,
    gold_tool_transitions: Counter[str],
    success_transitions: Counter[str],
    target_band: tuple[float, float],
) -> str:
    target_low, target_high = target_band
    success = float(new_metrics["task_success"])
    low, high = new_metrics["task_success_ci95_wilson"]
    if target_low <= success <= target_high:
        gate_status = "PASS — inside band"
    elif low <= target_high and high >= target_low:
        gate_status = "PASS — near band (95% interval overlaps)"
    else:
        gate_status = "ESCALATE — outside band"
    delta = success - float(old_metrics["task_success"])
    lines = [
        "# Phase 1.2 calibration receipt re-score",
        "",
        "## Gate result",
        "",
        (
            f"**{gate_status}.** The same {new_metrics['samples']} recorded model outputs score "
            f"**{_percent(success)}** task success against label rules v3 (95% Wilson CI "
            f"{_percent(low)}–{_percent(high)}), versus "
            f"**{_percent(float(old_metrics['task_success']))}** against v2 "
            f"(delta {delta * 100:+.1f} percentage points)."
        ),
        "",
        (
            "No API request was made: this is an offline re-score of immutable Phase 1.1 "
            "receipts, not a new calibration run or a Phase-3 R0 result."
        ),
        "",
        "## Decision-check delta",
        "",
        "| Check | v2 receipts | v3 re-score | Delta |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("Task success", "task_success"),
        ("Schema valid", "schema_valid"),
        ("Urgency match", "urgency_match"),
        ("Ambiguity flag match", "ambiguity_flag_match"),
        ("Tool choice match", "tool_choice_match"),
        ("Tool arguments structurally valid", "tool_arguments_structural_valid"),
        ("Mean scorer-v2 reward", "mean_reward"),
    ):
        old = float(old_metrics[key])
        new = float(new_metrics[key])
        lines.append(
            f"| {label} | {_percent(old)} | {_percent(new)} | {(new - old) * 100:+.1f} pp |"
        )
    lines.extend(
        [
            "",
            "## Gold-label delta on the 100 receipt IDs",
            "",
            f"- Gold labels changed from v2 to v3: {gold_changes}/100.",
            "- Gold tool transitions: "
            + (
                ", ".join(f"{key}={value}" for key, value in sorted(gold_tool_transitions.items()))
                if gold_tool_transitions
                else "none"
            )
            + ".",
            "- Task-success transitions: "
            + ", ".join(f"{key}={value}" for key, value in sorted(success_transitions.items()))
            + ".",
            "",
            "## Frozen-membership proof",
            "",
            "| Split | Rows | Phase 1 SHA-256 | v2 SHA-256 | v3 SHA-256 | Match |",
            "|---|---:|---|---|---|---|",
        ]
    )
    for name in SPLITS:
        split = manifest["splits"][name]
        lines.append(
            f"| {name} | {split['rows']} | `{split['phase1_membership_sha256']}` | "
            f"`{split['phase1_1_membership_sha256']}` | `{split['membership_sha256']}` | yes |"
        )
    lines.extend(
        [
            "",
            f"- Version-2 dataset hash: `{manifest['phase1_1_source']['dataset_hash']}`",
            f"- Version-3 dataset hash: `{manifest['dataset_hash']}`",
            f"- Label-rules-v3 SHA-256: `{manifest['label_rules']['sha256']}`",
            f"- Changed-row audit: `{manifest['audit']['path']}`",
            (
                "- Changed labels by split: "
                + ", ".join(
                    f"{name}={manifest['splits'][name]['labels_changed_from_v2']}"
                    for name in SPLITS
                )
            ),
            "",
            "## Receipt integrity and reproducibility",
            "",
            f"- Existing receipt file: `{receipts_path.resolve()}`",
            f"- Existing receipt SHA-256: `{sha256_file(receipts_path)}`",
            f"- Existing provider calls recorded: {ledger['calls_recorded']}",
            "- New network calls: 0",
            f"- Scorer version: `{SCORER_VERSION}`",
            f"- Model identity preserved from receipts: `{ledger['model']}`",
            f"- Append-only run record: `results/runs.jsonl` entry `{CALIBRATION_RUN_ID}`",
            "- Reproduction command: `make phase1-2`",
            "",
            "## Known limitations intentionally left open",
            "",
            (
                "- Negation-blind keyword matching remains (the delegated review estimated "
                "about 18 affected rows containing negated `identity theft`)."
            ),
            (
                "- The single-action taxonomy remains: escalation outranks refund for "
                "dual-remedy narratives."
            ),
            "",
            "**HUMAN REVIEW REQUIRED before label freeze or Phase 2 starts.**",
            "",
        ]
    )
    return "\n".join(lines)


def run_calibration_rescore(
    *,
    manifest: dict[str, Any],
    phase1_1_cal_path: Path = PHASE1_1_SPLIT_DIR / "cal.parquet",
    phase1_2_cal_path: Path = DEFAULT_OUTPUT_DIR / "cal.parquet",
    ledger_path: Path = DEFAULT_CALIBRATION_LEDGER_PATH,
    receipts_path: Path = DEFAULT_RECEIPTS_PATH,
    calibration_config_path: Path = DEFAULT_CALIBRATION_CONFIG_PATH,
    report_path: Path = DEFAULT_RESCORE_REPORT_PATH,
) -> dict[str, Any]:
    """Re-score existing parsed outputs against v3 labels without network access."""

    phase1_1_cal_path = Path(phase1_1_cal_path)
    phase1_2_cal_path = Path(phase1_2_cal_path)
    ledger_path = Path(ledger_path)
    receipts_path = Path(receipts_path)
    calibration_config_path = Path(calibration_config_path)
    report_path = Path(report_path)
    ledger, records = _load_receipts(
        ledger_path, receipts_path, manifest["phase1_1_source"]["dataset_hash"]
    )
    complaint_ids = [int(record["complaint_id"]) for record in records]
    v2_labels = _labels_for_ids(phase1_1_cal_path, complaint_ids)
    v3_labels = _labels_for_ids(phase1_2_cal_path, complaint_ids)
    old_scores = [record["score"] for record in records]
    new_scores = [
        _score_dict(
            score({"label": v3_labels[int(record["complaint_id"])]}, record["parsed_output"])
        )
        for record in records
    ]
    old_metrics = _metrics(old_scores)
    new_metrics = _metrics(new_scores)
    if old_metrics["task_success_count"] != ledger["metrics"]["task_success_count"]:
        raise ValueError("receipt scores do not reproduce the Phase 1.1 ledger")
    changed_ids = [
        complaint_id
        for complaint_id in complaint_ids
        if v2_labels[complaint_id] != v3_labels[complaint_id]
    ]
    gold_tool_transitions = Counter(
        (
            f"{v2_labels[complaint_id]['tool_call']['name']} -> "
            f"{v3_labels[complaint_id]['tool_call']['name']}"
        )
        for complaint_id in changed_ids
        if v2_labels[complaint_id]["tool_call"]["name"]
        != v3_labels[complaint_id]["tool_call"]["name"]
    )
    success_transitions = Counter(
        f"{str(bool(old['task_success'])).lower()} -> {str(bool(new['task_success'])).lower()}"
        for old, new in zip(old_scores, new_scores, strict=True)
    )
    config = yaml.safe_load(calibration_config_path.read_text())
    target_band = tuple(float(value) for value in config["target_task_success_band"])
    report = _render_rescore_report(
        manifest=manifest,
        ledger=ledger,
        receipts_path=receipts_path,
        old_metrics=old_metrics,
        new_metrics=new_metrics,
        gold_changes=len(changed_ids),
        gold_tool_transitions=gold_tool_transitions,
        success_transitions=success_transitions,
        target_band=(target_band[0], target_band[1]),
    )
    _write_text_atomic(report_path, report)
    return {
        "run_id": CALIBRATION_RUN_ID,
        "network_calls": 0,
        "receipts_sha256": sha256_file(receipts_path),
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "gold_labels_changed": len(changed_ids),
        "gold_tool_transitions": dict(sorted(gold_tool_transitions.items())),
        "task_success_transitions": dict(sorted(success_transitions.items())),
        "target_band": list(target_band),
        "report_path": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
    }


def run_phase1_2(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    audit_path: Path = DEFAULT_AUDIT_PATH,
    report_path: Path = DEFAULT_RESCORE_REPORT_PATH,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    manifest, noop = run_phase1_2_labels(
        output_dir=output_dir,
        manifest_path=manifest_path,
        audit_path=audit_path,
    )
    rescore = run_calibration_rescore(
        manifest=manifest,
        phase1_2_cal_path=Path(output_dir) / "cal.parquet",
        report_path=report_path,
    )
    return manifest, rescore, noop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forge.data.phase1_2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_RESCORE_REPORT_PATH)
    args = parser.parse_args(argv)
    manifest, rescore, noop = run_phase1_2(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        audit_path=args.audit,
        report_path=args.report,
    )
    status = "frozen no-op" if noop else "materialized"
    changed = sum(split["labels_changed_from_v2"] for split in manifest["splits"].values())
    metrics = rescore["new_metrics"]
    print(
        f"label rules v3: {status}; dataset_hash={manifest['dataset_hash']}; "
        f"changed_labels={changed}; task_success={metrics['task_success']:.3f}; "
        "network_calls=0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
