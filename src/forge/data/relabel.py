"""Version Phase-1 labels without changing frozen split membership."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import duckdb

from forge.data.ingest import ROW_GROUP_SIZE, sha256_file
from forge.data.labels import DEFAULT_RULES_PATH, canonical_label_json, load_rules
from forge.data.splits import AUDIT_ROWS_PER_SPLIT, AUDIT_SEED, DISPLAY_NAMES, SPLITS
from forge.verify.schema import TASK_SCHEMA

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE1_SPLIT_DIR = REPO_ROOT / "data" / "splits"
PHASE1_MANIFEST_PATH = PHASE1_SPLIT_DIR / "manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "phase1_1" / "splits"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "phase1_1" / "manifest.json"
DEFAULT_AUDIT_PATH = REPO_ROOT / "results" / "phase1_1_label_audit.md"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _quoted(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _new_temp_path(parent: Path, suffix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".phase1-1-", suffix=suffix, dir=parent)
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


def membership_sha256(path: Path) -> str:
    """Hash sorted complaint IDs independently of labels or Parquet encoding."""

    digest = hashlib.sha256()
    con = duckdb.connect()
    try:
        cursor = con.execute(
            "SELECT complaint_id FROM read_parquet(?) ORDER BY complaint_id",
            [str(path)],
        )
        while rows := cursor.fetchmany(10_000):
            for (complaint_id,) in rows:
                digest.update(f"{int(complaint_id)}\n".encode())
    finally:
        con.close()
    return digest.hexdigest()


def _phase1_paths(split_dir: Path) -> dict[str, Path]:
    return {name: split_dir / f"{name}.parquet" for name in SPLITS}


def _verify_phase1(split_dir: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Phase 1 split manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1 or manifest.get("frozen") is not True:
        raise ValueError("Phase 1 source manifest must be frozen version 1")
    paths = _phase1_paths(split_dir)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Phase 1 {name} split is missing: {path}")
        expected = manifest.get("splits", {}).get(name, {}).get("sha256")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Phase 1 {name} payload hash changed: expected {expected}, got {actual}"
            )
    return manifest, paths


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
            "phase1_1_label_json",
            label_json,
            return_type="VARCHAR",
            null_handling="special",
        )
        con.execute(
            f"""
            COPY (
                SELECT complaint_id, date_received, narrative, source_product,
                       source_issue, source_company, source_class, source_split,
                       phase1_1_label_json(
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


def _split_stats(source_path: Path, output_path: Path) -> dict[str, Any]:
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
        raise ValueError(f"duplicate complaint_id in relabeled split: {output_path}")
    if source_membership != output_membership:
        raise ValueError(f"split membership changed during label derivation: {output_path}")
    return {
        "path": str(output_path.resolve()),
        "rows": int(rows),
        "unique_complaint_ids": int(unique_ids),
        "payload_sha256": sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
        "membership_sha256": output_membership,
        "phase1_membership_sha256": source_membership,
        "membership_matches_phase1": True,
        "labels_changed_from_v1": changed,
    }


def _rank(complaint_id: int) -> bytes:
    return hashlib.blake2b(f"{AUDIT_SEED}:{complaint_id}".encode(), digest_size=16).digest()


def _changed_rows(source_path: Path, output_path: Path) -> list[tuple[Any, ...]]:
    con = duckdb.connect()
    try:
        return con.execute(
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


def _escape_cell(value: object, limit: int | None = None) -> str:
    text = "" if value is None else " ".join(str(value).split())
    if limit is not None and len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text.replace("|", "\\|")


def _write_changed_row_audit(
    source_paths: dict[str, Path], output_paths: dict[str, Path], audit_path: Path
) -> tuple[int, dict[str, int]]:
    all_changed = {name: _changed_rows(source_paths[name], output_paths[name]) for name in SPLITS}
    totals = {name: len(rows) for name, rows in all_changed.items()}
    selected = {
        name: sorted(rows, key=lambda row: (_rank(int(row[0])), int(row[0])))[:AUDIT_ROWS_PER_SPLIT]
        for name, rows in all_changed.items()
    }
    selected_count = sum(len(rows) for rows in selected.values())
    lines = [
        "# Phase 1.1 label-rules-v2 changed-row audit",
        "",
        (
            "Status: **HUMAN REVIEW REQUIRED**. Split membership is unchanged; this "
            "artifact shows only labels changed by the documented ambiguity-rule fix."
        ),
        "",
        (
            "Selection: up to 50 deterministically ranked changed rows per frozen split "
            f"(`blake2b('{AUDIT_SEED}:<complaint_id>')`), capped at 200 rows total."
        ),
        "",
        "Full changed-label counts: "
        + ", ".join(f"{DISPLAY_NAMES[name]}={totals[name]}" for name in SPLITS)
        + ".",
        "",
        (
            "| Split | ID | Date | Chars | Source product | Source issue | Company | "
            "Old ambiguous | New ambiguous | Old tool | New tool | Narrative excerpt |"
        ),
        "|---|---:|---|---:|---|---|---|---|---|---|---|---|",
    ]
    for name in SPLITS:
        for row in selected[name]:
            complaint_id, received, product, issue, company, narrative, old_json, new_json = row
            old_label = json.loads(old_json)
            new_label = json.loads(new_json)
            lines.append(
                "| "
                + " | ".join(
                    (
                        DISPLAY_NAMES[name],
                        str(complaint_id),
                        received.isoformat(),
                        str(len(narrative or "")),
                        _escape_cell(product, 55),
                        _escape_cell(issue, 75),
                        _escape_cell(company, 55),
                        str(old_label["ambiguity_flag"]).lower(),
                        str(new_label["ambiguity_flag"]).lower(),
                        old_label["tool_call"]["name"],
                        new_label["tool_call"]["name"],
                        _escape_cell(narrative, 180),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Reviewer checklist",
            "",
            (
                "- [ ] Each removed ambiguity flag was caused only by a weak phrase "
                "in a long narrative."
            ),
            "- [ ] The v2 urgency remains supported by issue plus narrative evidence.",
            "- [ ] The post-fix tool choice follows the documented priority order.",
            (
                "- [ ] Any correction is recorded as a new label-rules version; "
                "frozen membership is untouched."
            ),
            "",
        ]
    )
    _write_text_atomic(audit_path, "\n".join(lines))
    return selected_count, totals


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


def run_relabel(
    *,
    phase1_split_dir: Path = PHASE1_SPLIT_DIR,
    phase1_manifest_path: Path = PHASE1_MANIFEST_PATH,
    rules_path: Path = DEFAULT_RULES_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    audit_path: Path = DEFAULT_AUDIT_PATH,
) -> tuple[dict[str, Any], bool]:
    """Materialize label-rules v2 beside, never over, the Phase-1 split files."""

    phase1_split_dir = Path(phase1_split_dir)
    phase1_manifest_path = Path(phase1_manifest_path)
    rules_path = Path(rules_path)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    audit_path = Path(audit_path)
    rules = load_rules(rules_path)
    if rules.version != 2:
        raise ValueError(f"Phase 1.1 requires label rules version 2, got {rules.version}")

    phase1_manifest, source_paths = _verify_phase1(phase1_split_dir, phase1_manifest_path)
    fingerprint = _canonical_hash(
        {
            "phase1_manifest_sha256": sha256_file(phase1_manifest_path),
            "phase1_split_sha256": {
                name: phase1_manifest["splits"][name]["sha256"] for name in SPLITS
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
        raise RuntimeError("versioned v2 label artifact changed; refusing to overwrite")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {name: output_dir / f"{name}.parquet" for name in SPLITS}
    if any(path.exists() for path in output_paths.values()):
        raise RuntimeError("v2 split payload exists without its manifest; refusing to overwrite")
    for name in SPLITS:
        _materialize_split(source_paths[name], output_paths[name], rules_path)
    stats = {name: _split_stats(source_paths[name], output_paths[name]) for name in SPLITS}
    audit_rows, changed_totals = _write_changed_row_audit(source_paths, output_paths, audit_path)
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
    manifest = {
        "version": 2,
        "fingerprint": fingerprint,
        "dataset_hash": dataset_hash,
        "phase1_source": {
            "manifest_path": str(phase1_manifest_path.resolve()),
            "manifest_sha256": sha256_file(phase1_manifest_path),
        },
        "label_rules": {
            "path": str(rules_path.resolve()),
            "version": rules.version,
            "sha256": sha256_file(rules_path),
        },
        "protocol": {
            "membership_frozen": True,
            "membership_hash_algorithm": "sha256(sorted decimal complaint_id + newline)",
            "label_derivation_versioned_under_d3_1": True,
            "phase2_data_generation_started": False,
        },
        "splits": stats,
        "audit": {
            "path": str(audit_path.resolve()),
            "selection_seed": AUDIT_SEED,
            "selection_cap": AUDIT_ROWS_PER_SPLIT * len(SPLITS),
            "rows_emitted": audit_rows,
            "full_changed_rows_by_split": changed_totals,
            "sha256": sha256_file(audit_path),
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forge.data.relabel")
    parser.add_argument("--phase1-split-dir", type=Path, default=PHASE1_SPLIT_DIR)
    parser.add_argument("--phase1-manifest", type=Path, default=PHASE1_MANIFEST_PATH)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT_PATH)
    args = parser.parse_args(argv)
    manifest, noop = run_relabel(
        phase1_split_dir=args.phase1_split_dir,
        phase1_manifest_path=args.phase1_manifest,
        rules_path=args.rules,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        audit_path=args.audit,
    )
    status = "frozen no-op" if noop else "materialized"
    changed = sum(manifest["audit"]["full_changed_rows_by_split"].values())
    print(
        f"label rules v2: {status}; dataset_hash={manifest['dataset_hash']}; "
        f"changed_labels={changed}; audit_rows={manifest['audit']['rows_emitted']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
