"""One-time materialization of the four frozen Phase-1 task splits."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import duckdb

from forge.data.ingest import (
    DEFAULT_MANIFEST_PATH as DEFAULT_INGEST_MANIFEST_PATH,
)
from forge.data.ingest import (
    DEFAULT_OUTPUT_PATH as DEFAULT_INGEST_PATH,
)
from forge.data.ingest import (
    SMOKE_MANIFEST_PATH as SMOKE_INGEST_MANIFEST_PATH,
)
from forge.data.ingest import (
    SMOKE_OUTPUT_PATH as SMOKE_INGEST_PATH,
)
from forge.data.ingest import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "splits"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "manifest.json"
DEFAULT_AUDIT_PATH = REPO_ROOT / "results" / "phase1_label_audit.md"
SMOKE_OUTPUT_DIR = REPO_ROOT / "data" / "smoke" / "splits"
SMOKE_MANIFEST_PATH = SMOKE_OUTPUT_DIR / "manifest.json"
SMOKE_AUDIT_PATH = REPO_ROOT / "data" / "smoke" / "phase1_label_audit.md"
SPLITS = ("train", "cal", "test_iid", "test_drift")
DISPLAY_NAMES = {
    "train": "TRAIN",
    "cal": "CAL",
    "test_iid": "TEST-IID",
    "test_drift": "TEST-DRIFT",
}
AUDIT_SEED = 20260815
AUDIT_ROWS_PER_SPLIT = 50
ROW_GROUP_SIZE = 122_880


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _quoted(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _new_temp_path(parent: Path, suffix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".phase1-", suffix=suffix, dir=parent)
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = _new_temp_path(path.parent, path.suffix or ".tmp")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, value: object) -> None:
    _write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _verify_ingest(ingest_path: Path, ingest_manifest_path: Path) -> dict[str, Any]:
    if not ingest_path.is_file() or not ingest_manifest_path.is_file():
        raise FileNotFoundError("ingest artifact and manifest must exist before splits")
    manifest = json.loads(ingest_manifest_path.read_text())
    expected = manifest.get("artifact", {}).get("sha256")
    actual = sha256_file(ingest_path)
    if actual != expected:
        raise ValueError(f"ingest artifact hash mismatch: expected {expected}, got {actual}")
    return manifest


def _split_paths(output_dir: Path) -> dict[str, Path]:
    return {name: output_dir / f"{name}.parquet" for name in SPLITS}


def _verify_existing(
    manifest: dict[str, Any],
    *,
    fingerprint: str,
    output_dir: Path,
    audit_path: Path,
) -> bool:
    if manifest.get("fingerprint") != fingerprint:
        return False
    for name, path in _split_paths(output_dir).items():
        expected = manifest.get("splits", {}).get(name, {}).get("sha256")
        if not path.is_file() or sha256_file(path) != expected:
            return False
    audit = manifest.get("audit", {})
    return audit_path.is_file() and sha256_file(audit_path) == audit.get("sha256")


def _materialize_split(ingest_path: Path, split: str, output_path: Path) -> None:
    temporary = _new_temp_path(output_path.parent, ".parquet")
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")
        con.execute(
            f"""
            COPY (
                SELECT complaint_id, date_received, narrative, source_product,
                       source_issue, source_company, source_class, source_split,
                       label_json
                FROM read_parquet({_quoted(ingest_path)})
                WHERE output_split = {_quoted(split)}
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


def _split_stats(path: Path) -> dict[str, Any]:
    con = duckdb.connect()
    try:
        row = con.execute(
            """
            SELECT count(*), count(DISTINCT complaint_id), min(date_received), max(date_received)
            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()
        source_rows = con.execute(
            "SELECT source_split, count(*) FROM read_parquet(?) GROUP BY 1 ORDER BY 1",
            [str(path)],
        ).fetchall()
    finally:
        con.close()
    rows, unique_ids, minimum, maximum = row
    if rows != unique_ids:
        raise ValueError(f"duplicate complaint_id in split {path}: {rows} rows, {unique_ids} ids")
    return {
        "path": str(path.resolve()),
        "display_name": DISPLAY_NAMES[path.stem],
        "rows": int(rows),
        "unique_complaint_ids": int(unique_ids),
        "date_min": minimum.isoformat(),
        "date_max": maximum.isoformat(),
        "source_rows": {str(name): int(count) for name, count in source_rows},
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _check_cross_split_disjoint(paths: dict[str, Path]) -> None:
    con = duckdb.connect()
    try:
        selects = [
            f"SELECT complaint_id, {_quoted(name)} AS split FROM read_parquet({_quoted(path)})"
            for name, path in paths.items()
        ]
        overlap = con.execute(
            f"""
            SELECT complaint_id, count(DISTINCT split) AS split_count
            FROM ({" UNION ALL ".join(selects)})
            GROUP BY complaint_id HAVING split_count > 1 LIMIT 1
            """
        ).fetchone()
    finally:
        con.close()
    if overlap is not None:
        raise ValueError(f"complaint_id {overlap[0]} appears in {overlap[1]} output splits")


def _rank(complaint_id: int) -> bytes:
    return hashlib.blake2b(f"{AUDIT_SEED}:{complaint_id}".encode(), digest_size=16).digest()


def _audit_ids(path: Path, limit: int) -> list[int]:
    con = duckdb.connect()
    try:
        rows = con.execute("SELECT complaint_id FROM read_parquet(?)", [str(path)]).fetchall()
        ids = (int(row[0]) for row in rows)
        selected = heapq.nsmallest(
            limit,
            ids,
            key=lambda complaint_id: (_rank(complaint_id), complaint_id),
        )
    finally:
        con.close()
    return sorted(selected)


def _escape_cell(value: object, limit: int | None = None) -> str:
    text = "" if value is None else " ".join(str(value).split())
    if limit is not None and len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text.replace("|", "\\|")


def _audit_rows(path: Path, ids: list[int]) -> list[tuple[Any, ...]]:
    placeholders = ",".join("?" for _ in ids)
    con = duckdb.connect()
    try:
        return con.execute(
            f"""
            SELECT complaint_id, date_received, source_product, source_issue,
                   source_company, narrative, label_json
            FROM read_parquet(?)
            WHERE complaint_id IN ({placeholders})
            ORDER BY complaint_id
            """,
            [str(path), *ids],
        ).fetchall()
    finally:
        con.close()


def write_label_audit(paths: dict[str, Path], audit_path: Path) -> int:
    """Emit a deterministic 200-row human-review artifact."""

    lines = [
        "# Phase 1 label audit",
        "",
        (
            "Status: **HUMAN REVIEW REQUIRED**. These are deterministic rule labels, "
            "not human-approved ground truth."
        ),
        "",
        f"Selection: {AUDIT_ROWS_PER_SPLIT} rows per split by smallest ",
        (
            f"`blake2b('{AUDIT_SEED}:<complaint_id>')`; total target = 200. "
            "Narratives are truncated for review."
        ),
        "",
        (
            "| Split | ID | Date | Source product | Product | Urgency | Ambiguous | "
            "Tool | Issue | Company | Narrative excerpt |"
        ),
        "|---|---:|---|---|---|---|---|---|---|---|---|",
    ]
    total = 0
    for split in SPLITS:
        path = paths[split]
        limit = min(AUDIT_ROWS_PER_SPLIT, _split_stats(path)["rows"])
        for row in _audit_rows(path, _audit_ids(path, limit)):
            complaint_id, received, product, issue, company, narrative, label_json = row
            label = json.loads(label_json)
            lines.append(
                "| "
                + " | ".join(
                    (
                        DISPLAY_NAMES[split],
                        str(complaint_id),
                        received.isoformat(),
                        _escape_cell(product),
                        _escape_cell(label["product"]),
                        _escape_cell(label["urgency"]),
                        str(label["ambiguity_flag"]).lower(),
                        _escape_cell(label["tool_call"]["name"]),
                        _escape_cell(issue, 90),
                        _escape_cell(company, 70),
                        _escape_cell(narrative, 180),
                    )
                )
                + " |"
            )
            total += 1
    lines.extend(
        [
            "",
            "## Reviewer checklist",
            "",
            "- [ ] Product mapping matches the complaint narrative.",
            "- [ ] Issue normalization preserves the source meaning.",
            "- [ ] Urgency and ambiguity rules behave sensibly.",
            "- [ ] Tool choice and arguments are operationally appropriate.",
            (
                "- [ ] Any proposed correction is recorded separately; never edit "
                "frozen split files in place."
            ),
            "",
        ]
    )
    _write_text_atomic(audit_path, "\n".join(lines))
    return total


def run_splits(
    *,
    ingest_path: Path = DEFAULT_INGEST_PATH,
    ingest_manifest_path: Path = DEFAULT_INGEST_MANIFEST_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    audit_path: Path = DEFAULT_AUDIT_PATH,
    smoke: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Materialize the four splits once; return ``(manifest, was_noop)``."""

    ingest_path = Path(ingest_path)
    ingest_manifest_path = Path(ingest_manifest_path)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    audit_path = Path(audit_path)
    ingest_manifest = _verify_ingest(ingest_path, ingest_manifest_path)
    fingerprint = _canonical_hash(
        {
            "ingest_sha256": ingest_manifest["artifact"]["sha256"],
            "splitter_sha256": sha256_file(Path(__file__)),
            "splits": SPLITS,
            "audit_seed": AUDIT_SEED,
            "audit_rows_per_split": AUDIT_ROWS_PER_SPLIT,
        }
    )
    paths = _split_paths(output_dir)

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if _verify_existing(
            existing, fingerprint=fingerprint, output_dir=output_dir, audit_path=audit_path
        ):
            return existing, True
        if not smoke:
            raise RuntimeError(
                "frozen split artifact, audit, or input changed; refusing to overwrite"
            )
    elif any(path.exists() for path in paths.values()):
        if not smoke:
            raise RuntimeError("split files exist without a frozen manifest; refusing to overwrite")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        _materialize_split(ingest_path, name, path)
    _check_cross_split_disjoint(paths)
    stats = {name: _split_stats(path) for name, path in paths.items()}
    audit_rows = write_label_audit(paths, audit_path)
    expected_audit_rows = sum(min(AUDIT_ROWS_PER_SPLIT, stats[name]["rows"]) for name in SPLITS)
    if audit_rows != expected_audit_rows:
        raise AssertionError(f"audit has {audit_rows} rows, expected {expected_audit_rows}")
    manifest = {
        "version": 1,
        "frozen": not smoke,
        "fingerprint": fingerprint,
        "input": {
            "path": str(ingest_path.resolve()),
            "sha256": ingest_manifest["artifact"]["sha256"],
            "manifest_sha256": sha256_file(ingest_manifest_path),
        },
        "protocol": {
            "membership": "exact nlp-eval-lab temporal split membership",
            "test_drift": "disjoint union of 2023, 2024, 2025, and 2026-H1 drift slices",
            "cross_split_complaint_id_overlap": 0,
        },
        "splits": stats,
        "audit": {
            "path": str(audit_path.resolve()),
            "rows": audit_rows,
            "seed": AUDIT_SEED,
            "sha256": sha256_file(audit_path),
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forge.data.splits")
    parser.add_argument("--ingest", type=Path)
    parser.add_argument("--ingest-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    smoke = args.smoke or os.environ.get("SMOKE") == "1"
    manifest, noop = run_splits(
        ingest_path=args.ingest or (SMOKE_INGEST_PATH if smoke else DEFAULT_INGEST_PATH),
        ingest_manifest_path=args.ingest_manifest
        or (SMOKE_INGEST_MANIFEST_PATH if smoke else DEFAULT_INGEST_MANIFEST_PATH),
        output_dir=args.output_dir or (SMOKE_OUTPUT_DIR if smoke else DEFAULT_OUTPUT_DIR),
        manifest_path=args.manifest or (SMOKE_MANIFEST_PATH if smoke else DEFAULT_MANIFEST_PATH),
        audit_path=args.audit or (SMOKE_AUDIT_PATH if smoke else DEFAULT_AUDIT_PATH),
        smoke=smoke,
    )
    status = "frozen no-op" if noop else "materialized and frozen"
    rendered = ", ".join(f"{name}={item['rows']}" for name, item in manifest["splits"].items())
    print(f"splits: {status}; {rendered}; audit_rows={manifest['audit']['rows']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
