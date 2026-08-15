"""Pinned CFPB snapshot verification and deterministic label materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import yaml

from forge.data.labels import DEFAULT_RULES_PATH, canonical_label_json, load_rules
from forge.verify.schema import TASK_SCHEMA

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "data_sources.yaml"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "ingest" / "labeled_rows.parquet"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "ingest" / "manifest.json"
SMOKE_OUTPUT_PATH = REPO_ROOT / "data" / "smoke" / "ingest" / "labeled_rows.parquet"
SMOKE_MANIFEST_PATH = REPO_ROOT / "data" / "smoke" / "ingest" / "manifest.json"
SMOKE_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "complaints_50.jsonl"
ROW_GROUP_SIZE = 122_880


@dataclass(frozen=True)
class SourceSpec:
    source_split: str
    output_split: str
    path: Path
    sha256: str
    expected_rows: int


@dataclass(frozen=True)
class DataSourceConfig:
    config_path: Path
    upstream_root: Path
    raw_snapshot_path: Path
    raw_manifest_path: Path
    raw_sha256: str
    raw_size_bytes: int
    derived_manifest_path: Path
    derived_input_path: Path
    derived_input_sha256: str
    sources: tuple[SourceSpec, ...]


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_root(raw: dict[str, Any], repo_root: Path) -> Path:
    upstream = raw["upstream"]
    env_name = str(upstream["root_env"])
    configured = os.environ.get(env_name, str(upstream["default_root"]))
    root = Path(configured).expanduser()
    return (repo_root / root).resolve() if not root.is_absolute() else root.resolve()


def load_data_source_config(
    path: Path = DEFAULT_CONFIG_PATH, *, repo_root: Path = REPO_ROOT
) -> DataSourceConfig:
    """Load source paths with an optional machine-local upstream-root override."""

    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    root = _resolve_root(raw, Path(repo_root).resolve())
    upstream = raw["upstream"]
    snapshot = upstream["raw_snapshot"]
    derived = upstream["derived_snapshot"]
    sources = tuple(
        SourceSpec(
            source_split=str(item["source_split"]),
            output_split=str(item["output_split"]),
            path=root / str(item["path"]),
            sha256=str(item["sha256"]),
            expected_rows=int(item["expected_rows"]),
        )
        for item in raw["sources"]
    )
    return DataSourceConfig(
        config_path=path,
        upstream_root=root,
        raw_snapshot_path=root / str(snapshot["path"]),
        raw_manifest_path=root / str(snapshot["manifest"]),
        raw_sha256=str(snapshot["sha256"]),
        raw_size_bytes=int(snapshot["size_bytes"]),
        derived_manifest_path=root / str(derived["manifest"]),
        derived_input_path=root / str(derived["input_path"]),
        derived_input_sha256=str(derived["input_sha256"]),
        sources=sources,
    )


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")


def verify_pinned_sources(config: DataSourceConfig) -> list[dict[str, Any]]:
    """Fail closed unless the configured raw and derived snapshots are exact."""

    _require_file(config.raw_snapshot_path, "frozen CFPB snapshot")
    _require_file(config.raw_manifest_path, "snapshot manifest")
    _require_file(config.derived_manifest_path, "derived split manifest")
    _require_file(config.derived_input_path, "derived deduplicated snapshot")

    raw_manifest = yaml.safe_load(config.raw_manifest_path.read_text())
    if raw_manifest.get("sha256") != config.raw_sha256:
        raise ValueError("raw snapshot manifest hash disagrees with configs/data_sources.yaml")
    if int(raw_manifest.get("size_bytes", -1)) != config.raw_size_bytes:
        raise ValueError("raw snapshot manifest size disagrees with configs/data_sources.yaml")
    if config.raw_snapshot_path.stat().st_size != config.raw_size_bytes:
        raise ValueError("frozen CFPB snapshot size mismatch; refusing to re-download or continue")
    if sha256_file(config.raw_snapshot_path) != config.raw_sha256:
        raise ValueError("frozen CFPB snapshot hash mismatch; refusing to re-download or continue")

    derived_manifest = yaml.safe_load(config.derived_manifest_path.read_text())
    if derived_manifest.get("input_sha256") != config.derived_input_sha256:
        raise ValueError("derived manifest input hash disagrees with configs/data_sources.yaml")
    if sha256_file(config.derived_input_path) != config.derived_input_sha256:
        raise ValueError("deduplicated snapshot hash mismatch")

    verified: list[dict[str, Any]] = []
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        for source in config.sources:
            _require_file(source.path, f"source split {source.source_split}")
            actual_hash = sha256_file(source.path)
            if actual_hash != source.sha256:
                raise ValueError(
                    f"source split {source.source_split} hash mismatch: "
                    f"expected {source.sha256}, got {actual_hash}"
                )
            rows = int(
                con.execute("SELECT count(*) FROM read_parquet(?)", [str(source.path)]).fetchone()[
                    0
                ]
            )
            if rows != source.expected_rows:
                raise ValueError(
                    f"source split {source.source_split} row mismatch: "
                    f"expected {source.expected_rows}, got {rows}"
                )
            verified.append(
                {
                    "source_split": source.source_split,
                    "output_split": source.output_split,
                    "path": str(source.path),
                    "sha256": actual_hash,
                    "rows": rows,
                }
            )
    finally:
        con.close()
    return verified


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint(config: DataSourceConfig, rules_path: Path) -> str:
    payload = {
        "config_sha256": sha256_file(config.config_path),
        "rules_sha256": sha256_file(rules_path),
        "labeler_sha256": sha256_file(Path(__file__).with_name("labels.py")),
        "schema_sha256": _canonical_hash(TASK_SCHEMA),
        "raw_sha256": config.raw_sha256,
        "derived_input_sha256": config.derived_input_sha256,
        "source_hashes": [source.sha256 for source in config.sources],
    }
    return _canonical_hash(payload)


def _quoted(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _new_temp_path(parent: Path, suffix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".phase1-", suffix=suffix, dir=parent)
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = _new_temp_path(path.parent, ".json")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _artifact_stats(path: Path) -> dict[str, Any]:
    con = duckdb.connect()
    try:
        row_count = int(
            con.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
        )
        split_rows = con.execute(
            "SELECT output_split, count(*) FROM read_parquet(?) GROUP BY 1 ORDER BY 1",
            [str(path)],
        ).fetchall()
        tool_rows = con.execute(
            """
            SELECT json_extract_string(label_json, '$.tool_call.name') AS tool, count(*)
            FROM read_parquet(?) GROUP BY 1 ORDER BY 1
            """,
            [str(path)],
        ).fetchall()
        ambiguity_rows = con.execute(
            """
            SELECT CAST(json_extract(label_json, '$.ambiguity_flag') AS BOOLEAN), count(*)
            FROM read_parquet(?) GROUP BY 1 ORDER BY 1
            """,
            [str(path)],
        ).fetchall()
    finally:
        con.close()
    return {
        "rows": row_count,
        "rows_by_output_split": {str(name): int(count) for name, count in split_rows},
        "rows_by_tool": {str(name): int(count) for name, count in tool_rows},
        "rows_by_ambiguity": {str(flag).lower(): int(count) for flag, count in ambiguity_rows},
    }


def _materialize_full(config: DataSourceConfig, rules_path: Path, output_path: Path) -> None:
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
                "product": product,
                "issue": issue,
                "company": company,
                "narrative": narrative,
                "complaint_id": complaint_id,
            },
            rules,
        )

    selects = []
    for source in config.sources:
        selects.append(
            f"""
            SELECT complaint_id, date_received, narrative,
                   product AS source_product, issue AS source_issue,
                   company AS source_company, "class" AS source_class,
                   {_quoted(source.source_split)} AS source_split,
                   {_quoted(source.output_split)} AS output_split,
                   phase1_label_json(product, issue, company, narrative, complaint_id) AS label_json
            FROM read_parquet({_quoted(source.path)})
            """
        )
    union_sql = " UNION ALL ".join(selects)

    temporary = _new_temp_path(output_path.parent, ".parquet")
    con = duckdb.connect()
    try:
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=true")
        con.create_function(
            "phase1_label_json",
            label_json,
            return_type="VARCHAR",
            null_handling="special",
        )
        con.execute(
            f"""
            COPY (
                SELECT * FROM ({union_sql})
                ORDER BY output_split, date_received, complaint_id, source_split
            ) TO {_quoted(temporary)}
            (FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE {ROW_GROUP_SIZE})
            """
        )
        os.replace(temporary, output_path)
    finally:
        con.close()
        if temporary.exists():
            temporary.unlink()


def _smoke_rows() -> list[tuple[Any, ...]]:
    raw_rows = [json.loads(line) for line in SMOKE_FIXTURE_PATH.read_text().splitlines() if line]
    products = (
        ("Credit card", "card"),
        ("Credit reporting", "credit_reporting"),
        ("Debt collection", "debt_collection"),
        ("Checking or savings account", "deposit_account"),
        ("Money transfers", "money_service"),
        ("Mortgage", "mortgage"),
        ("Payday loan", "payday_personal_loan"),
        ("Student loan", "student_loan"),
        ("Vehicle loan or lease", "vehicle_loan"),
    )
    split_dates = (
        ("train", "train", date(2021, 1, 1)),
        ("cal", "cal", date(2022, 1, 1)),
        ("test_iid", "test_iid", date(2022, 7, 1)),
        ("test_drift_2023", "test_drift", date(2023, 1, 1)),
    )
    rows: list[tuple[Any, ...]] = []
    for index, raw in enumerate(raw_rows, start=1):
        source_product, source_class = products[(index - 1) % len(products)]
        source_split, output_split, start = split_dates[(index - 1) % len(split_dates)]
        narrative = (
            f"{raw['narrative']} The consumer disputes an account entry and requests review."
        )
        source_issue = "Incorrect information on account"
        source_company = f"Synthetic Company {(index - 1) % 5 + 1}"
        label = canonical_label_json(
            {
                "complaint_id": index,
                "product": source_product,
                "issue": source_issue,
                "company": source_company,
                "narrative": narrative,
            }
        )
        rows.append(
            (
                index,
                start + timedelta(days=index),
                narrative,
                source_product,
                source_issue,
                source_company,
                source_class,
                source_split,
                output_split,
                label,
            )
        )
    return rows


def _materialize_smoke(output_path: Path) -> None:
    temporary = _new_temp_path(output_path.parent, ".parquet")
    con = duckdb.connect()
    try:
        con.execute(
            """
            CREATE TABLE smoke_rows (
                complaint_id BIGINT, date_received DATE, narrative VARCHAR,
                source_product VARCHAR, source_issue VARCHAR, source_company VARCHAR,
                source_class VARCHAR, source_split VARCHAR, output_split VARCHAR,
                label_json VARCHAR
            )
            """
        )
        con.executemany(
            "INSERT INTO smoke_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _smoke_rows(),
        )
        con.execute(
            f"""
            COPY (SELECT * FROM smoke_rows ORDER BY output_split, complaint_id)
            TO {_quoted(temporary)}
            (FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE {ROW_GROUP_SIZE})
            """
        )
        os.replace(temporary, output_path)
    finally:
        con.close()
        if temporary.exists():
            temporary.unlink()


def run_ingest(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    rules_path: Path = DEFAULT_RULES_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    smoke: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Materialize labels once; return ``(manifest, was_noop)``."""

    rules_path = Path(rules_path).resolve()
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    if not smoke and (manifest_path.exists() or output_path.exists()):
        if not manifest_path.is_file() or not output_path.is_file():
            raise RuntimeError(
                "incomplete frozen ingest state (artifact/manifest pair); refusing to overwrite"
            )
        existing = json.loads(manifest_path.read_text())
        expected_hash = existing.get("artifact", {}).get("sha256")
        actual_hash = sha256_file(output_path)
        if existing.get("frozen") is True and actual_hash == expected_hash:
            # D3.1 derives v2 labels from these local frozen rows. The v1
            # receipt remains sufficient for a no-op without re-downloading or
            # reinterpreting its now-versioned label derivation inputs.
            return existing, True
        raise RuntimeError("frozen ingest artifact hash changed; refusing to overwrite it")

    if smoke:
        fingerprint = _canonical_hash(
            {
                "fixture": sha256_file(SMOKE_FIXTURE_PATH),
                "rules": sha256_file(rules_path),
                "labeler": sha256_file(Path(__file__).with_name("labels.py")),
                "schema": TASK_SCHEMA,
            }
        )
        verified_sources = [{"source_split": "synthetic-smoke", "rows": 50}]
        source_snapshot = {"kind": "synthetic-smoke", "sha256": sha256_file(SMOKE_FIXTURE_PATH)}
    else:
        config = load_data_source_config(config_path)
        fingerprint = _fingerprint(config, rules_path)
        verified_sources = verify_pinned_sources(config)
        source_snapshot = {
            "path": str(config.raw_snapshot_path),
            "sha256": config.raw_sha256,
            "derived_input_sha256": config.derived_input_sha256,
        }

    if manifest_path.exists() and output_path.exists():
        existing = json.loads(manifest_path.read_text())
        hash_matches = sha256_file(output_path) == existing.get("artifact", {}).get("sha256")
        fingerprint_matches = existing.get("fingerprint") == fingerprint
        if hash_matches and fingerprint_matches:
            return existing, True
        if not smoke:
            raise RuntimeError(
                "frozen ingest artifact or its inputs changed; refusing to overwrite it"
            )
    elif manifest_path.exists() or output_path.exists():
        if not smoke:
            raise RuntimeError(
                "incomplete frozen ingest state (artifact/manifest pair); refusing to overwrite"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if smoke:
        _materialize_smoke(output_path)
    else:
        _materialize_full(config, rules_path, output_path)
    stats = _artifact_stats(output_path)
    manifest = {
        "version": 1,
        "frozen": not smoke,
        "fingerprint": fingerprint,
        "source_snapshot": source_snapshot,
        "sources": verified_sources,
        "label_rules": {
            "path": str(rules_path),
            "sha256": sha256_file(rules_path),
            "version": load_rules(rules_path).version,
        },
        "task_schema_sha256": _canonical_hash(TASK_SCHEMA),
        "artifact": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
            **stats,
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forge.data.ingest")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    smoke = args.smoke or os.environ.get("SMOKE") == "1"
    output = args.output or (SMOKE_OUTPUT_PATH if smoke else DEFAULT_OUTPUT_PATH)
    manifest_path = args.manifest or (SMOKE_MANIFEST_PATH if smoke else DEFAULT_MANIFEST_PATH)
    manifest, noop = run_ingest(
        config_path=args.config,
        rules_path=args.rules,
        output_path=output,
        manifest_path=manifest_path,
        smoke=smoke,
    )
    status = "frozen no-op" if noop else "materialized"
    artifact = manifest["artifact"]
    print(
        f"ingest: {status}; rows={artifact['rows']}; "
        f"sha256={artifact['sha256']}; path={artifact['path']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
