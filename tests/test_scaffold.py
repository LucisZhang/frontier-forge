import importlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C3_TARGETS = (
    "test",
    "lint",
    "gateway-test",
    "gateway-tsan",
    "ingest",
    "splits",
    "calibrate-difficulty",
    "teacher-data",
    "teacher-audit",
    "train-sft",
    "train-dpo",
    "train-grpo",
    "eval",
    "export-model",
    "serve-bench",
    "spec-decode-bench",
    "structured-bench",
    "bench-report",
    "gateway-bench",
    "sync-up",
    "sync-down",
    "demo-build",
    "reproduce-headline",
)
PACKAGES = (
    "forge",
    "forge.data",
    "forge.verify",
    "forge.teacher",
    "forge.train",
    "forge.serve",
    "forge.bench",
    "forge.analysis",
)


def test_package_stubs_are_importable() -> None:
    for package in PACKAGES:
        importlib.import_module(package)


def test_ci_fixture_has_exactly_50_synthetic_rows() -> None:
    configured_path = Path(os.environ.get("FORGE_FIXTURE", "tests/fixtures/complaints_50.jsonl"))
    fixture_path = configured_path if configured_path.is_absolute() else ROOT / configured_path
    rows = [json.loads(line) for line in fixture_path.read_text().splitlines() if line]

    assert len(rows) == 50
    assert all(set(row) == {"id", "narrative"} for row in rows)
    assert len({row["id"] for row in rows}) == 50


def test_every_c3_target_accepts_smoke() -> None:
    for target in C3_TARGETS:
        subprocess.run(
            ["make", "--no-print-directory", "--dry-run", target, "SMOKE=1"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )


def test_smoke_ingest_materializes_phase1_fixture() -> None:
    completed = subprocess.run(
        ["make", "--no-print-directory", "ingest", "SMOKE=1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "rows=50" in completed.stdout
    assert "synthetic" not in completed.stderr.lower()
