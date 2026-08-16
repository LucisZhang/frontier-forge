"""Atomic receipts and append-only Phase 3 result helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from forge.train.config import canonical_json


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, value: object) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    write_text_atomic(path, "".join(canonical_json(record) + "\n" for record in records))


def append_jsonl_once(path: Path, record: dict[str, Any], *, key: str) -> bool:
    """Append one immutable record, or no-op if an identical keyed record exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.is_file():
        existing = [json.loads(line) for line in path.read_text().splitlines() if line]
    matches = [item for item in existing if item.get(key) == record.get(key)]
    if matches:
        if len(matches) != 1 or canonical_json(matches[0]) != canonical_json(record):
            raise RuntimeError(f"append-only conflict for {key}={record.get(key)!r} in {path}")
        return False
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True


def sha256_tree(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"cannot hash empty artifact directory: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()  # type: ignore[no-any-return, union-attr]
    return str(value)
