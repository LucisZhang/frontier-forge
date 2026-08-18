"""Audit and preserve model-native MTP tensors in an existing merged export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from forge.train.artifacts import sha256_tree, write_json_atomic
from forge.train.config import REPO_ROOT, relative_path, sha256_file


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors header length: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        raw_header = handle.read(header_length)
    try:
        header = json.loads(raw_header.rstrip(b" \t\r\n\0"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid safetensors JSON header: {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return header_length, header


def audit_index(
    *, index_path: Path, repository: str, revision: str, source_url: str
) -> dict[str, Any]:
    value = json.loads(index_path.read_text())
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint index has no weight_map object")
    keys = sorted(str(key) for key in weight_map)
    mtp_keys = [key for key in keys if key.startswith("mtp.")]
    return {
        "version": 1,
        "status": "complete",
        "audit_type": "no_gpu_checkpoint_index_metadata",
        "repository": repository,
        "revision": revision,
        "source_url": source_url,
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "weight_key_count": len(keys),
        "mtp_weight_key_count": len(mtp_keys),
        "mtp_weight_keys": mtp_keys,
        "mtp_shards": sorted({str(weight_map[key]) for key in mtp_keys}),
        "decision_branch": "native_mtp_reexport" if mtp_keys else "prompt_lookup_ngram",
    }


def _range_get(client: httpx.Client, url: str, start: int, end: int) -> bytes:
    """Fetch one byte range in retryable 4 MiB pieces."""

    chunks: list[bytes] = []
    cursor = start
    chunk_bytes = 4 * 1024 * 1024
    while cursor <= end:
        chunk_end = min(end, cursor + chunk_bytes - 1)
        expected = chunk_end - cursor + 1
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = client.get(url, headers={"Range": f"bytes={cursor}-{chunk_end}"})
                response.raise_for_status()
                if response.status_code != 206 or len(response.content) != expected:
                    raise RuntimeError(
                        f"range request was not honored for {url}: "
                        f"status={response.status_code}, expected={expected}, "
                        f"actual={len(response.content)}"
                    )
                chunks.append(response.content)
                cursor = chunk_end + 1
                break
            except (httpx.HTTPError, RuntimeError) as exc:
                last_error = exc
                if attempt == 4:
                    raise RuntimeError(
                        f"range download failed after retries for {url} bytes={cursor}-{chunk_end}"
                    ) from exc
                time.sleep(2**attempt)
        else:  # pragma: no cover - defensive; the loop either breaks or raises
            assert last_error is not None
            raise last_error
    return b"".join(chunks)


def _remote_safetensors_header(client: httpx.Client, url: str) -> tuple[int, dict[str, Any]]:
    raw_length = _range_get(client, url, 0, 7)
    header_length = struct.unpack("<Q", raw_length)[0]
    raw_header = _range_get(client, url, 8, 7 + header_length)
    header = json.loads(raw_header.rstrip(b" \t\r\n\0"))
    if not isinstance(header, dict):
        raise ValueError(f"remote safetensors header is not an object: {url}")
    return header_length, header


def _write_raw_safetensors(path: Path, tensors: dict[str, tuple[dict[str, Any], bytes]]) -> None:
    header: dict[str, Any] = {}
    offset = 0
    payloads: list[bytes] = []
    for key in sorted(tensors):
        metadata, payload = tensors[key]
        shape = metadata.get("shape")
        dtype = metadata.get("dtype")
        if not isinstance(shape, list) or not isinstance(dtype, str):
            raise ValueError(f"invalid tensor metadata for {key}")
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
        payloads.append(payload)
    raw_header = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode()
    padding = (-len(raw_header)) % 8
    raw_header += b" " * padding
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw_header)))
        handle.write(raw_header)
        for payload in payloads:
            handle.write(payload)


def fetch_mtp_bundle(
    *,
    index_path: Path,
    base_url: str,
    output_path: Path,
    repository: str,
    revision: str,
) -> dict[str, Any]:
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint index has no weight_map object")
    mtp_keys = sorted(str(key) for key in weight_map if str(key).startswith("mtp."))
    if not mtp_keys:
        raise RuntimeError("checkpoint index contains no mtp.* weights")
    by_shard: dict[str, list[str]] = {}
    for key in mtp_keys:
        by_shard.setdefault(str(weight_map[key]), []).append(key)

    tensors: dict[str, tuple[dict[str, Any], bytes]] = {}
    sources: list[dict[str, Any]] = []
    with httpx.Client(timeout=120, follow_redirects=True, trust_env=False) as client:
        for shard, shard_keys in sorted(by_shard.items()):
            url = f"{base_url.rstrip('/')}/{shard}"
            header_length, header = _remote_safetensors_header(client, url)
            data_start = 8 + header_length
            for key in shard_keys:
                metadata = header.get(key)
                if not isinstance(metadata, dict):
                    raise RuntimeError(f"index key {key} is absent from {shard}")
                offsets = metadata.get("data_offsets")
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(type(value) is not int for value in offsets)
                ):
                    raise ValueError(f"invalid data offsets for {key}")
                start, stop = offsets
                payload = _range_get(client, url, data_start + start, data_start + stop - 1)
                tensors[key] = (metadata, payload)
                sources.append(
                    {
                        "key": key,
                        "shard": shard,
                        "range_start": data_start + start,
                        "range_end": data_start + stop - 1,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
    _write_raw_safetensors(output_path, tensors)
    _, bundle_header = _read_safetensors_header(output_path)
    if sorted(bundle_header) != mtp_keys:
        raise RuntimeError("written MTP bundle does not match the checkpoint index")
    return {
        "version": 1,
        "status": "complete",
        "repository": repository,
        "revision": revision,
        "index_sha256": sha256_file(index_path),
        "bundle_path": str(output_path),
        "bundle_sha256": sha256_file(output_path),
        "bundle_bytes": output_path.stat().st_size,
        "mtp_weight_key_count": len(mtp_keys),
        "mtp_weight_keys": mtp_keys,
        "source_ranges": sources,
    }


def _copy_export_with_hardlinks(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite MTP-preserving export: {output}")
    output.mkdir(parents=True)
    for item in sorted(source.iterdir()):
        destination = output / item.name
        if item.is_dir():
            shutil.copytree(item, destination, copy_function=os.link)
        elif item.is_file():
            if item.name == "model.safetensors.index.json":
                shutil.copyfile(item, destination)
            else:
                os.link(item, destination)


def assemble_export(
    *,
    source: Path,
    output: Path,
    bundle: Path,
    base_audit: Path,
    adapter: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    audit = json.loads(base_audit.read_text())
    mtp_keys = audit.get("mtp_weight_keys")
    if not isinstance(mtp_keys, list) or not mtp_keys:
        raise RuntimeError("base index audit does not authorize an MTP re-export")
    _, bundle_header = _read_safetensors_header(bundle)
    if sorted(bundle_header) != sorted(mtp_keys):
        raise RuntimeError("MTP bundle keys do not match the archived base index audit")
    _, adapter_header = _read_safetensors_header(adapter)
    adapter_mtp_keys = sorted(key for key in adapter_header if key.startswith("mtp."))
    if adapter_mtp_keys:
        raise RuntimeError("R1b adapter modifies MTP weights; base-weight restoration is invalid")

    _copy_export_with_hardlinks(source, output)
    index_path = output / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("merged export has no sharded checkpoint index")
    collisions = sorted(set(weight_map).intersection(mtp_keys))
    if collisions:
        raise RuntimeError(f"source export unexpectedly already contains MTP weights: {collisions}")
    bundle_name = "model-mtp-preserved.safetensors"
    shutil.copyfile(bundle, output / bundle_name)
    for key in mtp_keys:
        weight_map[key] = bundle_name
    metadata = index.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise RuntimeError("checkpoint index metadata is not an object")
    tensor_bytes = sum(
        int(value["data_offsets"][1]) - int(value["data_offsets"][0])
        for key, value in bundle_header.items()
        if key != "__metadata__"
    )
    if isinstance(metadata.get("total_size"), int):
        metadata["total_size"] += tensor_bytes
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    artifact_hash = sha256_tree(output)
    manifest = {
        "version": 1,
        "status": "complete",
        "phase": 4,
        "operation": "mtp_preserving_r1b_reexport",
        "finished_at": datetime.now(UTC).isoformat(),
        "source_export": {
            "path": relative_path(source),
            "sha256": sha256_tree(source),
        },
        "source_adapter": {
            "path": relative_path(adapter.parent),
            "weights_path": relative_path(adapter),
            "weights_sha256": sha256_file(adapter),
            "mtp_weight_keys": adapter_mtp_keys,
        },
        "base_index_audit": {
            "path": relative_path(base_audit),
            "sha256": sha256_file(base_audit),
            "repository": audit["repository"],
            "revision": audit["revision"],
        },
        "preserved_mtp": {
            "bundle_sha256": sha256_file(bundle),
            "weight_key_count": len(mtp_keys),
            "weight_keys": sorted(mtp_keys),
            "source": "exact byte ranges from fixed-revision base checkpoint shards",
        },
        "full_precision_export": {
            "dtype": "bfloat16",
            "method": "peft_merge_plus_base_mtp_weight_preservation",
            "path": relative_path(output),
            "sha256": artifact_hash,
        },
        "notes": (
            "The existing merged export is immutable; this sibling export restores only base "
            "MTP tensors because the R1b LoRA adapter contains no MTP keys."
        ),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-index")
    audit.add_argument("--index", required=True)
    audit.add_argument("--repository", required=True)
    audit.add_argument("--revision", required=True)
    audit.add_argument("--source-url", required=True)
    audit.add_argument("--output", required=True)

    fetch = subparsers.add_parser("fetch-bundle")
    fetch.add_argument("--index", required=True)
    fetch.add_argument("--base-url", required=True)
    fetch.add_argument("--repository", required=True)
    fetch.add_argument("--revision", required=True)
    fetch.add_argument("--output", required=True)
    fetch.add_argument("--receipt", required=True)

    assemble = subparsers.add_parser("assemble-export")
    assemble.add_argument("--source", required=True)
    assemble.add_argument("--output", required=True)
    assemble.add_argument("--bundle", required=True)
    assemble.add_argument("--base-audit", required=True)
    assemble.add_argument("--adapter", required=True)
    assemble.add_argument("--manifest", required=True)
    args = parser.parse_args()

    if args.command == "audit-index":
        receipt = audit_index(
            index_path=Path(args.index),
            repository=args.repository,
            revision=args.revision,
            source_url=args.source_url,
        )
        write_json_atomic(Path(args.output), receipt)
    elif args.command == "fetch-bundle":
        receipt = fetch_mtp_bundle(
            index_path=Path(args.index),
            base_url=args.base_url,
            output_path=Path(args.output),
            repository=args.repository,
            revision=args.revision,
        )
        write_json_atomic(Path(args.receipt), receipt)
    else:
        receipt = assemble_export(
            source=REPO_ROOT / args.source,
            output=REPO_ROOT / args.output,
            bundle=Path(args.bundle),
            base_audit=REPO_ROOT / args.base_audit,
            adapter=REPO_ROOT / args.adapter,
            manifest_path=REPO_ROOT / args.manifest,
        )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
