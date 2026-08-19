#!/usr/bin/env python3
"""Publish and verify the Phase 6 R1b Hugging Face archive from the GPU pod.

The token is read only through Hugging Face's normal credential lookup (HF_TOKEN or
``hf auth login``) and is never accepted as a command-line argument. Before upload, every
local export tree must reproduce the immutable export-manifest hash. After upload, every
remote file is matched by path and exact content identity: LFS SHA-256 for LFS objects,
Git-blob SHA-1 for ordinary files, with a download-and-SHA-256 fallback if metadata is
unavailable. No inference or GPU work is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token, hf_hub_download

from forge.train.artifacts import write_json_atomic
from forge.train.config import REPO_ROOT, sha256_file

PHASE3_MANIFEST = REPO_ROOT / "results/phase3_export_manifest_r1b_trl_s0.json"
MTP_MANIFEST = REPO_ROOT / "results/phase4/r1b_mtp_reexport_manifest.json"
MODEL_CARD = REPO_ROOT / "MODEL_CARD.md"
SOURCE_MANIFEST = REPO_ROOT / "results/phase6/source_manifest.json"
DEFAULT_RECEIPT = REPO_ROOT / "results/phase6/hf_archive_receipt.json"
ARCHIVE_CONTRACT_PATH = "provenance/archive_contract.json"
REMOTE_RECEIPT_PATH = "provenance/archive_receipt.json"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    local_path: Path
    remote_path: str
    expected_tree_sha256: str
    manifest_path: Path


@dataclass(frozen=True)
class FileIdentity:
    relative_path: str
    size: int
    sha256: str
    git_blob_sha1: str


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _variant_specs() -> tuple[VariantSpec, ...]:
    phase3 = _json(PHASE3_MANIFEST)
    mtp = _json(MTP_MANIFEST)
    declarations = (
        ("bf16", phase3["full_precision_export"], "bf16", PHASE3_MANIFEST),
        ("gptq_int4", phase3["deployment_int4_export"], "gptq-int4", PHASE3_MANIFEST),
        (
            "bf16_mtp_preserved",
            mtp["full_precision_export"],
            "bf16-mtp-preserved",
            MTP_MANIFEST,
        ),
    )
    return tuple(
        VariantSpec(
            name=name,
            local_path=REPO_ROOT / declaration["path"],
            remote_path=remote_path,
            expected_tree_sha256=declaration["sha256"],
            manifest_path=manifest_path,
        )
        for name, declaration, remote_path, manifest_path in declarations
    )


def _inventory_tree(path: Path) -> tuple[str, tuple[FileIdentity, ...]]:
    """Compute the manifest tree hash and exact per-file identities in one read pass."""
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"cannot archive empty export tree: {path}")
    if any(item.is_symlink() for item in files):
        raise ValueError(f"archive tree contains a symlink: {path}")

    tree_digest = hashlib.sha256()
    identities: list[FileIdentity] = []
    for item in files:
        relative = item.relative_to(path).as_posix()
        relative_bytes = relative.encode()
        size = item.stat().st_size
        tree_digest.update(len(relative_bytes).to_bytes(8, "big"))
        tree_digest.update(relative_bytes)
        file_digest = hashlib.sha256()
        blob_digest = hashlib.sha1(usedforsecurity=False)
        blob_digest.update(f"blob {size}\0".encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                tree_digest.update(block)
                file_digest.update(block)
                blob_digest.update(block)
        identities.append(
            FileIdentity(
                relative_path=relative,
                size=size,
                sha256=file_digest.hexdigest(),
                git_blob_sha1=blob_digest.hexdigest(),
            )
        )
    return tree_digest.hexdigest(), tuple(identities)


def _remote_path(entry: Any) -> str:
    value = getattr(entry, "path", None) or getattr(entry, "rfilename", None)
    if not isinstance(value, str):
        raise TypeError(f"repo-tree entry has no path: {entry!r}")
    return value


def _remote_lfs_sha256(entry: Any) -> str | None:
    lfs = getattr(entry, "lfs", None)
    if lfs is None:
        return None
    value = lfs.get("sha256") if isinstance(lfs, Mapping) else getattr(lfs, "sha256", None)
    return value if isinstance(value, str) else None


def _repo_files(entries: Iterable[Any]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for entry in entries:
        entry_type = getattr(entry, "type", None)
        if entry_type == "directory" or entry.__class__.__name__ == "RepoFolder":
            continue
        path = _remote_path(entry)
        if path in files:
            raise RuntimeError(f"duplicate remote path: {path}")
        files[path] = entry
    return files


def _display_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def _verify_remote_variant(
    *,
    api: HfApi,
    repo_id: str,
    revision: str,
    spec: VariantSpec,
    identities: tuple[FileIdentity, ...],
    token: str,
) -> dict[str, Any]:
    entries = api.list_repo_tree(
        repo_id=repo_id,
        path_in_repo=spec.remote_path,
        recursive=True,
        expand=True,
        revision=revision,
        repo_type="model",
        token=token,
    )
    remote = _repo_files(entries)
    expected = {f"{spec.remote_path}/{item.relative_path}": item for item in identities}
    if set(remote) != set(expected):
        missing = sorted(set(expected) - set(remote))
        extra = sorted(set(remote) - set(expected))
        raise RuntimeError(
            f"remote tree mismatch for {spec.name}: missing={missing}, extra={extra}"
        )

    lfs_verified = 0
    git_blob_verified = 0
    downloaded_verified = 0
    with tempfile.TemporaryDirectory(prefix="forge-hf-verify-") as temporary:
        for remote_name, identity in expected.items():
            entry = remote[remote_name]
            remote_size = getattr(entry, "size", None)
            if remote_size is not None and int(remote_size) != identity.size:
                raise RuntimeError(f"remote size mismatch: {remote_name}")
            lfs_sha256 = _remote_lfs_sha256(entry)
            if lfs_sha256 is not None:
                if lfs_sha256 != identity.sha256:
                    raise RuntimeError(f"remote LFS SHA-256 mismatch: {remote_name}")
                lfs_verified += 1
                continue
            blob_id = getattr(entry, "blob_id", None)
            if isinstance(blob_id, str):
                if blob_id != identity.git_blob_sha1:
                    raise RuntimeError(f"remote Git blob mismatch: {remote_name}")
                git_blob_verified += 1
                continue
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_name,
                    revision=revision,
                    repo_type="model",
                    token=token,
                    local_dir=temporary,
                )
            )
            if sha256_file(downloaded) != identity.sha256:
                raise RuntimeError(f"downloaded remote SHA-256 mismatch: {remote_name}")
            downloaded_verified += 1
    return {
        "name": spec.name,
        "local_path": _display_path(spec.local_path),
        "remote_path": spec.remote_path,
        "manifest_path": _display_path(spec.manifest_path),
        "tree_sha256": spec.expected_tree_sha256,
        "file_count": len(identities),
        "bytes": sum(item.size for item in identities),
        "remote_verification": {
            "lfs_sha256_files": lfs_verified,
            "git_blob_sha1_files": git_blob_verified,
            "downloaded_sha256_files": downloaded_verified,
        },
    }


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_temp_json(directory: Path, name: str, value: object) -> Path:
    path = directory / name
    write_json_atomic(path, value)
    return path


def _upload_file(
    api: HfApi,
    *,
    source: Path,
    remote_path: str,
    repo_id: str,
    token: str,
    message: str,
) -> str:
    commit = api.upload_file(
        path_or_fileobj=source,
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message=message,
    )
    return str(commit.oid)


def _verify_remote_file(
    *, repo_id: str, remote_path: str, revision: str, expected_sha256: str, token: str
) -> None:
    with tempfile.TemporaryDirectory(prefix="forge-hf-file-") as temporary:
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                revision=revision,
                repo_type="model",
                token=token,
                local_dir=temporary,
                force_download=True,
            )
        )
        if sha256_file(downloaded) != expected_sha256:
            raise RuntimeError(f"remote file SHA-256 mismatch: {remote_path}")


def publish(args: argparse.Namespace) -> dict[str, Any]:
    token = get_token()
    if not token:
        raise RuntimeError("No Hugging Face token found; set HF_TOKEN or run `hf auth login`")
    api = HfApi(token=token)
    identity = api.whoami(token=token)
    owner = identity.get("name") or identity.get("fullname")
    if not isinstance(owner, str) or not owner:
        raise RuntimeError("Hugging Face account identity did not include an owner name")
    repo_id = args.repo_id or f"{owner}/frontier-forge-r1b"
    if repo_id.split("/", 1)[0].casefold() != owner.casefold():
        raise RuntimeError(f"refusing to publish outside authenticated account {owner!r}")

    specs = _variant_specs()
    local_inventories: dict[str, tuple[FileIdentity, ...]] = {}
    for spec in specs:
        actual_tree, identities = _inventory_tree(spec.local_path)
        if actual_tree != spec.expected_tree_sha256:
            raise RuntimeError(
                f"local {spec.name} tree hash mismatch: "
                f"expected {spec.expected_tree_sha256}, got {actual_tree}"
            )
        local_inventories[spec.name] = identities
        print(
            f"local verified: {spec.name} files={len(identities)} "
            f"bytes={sum(item.size for item in identities)} tree={actual_tree}"
        )
    if sha256_file(MODEL_CARD) != args.model_card_sha256:
        raise RuntimeError("MODEL_CARD.md changed after the reviewed archive command was prepared")

    existed = api.repo_exists(repo_id=repo_id, repo_type="model", token=token)
    if existed and not args.reuse_existing:
        raise RuntimeError(
            f"Hugging Face repository already exists: {repo_id}; inspect it and rerun with "
            "--reuse-existing only if it is the intended archive"
        )
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=args.reuse_existing,
        token=token,
    )

    with tempfile.TemporaryDirectory(prefix="forge-hf-contract-") as temporary_name:
        temporary = Path(temporary_name)
        contract = {
            "schema_version": "frontier-forge-hf-archive-contract-v1",
            "source_repository": "https://github.com/LucisZhang/frontier-forge",
            "source_git_sha": _git_sha(),
            "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
            "model_card_sha256": sha256_file(MODEL_CARD),
            "variants": {
                spec.remote_path: {
                    "tree_sha256": spec.expected_tree_sha256,
                    "manifest_path": spec.manifest_path.relative_to(REPO_ROOT).as_posix(),
                }
                for spec in specs
            },
        }
        contract_path = _write_temp_json(temporary, "archive_contract.json", contract)
        _upload_file(
            api,
            source=contract_path,
            remote_path=ARCHIVE_CONTRACT_PATH,
            repo_id=repo_id,
            token=token,
            message="chore: add archive contract",
        )
        _upload_file(
            api,
            source=MODEL_CARD,
            remote_path="README.md",
            repo_id=repo_id,
            token=token,
            message="docs: add model card",
        )
        for manifest in (PHASE3_MANIFEST, MTP_MANIFEST, SOURCE_MANIFEST):
            _upload_file(
                api,
                source=manifest,
                remote_path=f"provenance/{manifest.name}",
                repo_id=repo_id,
                token=token,
                message=f"chore: add {manifest.name}",
            )

        upload_commits: dict[str, str] = {}
        for spec in specs:
            print(f"uploading: {spec.name} -> {repo_id}/{spec.remote_path}")
            commit = api.upload_folder(
                folder_path=spec.local_path,
                path_in_repo=spec.remote_path,
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message=f"feat: archive {spec.name}",
            )
            upload_commits[spec.name] = str(commit.oid)

        artifact_commit = str(api.repo_info(repo_id, repo_type="model", token=token).sha)
        verification = [
            _verify_remote_variant(
                api=api,
                repo_id=repo_id,
                revision=artifact_commit,
                spec=spec,
                identities=local_inventories[spec.name],
                token=token,
            )
            for spec in specs
        ]
        _verify_remote_file(
            repo_id=repo_id,
            remote_path="README.md",
            revision=artifact_commit,
            expected_sha256=sha256_file(MODEL_CARD),
            token=token,
        )
        public_receipt = {
            "schema_version": "frontier-forge-hf-public-receipt-v1",
            "verified_artifact_commit": artifact_commit,
            "source_git_sha": _git_sha(),
            "model_card_sha256": sha256_file(MODEL_CARD),
            "variants": verification,
        }
        public_receipt_path = _write_temp_json(temporary, "archive_receipt.json", public_receipt)
        _upload_file(
            api,
            source=public_receipt_path,
            remote_path=REMOTE_RECEIPT_PATH,
            repo_id=repo_id,
            token=token,
            message="chore: add verified archive receipt",
        )
        receipt_commit = str(api.repo_info(repo_id, repo_type="model", token=token).sha)
        _verify_remote_file(
            repo_id=repo_id,
            remote_path=REMOTE_RECEIPT_PATH,
            revision=receipt_commit,
            expected_sha256=sha256_file(public_receipt_path),
            token=token,
        )

    receipt = {
        "schema_version": "frontier-forge-hf-archive-receipt-v1",
        "completed_at": datetime.now(UTC).isoformat(),
        "repo_id": repo_id,
        "repo_url": f"https://huggingface.co/{repo_id}",
        "private": bool(api.repo_info(repo_id, repo_type="model", token=token).private),
        "source_git_sha": _git_sha(),
        "verified_artifact_commit": artifact_commit,
        "receipt_commit": receipt_commit,
        "upload_commits": upload_commits,
        "model_card": {"remote_path": "README.md", "sha256": sha256_file(MODEL_CARD)},
        "variants": verification,
        "verification_method": (
            "local manifest tree SHA-256 before upload; exact path set plus remote LFS "
            "SHA-256 or Git blob identity per file after upload"
        ),
        "token_handling": "credential lookup only; token value neither logged nor persisted",
    }
    write_json_atomic(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", help="defaults to AUTHENTICATED_USER/frontier-forge-r1b")
    parser.add_argument("--private", action="store_true", help="create a private model repository")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="allow writing to an already-existing repository after manual inspection",
    )
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument(
        "--model-card-sha256",
        default=sha256_file(MODEL_CARD),
        help="review-time guard; normally left at the committed MODEL_CARD.md hash",
    )
    args = parser.parse_args()
    publish(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
