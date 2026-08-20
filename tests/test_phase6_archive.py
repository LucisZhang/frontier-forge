from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from forge.train.artifacts import sha256_tree

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ARCHIVE = _load_script("publish_phase6_hf", ROOT / "scripts/remote/publish_phase6_hf.py")


def test_inventory_reproduces_the_project_tree_hash_and_file_identities(
    tmp_path: Path,
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_bytes(b"alpha")
    (tmp_path / "nested/b.bin").write_bytes(b"beta\x00gamma")

    tree_hash, identities = ARCHIVE._inventory_tree(tmp_path)

    assert tree_hash == sha256_tree(tmp_path)
    assert [item.relative_path for item in identities] == ["a.txt", "nested/b.bin"]
    assert identities[0].sha256 == hashlib.sha256(b"alpha").hexdigest()
    assert (
        identities[0].git_blob_sha1
        == hashlib.sha1(b"blob 5\x00alpha", usedforsecurity=False).hexdigest()
    )


def test_variant_specs_are_bound_to_the_three_export_manifests() -> None:
    specs = {item.name: item for item in ARCHIVE._variant_specs()}

    assert set(specs) == {"bf16", "gptq_int4", "bf16_mtp_preserved"}
    assert specs["bf16"].expected_tree_sha256 == (
        "7cf43a2905513f61797b78b7e3fd7ebdacd1cba4fc89abea9ce209401e6e6435"
    )
    assert specs["gptq_int4"].expected_tree_sha256 == (
        "c99b42cf0e062cc75f2df8588725d0c29383666f3db0c1ae837ce15bfe6d39d2"
    )
    assert specs["bf16_mtp_preserved"].expected_tree_sha256 == (
        "7878b55f6fe6a9ecb12b9504b1a88d7bc6fef7ba72d91289b6e8d694f6bc75ce"
    )


class _Api:
    def __init__(self, entries: list[Any]) -> None:
        self.entries = entries

    def list_repo_tree(self, **_kwargs: Any) -> list[Any]:
        return self.entries


def test_remote_verifier_accepts_lfs_sha256_and_git_blob_identity(tmp_path: Path) -> None:
    (tmp_path / "weight.bin").write_bytes(b"weights")
    (tmp_path / "config.json").write_bytes(b"{}\n")
    tree_hash, identities = ARCHIVE._inventory_tree(tmp_path)
    by_name = {item.relative_path: item for item in identities}
    spec = ARCHIVE.VariantSpec("tiny", tmp_path, "tiny", tree_hash, ROOT / "manifest.json")
    entries = [
        SimpleNamespace(
            path="tiny/weight.bin",
            type="file",
            size=by_name["weight.bin"].size,
            lfs={"sha256": by_name["weight.bin"].sha256},
            blob_id=None,
        ),
        SimpleNamespace(
            path="tiny/config.json",
            type="file",
            size=by_name["config.json"].size,
            lfs=None,
            blob_id=by_name["config.json"].git_blob_sha1,
        ),
    ]

    result = ARCHIVE._verify_remote_variant(
        api=_Api(entries),
        repo_id="owner/repo",
        revision="abc",
        spec=spec,
        identities=identities,
        token="not-logged",
    )

    assert result["tree_sha256"] == tree_hash
    assert result["remote_verification"] == {
        "lfs_sha256_files": 1,
        "git_blob_sha1_files": 1,
        "downloaded_sha256_files": 0,
    }


def test_remote_verifier_fails_on_missing_or_mutated_files(tmp_path: Path) -> None:
    (tmp_path / "model.bin").write_bytes(b"model")
    tree_hash, identities = ARCHIVE._inventory_tree(tmp_path)
    spec = ARCHIVE.VariantSpec("tiny", tmp_path, "tiny", tree_hash, ROOT / "manifest.json")

    with pytest.raises(RuntimeError, match="remote tree mismatch"):
        ARCHIVE._verify_remote_variant(
            api=_Api([]),
            repo_id="owner/repo",
            revision="abc",
            spec=spec,
            identities=identities,
            token="not-logged",
        )

    entry = SimpleNamespace(
        path="tiny/model.bin",
        type="file",
        size=5,
        lfs={"sha256": "0" * 64},
        blob_id=None,
    )
    with pytest.raises(RuntimeError, match="LFS SHA-256 mismatch"):
        ARCHIVE._verify_remote_variant(
            api=_Api([entry]),
            repo_id="owner/repo",
            revision="abc",
            spec=spec,
            identities=identities,
            token="not-logged",
        )
