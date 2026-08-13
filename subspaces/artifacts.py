"""Minimal artifact plumbing: versioned manifests, atomic writes, hashes, refs.

Deliberately small — this is not a generic artifact system. It provides:

- atomic JSON reads/writes with a versioned envelope;
- content hashes for files and canonical-JSON objects;
- semantic fingerprints that EXCLUDE timestamps and locator paths, so moving a
  run directory or re-stamping a manifest never changes identity;
- input references (relative path + sha256) with validation that refuses
  mismatched inputs instead of silently proceeding.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from subspaces.paths import ProjectPaths

# Keys stripped (recursively, at any depth) before computing a semantic
# fingerprint: creation stamps, code locators, and filesystem locations.
# Locator-style keys (matrix_dir, cache_root) are excluded so MOVING an
# artifact never changes its identity — content hashes carry the identity.
# NOTE: content-identifying reference keys must NOT be volatile — refs use
# "content_fingerprint"/"semantic_fingerprint"/"sha256" so they participate in
# reuse identities (a bare "fingerprint" key was once stripped here, silently
# erasing z-cache refs from every reuse decision — audit finding, fixed).
VOLATILE_KEYS = frozenset(
    {
        "created_at",
        "code",
        "path",
        "matrix_dir",
        "reuse_path",
        "cache_root",
        # Presentation, not identity: the lineage-tree node display name
        # (subspaces.step1.tree). Renaming a node must never fork fingerprints —
        # the node hash key is the effective checkpoint content.
        "node_name",
        # Observability, not identity: peak VRAM depends on the device and
        # allocation history, so two scientifically identical regenerations
        # can differ — it must never enter fingerprints or lineage checks.
        "gpu_memory",
        # Derived checkpoints (subspaces.step1.matrix): the source locator and the
        # saved-file byte sha are non-contractual — the tensor content_digest
        # carries identity, so moving the source matrix or re-saving the mean
        # under a different torch version never forks the node.
        "source_matrix_dir",
        "file_sha256",
    }
)


class ArtifactError(RuntimeError):
    pass


# -- hashing ----------------------------------------------------------------


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: _strip_volatile(value)
            for key, value in obj.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(obj, list):
        return [_strip_volatile(item) for item in obj]
    return obj


def semantic_fingerprint(manifest: dict) -> str:
    """Identity hash of a manifest, ignoring timestamps and locator paths."""
    return content_hash(_strip_volatile(manifest))


def dataset_fingerprint(task_dir: str | Path) -> str:
    """Hash of a task directory: sorted (filename, file-sha256) pairs."""
    task_path = Path(task_dir)
    if not task_path.is_dir():
        raise ArtifactError(f"dataset dir not found: {task_path}")
    entries = sorted(
        (entry.name, sha256_file(entry))
        for entry in task_path.iterdir()
        if entry.is_file() and not entry.name.startswith(".")
    )
    if not entries:
        raise ArtifactError(f"dataset dir is empty: {task_path}")
    return content_hash(entries)


# -- git state ---------------------------------------------------------------


def git_state(root: Path) -> dict:
    """Actual commit + dirty status of the working tree (recorded, not hashed)."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": None, "dirty": None}


# -- envelope / IO -----------------------------------------------------------


def make_manifest(
    *,
    kind: str,
    schema_version: int,
    paths: ProjectPaths,
    config: dict | None = None,
    inputs: dict[str, dict] | None = None,
    payload: dict | None = None,
) -> dict:
    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": git_state(paths.root),
    }
    if config is not None:
        manifest["config"] = config
    if inputs is not None:
        manifest["inputs"] = inputs
    if payload is not None:
        manifest.update(payload)
    return manifest


def write_json_atomic(path: str | Path, obj: dict) -> None:
    """Atomic write via a UNIQUE same-directory temp file (concurrency-safe)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=False, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def read_manifest(
    path: str | Path, *, expect_kind: str, max_schema_version: int
) -> dict:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ArtifactError(f"artifact manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    kind = manifest.get("kind")
    if kind != expect_kind:
        raise ArtifactError(
            f"{manifest_path}: expected kind={expect_kind!r}, found {kind!r}"
        )
    version = manifest.get("schema_version")
    if not isinstance(version, int) or version < 1 or version > max_schema_version:
        raise ArtifactError(
            f"{manifest_path}: schema_version {version!r} not supported "
            f"(max {max_schema_version})"
        )
    return manifest


# Identity subset used for reuse decisions: what the artifact was derived
# from (inputs), under which settings (config), by which implementation.
IDENTITY_KEYS = ("kind", "schema_version", "inputs", "config", "impl")


def identity_of(manifest: dict, keys: tuple[str, ...] = IDENTITY_KEYS) -> str:
    """Semantic hash of the identity subset of a manifest."""
    subset = {key: manifest.get(key) for key in keys}
    return semantic_fingerprint(subset)


def reuse_or_refuse(
    path: str | Path,
    expected: dict,
    *,
    expect_kind: str,
    max_schema_version: int,
    keys: tuple[str, ...] = IDENTITY_KEYS,
) -> dict | None:
    """The single reuse rule for every substep.

    Returns None when no artifact exists (produce it), the existing manifest
    when its identity subset matches ``expected`` exactly (reuse it), and
    raises otherwise (refuse — never silently overwrite or accept a stale
    artifact).
    """
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    existing = read_manifest(
        artifact_path, expect_kind=expect_kind, max_schema_version=max_schema_version
    )
    existing_identity = identity_of(existing, keys)
    expected_identity = identity_of(expected, keys)
    if existing_identity != expected_identity:
        raise ArtifactError(
            f"{artifact_path} exists but was produced from different inputs/"
            f"config/implementation (identity {existing_identity[:12]} vs "
            f"expected {expected_identity[:12]}); refusing to reuse or "
            "overwrite. Use a new run or remove the artifact explicitly."
        )
    return existing


# -- input references ---------------------------------------------------------


def file_ref(path: str | Path, paths: ProjectPaths) -> dict:
    """Reference to an input file: relative locator + content hash."""
    resolved = paths.resolve(path)
    if not resolved.is_file():
        raise ArtifactError(f"referenced file not found: {resolved}")
    return {"path": paths.relativize(resolved), "sha256": sha256_file(resolved)}


def manifest_ref(path: str | Path, paths: ProjectPaths, manifest: dict) -> dict:
    """Reference to another artifact: locator + its semantic fingerprint."""
    resolved = paths.resolve(path)
    return {
        "path": paths.relativize(resolved),
        "semantic_fingerprint": semantic_fingerprint(manifest),
    }


def check_file_ref(ref: dict, paths: ProjectPaths, *, name: str = "input") -> Path:
    """Validate a file reference; refuse on hash mismatch."""
    resolved = paths.resolve(ref["path"])
    actual = sha256_file(resolved) if resolved.is_file() else None
    if actual != ref["sha256"]:
        raise ArtifactError(
            f"{name}: content mismatch for {resolved}\n"
            f"  expected sha256 {ref['sha256']}\n"
            f"  actual   sha256 {actual}\n"
            "Refusing to proceed with a changed or missing input."
        )
    return resolved


def check_manifest_ref(
    ref: dict, paths: ProjectPaths, *, expect_kind: str, max_schema_version: int
) -> dict:
    """Load a referenced artifact and refuse on fingerprint mismatch."""
    resolved = paths.resolve(ref["path"])
    manifest = read_manifest(
        resolved, expect_kind=expect_kind, max_schema_version=max_schema_version
    )
    actual = semantic_fingerprint(manifest)
    expected = ref.get("semantic_fingerprint")
    if expected is not None and actual != expected:
        raise ArtifactError(
            f"{expect_kind}: semantic fingerprint mismatch for {resolved}\n"
            f"  expected {expected}\n"
            f"  actual   {actual}\n"
            "The referenced artifact changed since it was linked."
        )
    return manifest
