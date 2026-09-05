"""Minimal artifact plumbing: versioned manifests, atomic writes, hashes, refs.

Deliberately small — this is not a generic artifact system. It provides:

- atomic JSON reads/writes with a versioned envelope;
- content hashes for files and canonical-JSON objects;
- semantic fingerprints that EXCLUDE timestamps and locator paths, so moving a
  run directory or re-stamping a manifest never changes identity, and that are
  computed over the legacy head-set spelling (see the terminology block);
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


# -- head-set terminology (camera-ready rename, 2026-09) ----------------------
#
# The COLM 2026 camera-ready renamed the three nested head sets, and the code
# follows the paper: ``selected`` (sparse optimization; 33 on the paper matrix;
# formerly "significant"/"sig"), ``significant`` (paired McNemar + BH over the
# recovery scan; 13; formerly "recovery-positive"/"recpos"), ``main`` (the
# quarter-gain rule; 3; unchanged). Code, CLI, directory names and NEW
# manifests spell the paper vocabulary. Every manifest written before the
# rename spells its keys/kinds the old way, and every recorded fingerprint and
# identity was computed over that spelling. Identity must not fork on a
# spelling change (AGENTS.md: reuse identical work), so:
#
# - ``semantic_fingerprint``/``identity_content_hash`` hash the LEGACY spelling
#   (``canonicalize``): a manifest regenerated under the new spelling keeps its
#   identity and every recorded reference to it still verifies;
# - ``read_manifest`` translates legacy manifests to the new spelling on read
#   (``modernize``), so consumers see one vocabulary.
#
# These two tables are the single source of truth for the mapping. Extend them
# for every renamed persisted key/value. A modern spelling must never equal a
# key that already existed with another meaning (``n_selected`` — the count a
# selector chose — is such a key, which is why the selected-set size is
# ``n_selected_set`` in headset evaluations and ``n_scanned`` in selector
# verdicts); ``_respell`` refuses the resulting collision, and
# tests/test_artifacts.py pins the round trip.
_PKG = __name__.split(".")[0]
LEGACY_KEY_SPELLING: dict[str, str] = {  # modern key -> legacy key, any depth
    "selected": "significant",  # config slice / nodes summary
    "selected_heads": "significant_heads",  # artifact kind, input ref, payload
    "n_selected_set": "n_significant",  # headset-eval payload
    "n_scanned": "n_sig",  # selector verdicts
    "n_selected_heads": "n_sig_heads",  # head_scan payload
    "full_selected": "full_sig",  # headset-eval / scan arm name
    "full_selected_acc": "full_significant_acc",  # scan baselines
    "full_selected_fv_acc": "full_significant_fv_acc",  # legacy stage summary
    "A_selected_ref": "A_sig_ref",  # legacy stage summary weak_params
    "significant_main_heads": "recpos_main_heads",  # projection-causal input ref
    "headroom_vs_sumSelected": "headroom_vs_sumSig",  # recovery_weak decisions
    f"{_PKG}.step1.selected": f"{_PKG}.step1.significant",  # algorithm_versions
}
LEGACY_VALUE_SPELLING: dict[tuple[str, str], str] = {  # (key, modern) -> legacy
    ("kind", "selected_heads"): "significant_heads",
    ("module", f"{_PKG}.step1.selected"): f"{_PKG}.step1.significant",
}
MODERN_KEY_SPELLING = {legacy: modern for modern, legacy in LEGACY_KEY_SPELLING.items()}
MODERN_VALUE_SPELLING = {
    (key, legacy): modern for (key, modern), legacy in LEGACY_VALUE_SPELLING.items()
}


def _respell(obj: Any, key_map: dict, value_map: dict) -> Any:
    if isinstance(obj, dict):
        out: dict = {}
        for key, value in obj.items():
            new_key = key_map.get(key, key) if isinstance(key, str) else key
            if new_key in out:
                raise ArtifactError(
                    f"ambiguous head-set spelling: both {key!r} and {new_key!r} "
                    "present in one mapping"
                )
            if isinstance(value, str):
                value = value_map.get((new_key, value), value)
            else:
                value = _respell(value, key_map, value_map)
            out[new_key] = value
        return out
    if isinstance(obj, list):
        return [_respell(item, key_map, value_map) for item in obj]
    return obj


def canonicalize(obj: Any) -> Any:
    """Modern -> legacy spelling (the hashed form)."""
    return _respell(obj, LEGACY_KEY_SPELLING, LEGACY_VALUE_SPELLING)


def modernize(obj: Any) -> Any:
    """Legacy -> modern spelling (applied to every manifest read)."""
    return _respell(obj, MODERN_KEY_SPELLING, MODERN_VALUE_SPELLING)


def legacy_spellings(key: str) -> tuple[str, ...]:
    """The modern key followed by its legacy spelling (for array stores such
    as npz files, which are not translated on read)."""
    legacy = LEGACY_KEY_SPELLING.get(key)
    return (key,) if legacy is None else (key, legacy)


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
    """Identity hash of a manifest, ignoring timestamps and locator paths, and
    spelling-independent (hashed over the legacy head-set spelling)."""
    return content_hash(canonicalize(_strip_volatile(manifest)))


def identity_content_hash(obj: Any) -> str:
    """``content_hash`` over the canonical (legacy) head-set spelling — for
    identity records that are hashed verbatim (run identities)."""
    return content_hash(canonicalize(obj))


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
        manifest = modernize(json.load(fh))
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
