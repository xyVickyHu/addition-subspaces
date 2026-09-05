"""Runtime resolution: requested config -> resolved run identity.

The run ID is computed from NORMALIZED REQUESTED SEMANTICS plus STABLE
RESOLVED CONTENT IDs, before the run directory is created:

- included: protocol, task family/prompt format/n_shot, sample specs and
  seeds, scan grid, selected/selector schemas and parameters, sites, model
  name + dtype + EFFECTIVE revision (config pin, else cache-resolved),
  selected-checkpoint SHA + epoch, dataset and split fingerprints, and the
  substep algorithm versions;
- excluded: locators (``matrix.reuse_path``, paths), ``run_name``, compute
  settings, model device, and every cache-availability/status field.

Validation status is recorded SEPARATELY (observed tokenizer hash, cache root,
resolution status, environment versions, provenance details) and may be
updated between substeps without constituting identity drift — so a CPU
substep run without the HF cache and a later GPU substep with it land in the
same run when the model revision is pinned. Pin ``model.revision`` in
acceptance configs. For UNPINNED runs the effective revision participates in
the identity, so a cache-availability change starts a NEW run (a fork, not an
error) — pinning is what keeps a run stable across cache visibility. The
in-place drift refusal below guards a fixed run directory against tampering.

The run directory keeps ``requested_config.yaml`` (validated input, verbatim)
and ``resolved_config.yaml`` (input + the ``resolved`` block).
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import os
import platform
import tempfile
from pathlib import Path

import yaml

from subspaces.artifacts import (
    ArtifactError,
    dataset_fingerprint,
    identity_content_hash,
    sha256_file,
)
from subspaces.config import ConfigError, Step1Config, requested_semantics, to_dict
from subspaces.paths import ProjectPaths
from subspaces.step1.matrix import resolve_reuse_matrix
from subspaces.step1.split import TaskSplit

RUN_IDENTITY_SCHEMA_VERSION = 2

_VERSIONED_PACKAGES = ("torch", "transformers", "transformer-lens", "numpy", "scipy")


def algorithm_versions() -> dict[str, int]:
    """Implementation identity of every Step-1 substep, part of run identity
    (INCLUDING the shared GPU primitives and the z-cache layer — a version
    bump in either must fork run identities)."""
    from subspaces.step1 import (
        eval_gpu,
        headset_eval,
        matrix,
        recovery,
        samples,
        selected,
        zcache,
    )

    return {
        module.IMPL["module"]: module.IMPL["algorithm_version"]
        for module in (
            matrix,
            samples,
            selected,
            recovery,
            headset_eval,
            eval_gpu,
            zcache,
        )
    }


def _hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for env_var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(env_var)
        if not value:
            continue
        base = Path(value)
        roots.extend([base / "hub", base])
    return [root for root in roots if root.is_dir()]


def resolve_model_identity(model_name: str) -> dict:
    """Observed revision + tokenizer identity from the local HF cache."""
    slug = "models--" + model_name.replace("/", "--")
    for root in _hf_cache_roots():
        model_dir = root / slug
        if not model_dir.is_dir():
            continue
        revision = None
        ref_main = model_dir / "refs" / "main"
        if ref_main.is_file():
            revision = ref_main.read_text(encoding="utf-8").strip()
        else:
            snapshots = sorted((model_dir / "snapshots").glob("*"))
            if len(snapshots) == 1:
                revision = snapshots[0].name
        if revision is None:
            continue
        tokenizer_sha = None
        for name in ("tokenizer.json", "tokenizer_config.json"):
            candidate = model_dir / "snapshots" / revision / name
            if candidate.is_file():
                tokenizer_sha = {"file": name, "sha256": sha256_file(candidate)}
                break
        return {
            "name": model_name,
            "revision": revision,
            "tokenizer": tokenizer_sha,
            "status": "resolved",
            "cache_root": str(root),
        }
    return {
        "name": model_name,
        "revision": None,
        "tokenizer": None,
        "status": "unresolved",
        "cache_root": None,
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in _VERSIONED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def resolve_runtime(
    cfg: Step1Config, paths: ProjectPaths, split: TaskSplit | None
) -> dict:
    """Build the resolved block: ``identity`` (hashed into the run ID) +
    ``validation`` (recorded, updatable, never part of identity)."""
    env_format = os.environ.get("FV_PROMPT_FORMAT")
    if env_format and env_format != cfg.task.prompt_format:
        raise ConfigError(
            f"FV_PROMPT_FORMAT={env_format!r} contradicts task.prompt_format="
            f"{cfg.task.prompt_format!r}; unset the env var or fix the config."
        )

    observed_model = resolve_model_identity(cfg.model.name)
    if (
        cfg.model.revision is not None
        and observed_model["revision"] is not None
        and observed_model["revision"] != cfg.model.revision
    ):
        raise ConfigError(
            f"model.revision={cfg.model.revision} but the local cache resolves "
            f"to {observed_model['revision']}; refusing."
        )
    effective_revision = cfg.model.revision or observed_model["revision"]

    requested = requested_semantics(cfg)
    requested["model"].pop("revision", None)  # effective revision lives below

    task_dir = paths.task_dir(cfg.task.dataset_dir)
    identity: dict = {
        "requested": requested,
        "model_revision": effective_revision,
        "dataset_fingerprint": dataset_fingerprint(task_dir),
        "algorithm_versions": algorithm_versions(),
    }
    validation: dict = {
        "model": observed_model,
        "model_revision_source": (
            "config_pin" if cfg.model.revision is not None else observed_model["status"]
        ),
        "environment": _package_versions(),
        "batch_size": {
            "requested": cfg.compute.batch_size,
            "rule": "substep default: one batch per sample-manifest chunk; the "
            "effective batch size is recorded by each GPU substep at execution",
        },
    }
    if split is not None:
        identity["split_sha256"] = sha256_file(paths.resolve(cfg.task.task_split))
        validation["split"] = {
            "path": cfg.task.task_split,
            "n_train": len(split.train_tasks),
            "n_eval": len(split.eval_tasks),
        }
    if cfg.matrix.mode == "reuse":
        matrix_ref = resolve_reuse_matrix(cfg, paths)
        selected = matrix_ref["selected_checkpoint"]
        if selected.get("content_digest"):
            # derived checkpoint: the file sha is volatile (saved bytes are
            # not contractual) — identity is the tensor content digest plus
            # the derivation rule.
            identity["checkpoint"] = {
                "content_digest": selected["content_digest"],
                "rule": selected.get("selection_rule"),
                "k": selected.get("k"),
            }
        else:
            identity["checkpoint"] = {
                "sha256": selected["sha256"],
                "epoch": selected["epoch"],
            }
        validation["matrix_provenance"] = matrix_ref.get("provenance_check")
        validation["matrix_dir"] = matrix_ref["matrix_dir"]
    else:
        identity["checkpoint"] = "pending_training"

    return {
        "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "kind": "run_identity",
        "identity": identity,
        "validation": validation,
    }


def identity_hash(resolved: dict) -> str:
    return identity_content_hash(resolved["identity"])[:12]


def run_id_for(cfg: Step1Config, resolved: dict) -> str:
    return f"{cfg.run_name}__{identity_hash(resolved)}"


def write_resolved_config(cfg: Step1Config, resolved: dict, run_dir: Path) -> None:
    """Persist requested + resolved configs.

    Identity drift is refused; the validation block is allowed to change
    (e.g. an unresolved model identity becoming resolved once HF_HOME points
    at the cache) and is updated in place.
    """
    requested_path = run_dir / "requested_config.yaml"
    if not requested_path.exists():
        payload = {"schema_version": 1, **to_dict(cfg)}
        requested_path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )

    resolved_path = run_dir / "resolved_config.yaml"
    document = {"schema_version": 1, **to_dict(cfg), "resolved": resolved}
    if resolved_path.exists():
        existing = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        existing_resolved = existing.get("resolved", {})
        old_hash = identity_content_hash(existing_resolved.get("identity", {}))
        new_hash = identity_content_hash(resolved["identity"])
        if old_hash != new_hash:
            raise ArtifactError(
                f"{resolved_path}: resolved run identity changed since this "
                f"run was created ({old_hash[:12]} -> {new_hash[:12]}). An "
                "input (dataset, split, checkpoint, or unpinned model "
                "revision) moved under the run; refusing — start a new run, "
                "or pin model.revision if the change was cache availability."
            )
        if existing_resolved.get("validation") == resolved["validation"]:
            return  # nothing to update
    # Atomic replace (unique temp + os.replace, the write_json_atomic
    # discipline): an in-place truncate-write would mutate a hardlinked
    # journal shared with another worktree and could leave a torn file.
    fd, temp_name = tempfile.mkstemp(
        dir=resolved_path.parent, prefix=resolved_path.name, suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(yaml.safe_dump(document, sort_keys=False))
        os.replace(temp_name, resolved_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise
