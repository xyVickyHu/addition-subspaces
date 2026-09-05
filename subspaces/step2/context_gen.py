"""Generate a Step-2 analysis context from a Step-1 head artifact (CPU).

Walks the artifact lineage instead of guessing: heads artifact ->
``inputs.head_scan.path`` -> ``head_scan.json`` ->
``inputs.z_cache.content_fingerprint`` -> ``log/cache/z/<fp16>/meta.json``
(model identity, dataset fingerprint, activation-samples fingerprint). The
task family / prompt format / n_shot come from the activation samples
manifest matched by fingerprint under ``log/cache/samples/``.

The held-out z cache (the cell's ``heldout_activation`` extraction, created
by evaluate-headset) is discovered as the unique cache sharing the train
cache's identity apart from the sample manifest, with a DISJOINT task set
AND the SAME prompt protocol (prompt_format, n_shot) — without the protocol
filter, every other format/n-shot cell's held-out cache of the same family
qualifies (and one wrong pick shifts the statistic; the merge guard in
:mod:`subspaces.step2.pca` refuses such mixes at run time as well).

The generated mapping is an ordinary context file (``subspaces/context.py``) and
carries no identity of its own.
"""

from __future__ import annotations

import json
from pathlib import Path

from subspaces.artifacts import ArtifactError, modernize, semantic_fingerprint
from subspaces.head_sets import read_heads_manifest
from subspaces.paths import ProjectPaths
from subspaces.step2.pca import samples_protocol_index


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return modernize(json.load(fh))


def walk_to_scan(paths: ProjectPaths, artifact_path: Path) -> dict:
    """heads.json -> main_heads.json -> head_scan.json (or directly)."""
    manifest, kind = read_heads_manifest(artifact_path)
    if kind == "heads":
        main_ref = (manifest.get("inputs") or {}).get("main_heads")
        if not main_ref:
            raise ArtifactError(f"{artifact_path}: no inputs.main_heads reference")
        manifest = _load_json(paths.resolve(main_ref["path"]))
    scan_ref = (manifest.get("inputs") or {}).get("head_scan")
    if not scan_ref:
        raise ArtifactError(f"{artifact_path}: no inputs.head_scan reference")
    return _load_json(paths.resolve(scan_ref["path"]))


def find_samples_manifest(paths: ProjectPaths, samples_fp: str) -> Path | None:
    for candidate in sorted(paths.samples_cache_dir.glob("*/*.json")):
        try:
            manifest = _load_json(candidate)
        except (OSError, json.JSONDecodeError):
            continue
        if semantic_fingerprint(manifest) == samples_fp:
            return candidate
    return None


def discover_heldout(
    paths: ProjectPaths, train_meta: dict, train_tasks: set[str]
) -> list[tuple[Path, dict | None]]:
    """Candidate held-out caches: same base identity (model/dataset/hook/
    impl), disjoint tasks, and the same prompt protocol as the train cache
    (see module docstring). Returns ``[(cache_dir, protocol_or_None), ...]``.
    """
    protocol_index = samples_protocol_index(paths)
    train_identity = train_meta["identity"]
    train_protocol = protocol_index.get(train_identity.get("samples"))
    base = {key: train_identity[key] for key in train_identity if key != "samples"}
    matches: list[tuple[Path, dict | None]] = []
    for meta_path in sorted(paths.z_cache_dir.glob("*/meta.json")):
        try:
            meta = _load_json(meta_path)
        except (OSError, json.JSONDecodeError):
            continue
        identity = meta.get("identity") or {}
        if identity.get("samples") == train_identity.get("samples"):
            continue  # the train cache itself
        candidate_base = {key: identity[key] for key in identity if key != "samples"}
        if candidate_base != base:
            continue
        tasks = set(meta.get("tasks") or [])
        if not tasks or tasks & train_tasks:
            continue
        protocol = protocol_index.get(identity.get("samples"))
        if (
            train_protocol is not None
            and protocol is not None
            and (protocol["prompt_format"], protocol["n_shot"])
            != (train_protocol["prompt_format"], train_protocol["n_shot"])
        ):
            continue
        matches.append((meta_path.parent, protocol))
    return matches


def build_context(
    paths: ProjectPaths,
    heads_artifact: str | Path,
    *,
    include_heldout: bool = False,
    heldout_fingerprint: str | None = None,
) -> dict:
    """Assemble a schema-v1 context mapping for one head artifact.

    Raises ``ArtifactError`` on missing lineage refs or ambiguous held-out
    discovery. The ``task`` block is left as ``None`` values (with a
    ``warnings`` entry) when no cached samples manifest matches the z cache's
    fingerprint — the caller decides whether to fill it by hand.
    """
    artifact_path = paths.resolve(heads_artifact)
    heads_manifest, _kind = read_heads_manifest(artifact_path)

    scan = walk_to_scan(paths, artifact_path)
    z_ref = (scan.get("inputs") or {}).get("z_cache") or {}
    train_fp = z_ref.get("content_fingerprint")
    if not train_fp:
        raise ArtifactError("head_scan.json carries no inputs.z_cache fingerprint")
    train_dir = paths.z_cache_dir / train_fp
    train_meta = _load_json(train_dir / "meta.json")
    identity = train_meta["identity"]

    activations: dict = {"train": {"path": paths.relativize(train_dir)}}
    if heldout_fingerprint:
        activations["heldout"] = {
            "path": paths.relativize(paths.z_cache_dir / heldout_fingerprint)
        }
    elif include_heldout:
        matches = discover_heldout(paths, train_meta, set(train_meta["tasks"]))
        if len(matches) != 1:
            described = ", ".join(
                f"{directory.name}"
                + (
                    f" [{proto['prompt_format']}/n_shot={proto['n_shot']}"
                    f"/{proto['sample_kind']}]"
                    if proto
                    else " [protocol unresolved]"
                )
                for directory, proto in matches
            )
            raise ArtifactError(
                f"held-out z-cache discovery found {len(matches)} same-protocol "
                f"candidates ({described or 'none'}); pass "
                "--heldout-fingerprint. (Wrong-protocol picks are refused at "
                "run time by the merge guard.)"
            )
        activations["heldout"] = {"path": paths.relativize(matches[0][0])}

    warnings: list[str] = []
    samples_path = find_samples_manifest(paths, identity["samples"])
    samples_block: dict = {}
    if samples_path is not None:
        samples_manifest = _load_json(samples_path)
        task_cfg = (samples_manifest.get("config") or {}).get("task") or {}
        task_block = {
            "family": task_cfg.get("dataset_dir"),
            "prompt_format": task_cfg.get("prompt_format"),
            "n_shot": task_cfg.get("n_shot"),
        }
        samples_block = {
            samples_manifest["config"]["sample_kind"]: {
                "path": paths.relativize(samples_path)
            }
        }
    else:
        warnings.append(
            "no samples manifest matches the z cache's fingerprint; fill "
            "context.task by hand."
        )
        task_block = {"family": None, "prompt_format": None, "n_shot": None}

    context = {
        "schema_version": 1,
        "model": dict(identity["model"]),
        "task": task_block,
        **({"samples": samples_block} if samples_block else {}),
        "activations": activations,
        "heads_artifact": {
            "path": paths.relativize(artifact_path),
            "semantic_fingerprint": semantic_fingerprint(heads_manifest),
        },
    }
    return {"context": context, "warnings": warnings}
