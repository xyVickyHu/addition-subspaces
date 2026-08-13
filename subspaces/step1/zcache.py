"""Fingerprinted activation (z) caches under ``log/cache/z/<fingerprint>/``.

A cache is identified by everything that determines its content: model name +
revision, dataset fingerprint, the SAMPLE MANIFEST's semantic identity (which
already pins task list, prompt format, n_shot, examples-per-task, seed, and
sampling mode), dtype, hook site, and the extraction implementation version.
``meta.json`` records the identity; loading refuses on mismatch. This retires
the legacy model-collision hazard (the old cache filename omitted the model).
"""

from __future__ import annotations

from pathlib import Path

from subspaces.artifacts import (
    ArtifactError,
    content_hash,
    make_manifest,
    read_manifest,
    semantic_fingerprint,
    write_json_atomic,
)
from subspaces.config import Step1Config
from subspaces.paths import ProjectPaths

ZCACHE_SCHEMA_VERSION = 1
IMPL = {"module": "subspaces.step1.zcache", "algorithm_version": 1}
HOOK_SITE = "attn.hook_result[last_token]"


def cache_identity(cfg: Step1Config, resolved: dict, samples_manifest: dict) -> dict:
    return {
        "model": {
            "name": cfg.model.name,
            "revision": resolved["identity"]["model_revision"],
            "dtype": cfg.model.dtype,
        },
        "dataset_fingerprint": resolved["identity"]["dataset_fingerprint"],
        "samples": semantic_fingerprint(samples_manifest),
        "hook_site": HOOK_SITE,
        "impl": IMPL,
    }


def cache_fingerprint(identity: dict) -> str:
    return content_hash(identity)[:16]


def tensors_content_sha(z_results: dict) -> str:
    """Deterministic digest of the z tensors (payload integrity for the cache;
    bf16 is upcast to float32 for portable byte extraction)."""
    import hashlib

    import torch  # heavy

    digest = hashlib.sha256()
    for task in sorted(z_results):
        tensor = z_results[task].detach().cpu().contiguous()
        digest.update(task.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.to(torch.float32).numpy().tobytes())
    return digest.hexdigest()


def cache_dir(paths: ProjectPaths, fingerprint: str) -> Path:
    return paths.z_cache_dir / fingerprint


def load_zcache(paths: ProjectPaths, identity: dict):
    """Return the cached z dict, or None if absent. Refuses identity clashes."""
    import torch  # heavy

    fingerprint = cache_fingerprint(identity)
    directory = cache_dir(paths, fingerprint)
    meta_path = directory / "meta.json"
    tensor_path = directory / "z_results.pth"
    if not meta_path.is_file() or not tensor_path.is_file():
        return None
    meta = read_manifest(
        meta_path, expect_kind="z_cache", max_schema_version=ZCACHE_SCHEMA_VERSION
    )
    if meta.get("identity") != identity:
        raise ArtifactError(
            f"{directory}: cached z identity does not match the requested "
            "identity despite an identical fingerprint; refusing."
        )
    z_results = torch.load(tensor_path, map_location="cpu")
    recorded = meta.get("content_sha256")
    if recorded is not None and tensors_content_sha(z_results) != recorded:
        raise ArtifactError(
            f"{directory}: z tensor payload does not match the recorded "
            "content digest; refusing a corrupted cache."
        )
    return z_results


def load_zcache_dir(directory: str | Path) -> tuple[dict, dict]:
    """Load a z cache BY DIRECTORY (no identity reconstruction) with full
    self-consistency checks: manifest kind/schema, the directory name and
    recorded fingerprint must both equal the identity's fingerprint, and the
    tensor payload must match the recorded content digest. CPU-only.

    Returns ``(z_results, meta)``. This is the Step-2/3 access path — walk an
    artifact's ``inputs.z_cache.content_fingerprint`` to the cache dir and
    load it here, instead of rebuilding the identity dict from a config.
    """
    import torch  # heavy

    directory = Path(directory)
    meta_path = directory / "meta.json"
    tensor_path = directory / "z_results.pth"
    if not meta_path.is_file() or not tensor_path.is_file():
        raise ArtifactError(f"{directory}: not a z cache (meta.json/z_results.pth)")
    meta = read_manifest(
        meta_path, expect_kind="z_cache", max_schema_version=ZCACHE_SCHEMA_VERSION
    )
    expected_fp = cache_fingerprint(meta["identity"])
    if meta.get("cache_fingerprint") != expected_fp or directory.name != expected_fp:
        raise ArtifactError(
            f"{directory}: cache fingerprint does not match its identity "
            f"(expected {expected_fp}); refusing a relabeled cache."
        )
    try:
        z_results = torch.load(tensor_path, map_location="cpu")
    except Exception as err:  # torch.load raises many concrete types
        raise ArtifactError(
            f"{directory}: z_results.pth is unreadable or corrupted: {err}"
        ) from err
    if not isinstance(z_results, dict) or not all(
        isinstance(value, torch.Tensor) for value in z_results.values()
    ):
        raise ArtifactError(
            f"{directory}: z_results.pth is not a task -> tensor dict; "
            "refusing a malformed cache."
        )
    # store_zcache always records content_sha256; a missing digest here means
    # a hand-edited meta.json, so the by-directory path treats it as mandatory
    # (unlike load_zcache, which keeps its legacy tolerance).
    recorded = meta.get("content_sha256")
    if recorded is None:
        raise ArtifactError(
            f"{directory}: meta.json lacks content_sha256; refusing a cache "
            "without a payload digest."
        )
    if tensors_content_sha(z_results) != recorded:
        raise ArtifactError(
            f"{directory}: z tensor payload does not match the recorded "
            "content digest; refusing a corrupted cache."
        )
    if sorted(z_results) != meta.get("tasks"):
        raise ArtifactError(
            f"{directory}: meta.json task list is missing or does not match "
            "the tensor payload; refusing an inconsistent cache."
        )
    return z_results, meta


def store_zcache(paths: ProjectPaths, identity: dict, z_results: dict) -> str:
    import os
    import tempfile

    import torch  # heavy

    fingerprint = cache_fingerprint(identity)
    directory = cache_dir(paths, fingerprint)
    directory.mkdir(parents=True, exist_ok=True)
    # atomic tensor write: unique temp file in the same directory, then rename
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".z_results.", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(z_results, tmp_name)
        os.replace(tmp_name, directory / "z_results.pth")
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    meta = make_manifest(
        kind="z_cache",
        schema_version=ZCACHE_SCHEMA_VERSION,
        paths=paths,
        payload={
            "impl": IMPL,
            "identity": identity,
            "cache_fingerprint": fingerprint,
            "content_sha256": tensors_content_sha(z_results),
            "tasks": sorted(z_results),
        },
    )
    write_json_atomic(directory / "meta.json", meta)
    return fingerprint


def ensure_zcache(
    model_loader,
    samples_manifest: dict,
    cfg: Step1Config,
    paths: ProjectPaths,
    resolved: dict,
):
    """Load or compute the z cache for one sample manifest.

    ``model_loader`` is a zero-arg callable returning the loaded model — only
    invoked on a cache miss, so cache hits stay CPU-only.
    Returns ``(z_results, fingerprint)``.
    """
    from subspaces.step1.eval_gpu import compute_z_for_tasks

    identity = cache_identity(cfg, resolved, samples_manifest)
    cached = load_zcache(paths, identity)
    if cached is not None:
        return cached, cache_fingerprint(identity)
    model = model_loader()
    task_ids = list(samples_manifest["task_order"])
    z_results = compute_z_for_tasks(model, samples_manifest, task_ids)
    fingerprint = store_zcache(paths, identity, z_results)
    return z_results, fingerprint
