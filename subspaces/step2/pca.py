"""Step-2 core — task-generic activation-subspace PCA per head.

For each requested head, the input matrix ``X`` stacks the head's per-task
PROMPT-MEAN output vectors (one row per task; each row is the mean over the
sample manifest's prompts of the last-token ``attn.hook_result`` output, i.e.
the head's post-``W_O`` residual-stream contribution — exactly what the
fingerprinted z caches under ``log/cache/z/<fp16>/`` store). The analysis is
sklearn ``PCA()`` (mean-centered over tasks, full rank, float32) and the
recorded statistic is the 1-indexed count of principal components at which the
cumulative explained-variance ratio first reaches the threshold — the legacy
``stage3_pca`` formula ``int(np.searchsorted(cumvar, threshold) + 1)``
(``subspaces/runners/run_pipeline.py``), which at threshold 0.95 reproduces the
published ``pcs_at_95`` numbers (paper trio: 6/6/6 on the 30 add-k tasks).

Unlike the legacy runners this module is task-family agnostic: no add-k task
naming is assumed anywhere. Addition-specific analyses (periodic mod-vector
fits, causal subspace projection) are optional plugins and are NOT implemented
here.

Methods are a registry keyed by ``(name, version)`` (the
``subspaces/step1/selected.py`` pattern): the module-level ``algorithm_version``
is the shared-engine version, and each method's own ``(name, version)`` rides
in artifact identity via ``impl_for``. NOTE: this engine is deliberately NOT
registered in ``subspaces/step1/resolve.py::algorithm_versions()`` — that dict is
hashed into every Step-1 run identity.

CPU-only: given existing z caches, no model is loaded and no GPU is needed.
"""

from __future__ import annotations

from pathlib import Path

from subspaces.artifacts import (
    ArtifactError,
    identity_of,
    make_manifest,
    manifest_ref,
    modernize,
    reuse_or_refuse,
    write_json_atomic,
)
from subspaces.head_sets import read_heads_manifest, validate_heads
from subspaces.paths import ProjectPaths

STEP2_SUBSPACE_KIND = "step2_subspace"
STEP2_SUBSPACE_SCHEMA_VERSION = 1
ENGINE_VERSION = 1
IMPL = {"module": "subspaces.step2.pca", "algorithm_version": ENGINE_VERSION}

# Exact parameter schemas per (name, version) — mirrors
# subspaces/step1/selected.py::METHOD_PARAM_SCHEMAS.
METHOD_PARAM_SCHEMAS: dict[tuple[str, int], frozenset[str]] = {
    ("cumvar_threshold", 1): frozenset({"threshold"}),
}

# Frozen pre-tree flat-run layouts a defaulted output dir must never land in
# (same guard as subspaces/step1/pipeline.py::apply_selector and the compose CLI).
_FROZEN_PARENT_NAMES = ("step1", "main_heads")


def impl_for(cfg) -> dict:
    """Implementation identity: shared-engine version plus the method's own
    (name, version)."""
    return {
        **IMPL,
        "method": {"name": cfg.method, "version": cfg.method_version},
    }


def identity_config(cfg, heads: list[tuple[int, int]], head_set: str) -> dict:
    """The config slice that enters artifact identity: method, method_version,
    ONLY the parameters registered for that (name, version), the PCA report
    dimension, and the resolved head set. Heads are sorted so the identity is
    invariant to selector output order (per-head PCAs are independent)."""
    schema = METHOD_PARAM_SCHEMAS[(cfg.method, cfg.method_version)]
    identity = {"method": cfg.method, "method_version": cfg.method_version}
    for param in sorted(schema):
        identity[param] = getattr(cfg, param)
    identity["n_pcs"] = cfg.n_pcs
    identity["heads"] = [list(head) for head in sorted(heads)]
    identity["head_set"] = head_set
    return identity


# -- z-cache access ------------------------------------------------------------


def resolve_z_cache_dir(token: str, paths: ProjectPaths) -> Path:
    """Resolve one --z-cache token: a 16-hex fingerprint (looked up under
    ``log/cache/z/``) or a path to a cache directory."""
    text = str(token).strip()
    if len(text) == 16 and all(ch in "0123456789abcdef" for ch in text):
        return paths.z_cache_dir / text
    return paths.resolve(text)


def samples_protocol_index(paths: ProjectPaths) -> dict[str, dict]:
    """Map each cached sample manifest's semantic fingerprint to its protocol
    slice ``{sample_kind, prompt_format, n_shot}`` (from the manifest's own
    config block). Used to cross-check that merged z caches were extracted
    under ONE prompt protocol — the protocol lives only in the samples
    fingerprint, which the base-identity comparison deliberately strips."""
    import json

    from subspaces.artifacts import semantic_fingerprint

    index: dict[str, dict] = {}
    if not paths.samples_cache_dir.is_dir():
        return index
    for candidate in sorted(paths.samples_cache_dir.glob("*/*.json")):
        try:
            with open(candidate, encoding="utf-8") as fh:
                manifest = modernize(json.load(fh))
        except (OSError, json.JSONDecodeError):
            continue
        config = manifest.get("config") or {}
        task = config.get("task") or {}
        index[semantic_fingerprint(manifest)] = {
            "sample_kind": config.get("sample_kind"),
            "prompt_format": task.get("prompt_format"),
            "n_shot": task.get("n_shot"),
        }
    return index


def load_and_merge_zcaches(
    directories: list[Path], paths: ProjectPaths | None = None
) -> tuple[dict, dict]:
    """Load one or more fingerprinted z caches and merge their task dicts.

    All caches must share the extraction identity apart from the sample
    manifest (same model name/revision/dtype, dataset fingerprint, hook site,
    and extraction impl) and must cover DISJOINT task sets — e.g. the train
    ``activation`` cache plus the ``heldout_activation`` cache of one cell.
    When ``paths`` is given, each cache's samples fingerprint is additionally
    resolved against ``log/cache/samples/`` and caches whose resolved
    protocols (prompt_format, n_shot) differ REFUSE — the protocol lives only
    in the samples fingerprint, so the base-identity check cannot see it.

    Returns ``(z_results, info)`` where ``z_results`` maps task id to the
    per-task ``[n_layers, n_heads, d_model]`` prompt-mean tensor and ``info``
    records the shared identity, dims, task order, and per-cache provenance.
    """
    from subspaces.step1.zcache import load_zcache_dir

    if not directories:
        raise ArtifactError("no z cache to load")
    protocol_index = samples_protocol_index(paths) if paths is not None else {}
    merged: dict = {}
    caches: list[dict] = []
    shared: dict | None = None
    dims: tuple[int, int, int] | None = None
    for directory in directories:
        z_results, meta = load_zcache_dir(directory)
        identity = meta["identity"]
        base = {key: identity[key] for key in identity if key != "samples"}
        if shared is None:
            shared = base
        elif base != shared:
            raise ArtifactError(
                f"{directory}: z-cache identity is incompatible with "
                f"{directories[0]} (model/dataset/hook/impl must match; only "
                "the sample manifest may differ). Refusing to mix caches."
            )
        for task_id, tensor in z_results.items():
            if task_id in merged:
                raise ArtifactError(
                    f"task {task_id!r} appears in more than one z cache; "
                    "merged caches must cover disjoint task sets."
                )
            shape = tuple(tensor.shape)
            if len(shape) != 3:
                raise ArtifactError(
                    f"{directory}: task {task_id!r} tensor has shape {shape}, "
                    "expected [n_layers, n_heads, d_model]"
                )
            if dims is None:
                dims = shape
            elif shape != dims:
                raise ArtifactError(
                    f"{directory}: task {task_id!r} tensor shape {shape} "
                    f"differs from {dims}; refusing to mix caches."
                )
            merged[task_id] = tensor
        caches.append(
            {
                "content_fingerprint": meta["cache_fingerprint"],
                "samples": identity.get("samples"),
                "protocol": protocol_index.get(identity.get("samples")),
                "tasks": sorted(z_results),
            }
        )
    fingerprints = [cache["content_fingerprint"] for cache in caches]
    if len(set(fingerprints)) != len(fingerprints):
        raise ArtifactError(
            f"duplicate z-cache content fingerprints in the merge "
            f"({fingerprints}): identical identities can only coexist via a "
            "relabeled or forged cache; refusing."
        )
    resolved = [
        (cache["content_fingerprint"], cache["protocol"])
        for cache in caches
        if cache["protocol"] is not None
    ]
    distinct = {(proto["prompt_format"], proto["n_shot"]) for _fp, proto in resolved}
    if len(distinct) > 1:
        detail = "; ".join(
            f"{fp}: {proto['prompt_format']}/n_shot={proto['n_shot']}"
            f" ({proto['sample_kind']})"
            for fp, proto in resolved
        )
        raise ArtifactError(
            "merged z caches were extracted under DIFFERENT prompt protocols "
            f"— {detail}. A subspace analysis covers one model x dataset x "
            "protocol cell; refusing."
        )
    assert shared is not None and dims is not None
    info = {
        "model": shared.get("model"),
        "hook_site": shared.get("hook_site"),
        "dataset_fingerprint": shared.get("dataset_fingerprint"),
        "extraction_impl": shared.get("impl"),
        "dims": {"n_layers": dims[0], "n_heads": dims[1], "d_model": dims[2]},
        "task_order": sorted(merged),
        "z_caches": caches,
    }
    return merged, info


# -- the PCA statistic ---------------------------------------------------------


def pcs_from_cumvar(cumvar, threshold: float) -> int:
    """1-indexed FIRST component count whose cumulative explained-variance
    ratio is ``>= threshold`` — the legacy ``stage3_pca`` formula
    ``int(np.searchsorted(cumvar, threshold) + 1)`` (searchsorted
    side='left'), clamped to the component count for the
    float-rounding-only case where ``cumvar`` never reaches the threshold
    (legacy would report an out-of-range index there)."""
    import numpy as np

    position = int(np.searchsorted(cumvar, threshold) + 1)
    return min(position, len(cumvar))


def head_pca_summary(x, *, threshold: float, n_pcs: int) -> dict:
    """PCA summary of one head's per-task matrix ``x`` (n_tasks, d_model).

    Legacy-faithful math (``stage3_pca``): sklearn ``PCA()`` on float32 rows
    (implicit mean-centering over tasks), cumulative
    ``explained_variance_ratio_``, and ``pcs_to_threshold`` per
    :func:`pcs_from_cumvar`.
    """
    import numpy as np
    from sklearn.decomposition import PCA

    if x.shape[0] < 2:
        raise ArtifactError(
            f"PCA over tasks needs at least 2 task vectors, got {x.shape[0]}"
        )
    pca = PCA().fit(x)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    if not np.all(np.isfinite(cumvar)):
        raise ArtifactError(
            "non-finite explained-variance ratios (degenerate task vectors?); "
            "refusing to report a PC count."
        )
    return {
        "pcs_to_threshold": pcs_from_cumvar(cumvar, threshold),
        "n_components": int(len(cumvar)),
        "cumvar_at_n_pcs": float(cumvar[min(n_pcs, len(cumvar)) - 1]),
        "pca_cumvar": [float(value) for value in cumvar],
    }


# -- orchestration -------------------------------------------------------------


def default_out_dir(
    heads_artifact_path: Path | None,
    paths: ProjectPaths | None = None,
    kind_label: str = "step2_subspace",
) -> Path:
    """Default output dir: the heads artifact's node directory. Refuses frozen
    pre-tree layouts (same rule as select-main/compose) and explicit-heads
    runs, which have no parent node."""
    if heads_artifact_path is None:
        raise ArtifactError(
            "explicit --heads runs have no parent node: pass --out-dir to "
            f"place the {kind_label} artifact."
        )
    parent = Path(heads_artifact_path).parent
    # Frozen pre-tree runs keep everything under <run>/step1/ (incl. composed
    # heads at step1/heads/); flat-era selector variants live in main_heads/.
    # Only repo-relative components count — a checkout under a directory that
    # happens to be named step1 must not poison every default out-dir.
    try:
        checked_parts = (
            parent.resolve().relative_to(paths.root).parts
            if paths is not None
            else parent.parts
        )
    except ValueError:
        checked_parts = parent.parts
    if parent.name in _FROZEN_PARENT_NAMES or "step1" in checked_parts:
        raise ArtifactError(
            f"{heads_artifact_path} lives in a frozen pre-tree run: step 2 "
            "must not write into it; pass --out-dir to place the "
            f"{kind_label} artifact elsewhere."
        )
    return parent


def run_subspace_analysis(
    cfg,
    heads: list[tuple[int, int]],
    paths: ProjectPaths,
    *,
    head_set: str,
    heads_artifact: str | Path | None,
    z_cache_dirs: list[Path],
    out_dir: Path | None = None,
    plots: bool = False,
    expected_model: dict | None = None,
) -> tuple[dict, Path, bool]:
    """Compute (or reuse) one step2_subspace artifact.

    Returns ``(manifest, artifact_path, reused)``. The artifact is a
    hash-named single-slot sibling of the heads artifact
    (``subspace-<identity10>.json``, the ``eval-<h10>`` pattern), immutable
    under the shared reuse-or-refuse rule. ``expected_model`` (the context's
    declared model) is cross-checked against the caches' extraction identity
    so a mislabeled context cannot land another model's numbers in a cell.
    """
    import torch  # heavy import confined to the compute substep

    if (cfg.method, cfg.method_version) not in METHOD_PARAM_SCHEMAS:
        raise ArtifactError(
            f"unknown step2 PCA method ({cfg.method!r}, v{cfg.method_version}); "
            f"registered: {sorted(METHOD_PARAM_SCHEMAS)}"
        )
    z_results, info = load_and_merge_zcaches(z_cache_dirs, paths)
    cache_model = info["model"] or {}
    if expected_model:
        for field in ("name", "revision"):
            declared = expected_model.get(field)
            actual = cache_model.get(field)
            if declared and actual and declared != actual:
                raise ArtifactError(
                    f"context.model.{field} ({declared!r}) does not match the "
                    f"z caches' extraction identity ({actual!r}); refusing a "
                    "mislabeled analysis."
                )
    dims = info["dims"]
    validate_heads(heads, n_layers=dims["n_layers"], n_heads=dims["n_heads"])

    inputs: dict = {
        "z_caches": [
            {"content_fingerprint": cache["content_fingerprint"]}
            for cache in sorted(
                info["z_caches"], key=lambda c: c["content_fingerprint"]
            )
        ]
    }
    resolved_artifact: Path | None = None
    if heads_artifact is not None:
        resolved_artifact = paths.resolve(heads_artifact)
        heads_manifest, _kind = read_heads_manifest(resolved_artifact)
        inputs["heads_artifact"] = manifest_ref(
            resolved_artifact, paths, heads_manifest
        )

    expected = {
        "kind": STEP2_SUBSPACE_KIND,
        "schema_version": STEP2_SUBSPACE_SCHEMA_VERSION,
        "inputs": inputs,
        "config": identity_config(cfg, heads, head_set),
        "impl": impl_for(cfg),
    }
    target_dir = (
        paths.resolve(out_dir)
        if out_dir is not None
        else default_out_dir(resolved_artifact, paths)
    )
    stem = f"subspace-{identity_of(expected)[:10]}"
    out_path = target_dir / f"{stem}.json"
    existing = reuse_or_refuse(
        out_path,
        expected,
        expect_kind=STEP2_SUBSPACE_KIND,
        max_schema_version=expected["schema_version"],
    )
    if existing is not None:
        if plots:
            # plots carry no identity and are not referenced by the manifest;
            # regenerating them for a reused artifact is safe.
            _write_plots(
                existing.get("per_head") or {},
                target_dir / f"{stem}-plots",
                cfg.threshold,
            )
        return existing, out_path, True

    task_order = info["task_order"]
    per_head: dict[str, dict] = {}
    summary: dict[str, int] = {}
    for layer_idx, head_idx in sorted(heads):
        x = (
            torch.stack(
                [z_results[task][layer_idx, head_idx] for task in task_order], dim=0
            )
            .to(torch.float32)
            .numpy()
        )
        result = head_pca_summary(x, threshold=cfg.threshold, n_pcs=cfg.n_pcs)
        key = f"{layer_idx}:{head_idx}"
        per_head[key] = result
        summary[key] = result["pcs_to_threshold"]

    manifest = make_manifest(
        kind=STEP2_SUBSPACE_KIND,
        schema_version=expected["schema_version"],
        paths=paths,
        config=expected["config"],
        inputs=expected["inputs"],
        payload={
            "impl": expected["impl"],
            "model": info["model"],
            "hook_site": info["hook_site"],
            "dataset_fingerprint": info["dataset_fingerprint"],
            "dims": dims,
            "task_order": task_order,
            "n_tasks": len(task_order),
            "z_caches": info["z_caches"],
            "per_head": per_head,
            "pcs_to_threshold": summary,
        },
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_path, manifest)
    if plots:
        _write_plots(per_head, target_dir / f"{stem}-plots", cfg.threshold)
    return manifest, out_path, False


def _write_plots(per_head: dict, plots_dir: Path, threshold: float) -> None:
    """Per-head cumulative-variance plots. Never blocks the numeric artifact;
    plots carry no identity and are not referenced in the manifest."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        plots_dir.mkdir(parents=True, exist_ok=True)
        for key, result in per_head.items():
            layer_idx, head_idx = key.split(":")
            cumvar = result["pca_cumvar"]
            plt.figure()
            plt.plot(np.arange(1, len(cumvar) + 1), cumvar, marker="o")
            plt.axhline(threshold, linestyle="--", alpha=0.5)
            plt.title(f"PCA cumvar L{layer_idx}H{head_idx}")
            plt.xlabel("# PCs")
            plt.ylabel("Cumulative explained variance")
            plt.ylim(0, 1.01)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                str(plots_dir / f"pca_cumvar_L{layer_idx}H{head_idx}.png"), dpi=120
            )
            plt.close()
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"[step2] plot step skipped: {exc}")
