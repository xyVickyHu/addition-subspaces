"""Step-2 API: task-generic per-head activation-subspace PCA.

For each requested head (typically a Step-1 MAIN head set), Step 2 runs PCA
over the head's per-task prompt-mean output vectors and records the number of
principal components at which the cumulative explained-variance ratio first
reaches the threshold (default 0.95 — the legacy ``pcs_at_95`` statistic).
The math and artifact contract live in :mod:`subspaces.step2.pca`.

Addition-specific analyses are opt-in plugins. ``--plugin periodic`` runs the
§4 mod-vector decomposition (:mod:`subspaces.step2.periodic`; add-k tasks only)
INSTEAD of the task-generic summary — its artifact is a sibling of the
subspace artifact, never a replacement. ``causal_projection`` remains a
refusing placeholder: the causal consumer of the subspace artifact is the
Step-1 headset-eval ``proj_meanab`` arm (step-2 methods notes, "Projected
intervention"), which needs the full eval context that lives in Step 1.

Input contract
--------------
- heads: an explicit ``L:H`` list OR a Step-1 head artifact — the composed
  ``heads.json`` or a selector's ``main_heads.json``
  (``--heads-artifact ... --head-set main|significant|minor``);
- activations: one or more fingerprinted z caches under ``log/cache/z/``
  (CLI ``--z-cache`` refs override ``context.activations``); merged caches
  must share model/dataset/hook identity and cover disjoint task sets;
- an analysis context (``--context``) is required for execution — it records
  the model, task family, prompt protocol, and reference fingerprints that
  make the run reproducible. ``--dry-run`` stays permissive.

Output contract
---------------
kind ``step2_subspace``: a hash-named single-slot sibling of the heads
artifact (``subspace-<identity10>.json``, default; ``--out-dir`` overrides),
with per-head cumulative variance and PCs-to-threshold, plus optional plots.
With ``--plugin periodic`` the run instead writes kind ``step2_periodic``
(``periodic-<identity10>.json`` + ``-modvectors.npz`` sidecar, sha256-pinned
in the manifest payload) — see :mod:`subspaces.step2.periodic`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from subspaces.context import AnalysisContext, ContextError
from subspaces.head_sets import Head, HeadSpecError, format_heads
from subspaces.paths import ProjectPaths
from subspaces.step2 import pca as pca_mod

STEP2_SCHEMA_VERSION = 1

KNOWN_PLUGINS = ("periodic", "causal_projection")


@dataclass
class Step2Config:
    n_pcs: int = 6
    # PCA rule: a registered (method, version) with its exact parameter set
    # (subspaces/step2/pca.py::METHOD_PARAM_SCHEMAS). cumvar_threshold v1 at 0.95
    # reproduces the legacy stage-3 pcs_at_95 statistic.
    method: str = "cumvar_threshold"
    method_version: int = 1
    threshold: float = 0.95
    # Addition-specific analyses are opt-in plugins, not defaults.
    plugins: list[str] = field(default_factory=list)
    # periodic plugin (subspaces/step2/periodic.py, method mod_fit v1): phase-search
    # grid step for opt_shift, and an optional ground-truth mod_vectors_dict
    # (.pth, {(l, h): (d_model, 6)}) to compare the fitted directions against.
    theta_step: float = 0.01
    gt_mod_vectors: str | None = None

    def validate(self) -> None:
        unknown = set(self.plugins) - set(KNOWN_PLUGINS)
        if unknown:
            raise ValueError(
                f"unknown step2 plugin(s) {sorted(unknown)}; known: {KNOWN_PLUGINS}"
            )
        if self.n_pcs < 1:
            raise ValueError(f"n_pcs must be >= 1: {self.n_pcs}")
        key = (self.method, self.method_version)
        if key not in pca_mod.METHOD_PARAM_SCHEMAS:
            raise ValueError(
                f"unknown step2 PCA method {key}; registered: "
                f"{sorted(pca_mod.METHOD_PARAM_SCHEMAS)}"
            )
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1]: {self.threshold}")
        if self.theta_step <= 0:
            raise ValueError(f"theta_step must be > 0: {self.theta_step}")
        if self.gt_mod_vectors is not None and "periodic" not in self.plugins:
            raise ValueError(
                "--gt-mod-vectors only applies to the periodic plugin: "
                "add --plugin periodic (the generic PCA summary would "
                "silently ignore it)"
            )


def describe(
    cfg: Step2Config,
    heads: list[Head],
    paths: ProjectPaths,
    context: AnalysisContext | None = None,
    z_caches: list[str] | None = None,
) -> str:
    """Human-readable dry-run plan; performs validation, writes nothing."""
    cfg.validate()
    if context is not None:
        context.validate()
    plugins = ", ".join(cfg.plugins) if cfg.plugins else "none"
    lines = [
        "step2 (dry run)",
        f"  heads:     {format_heads(heads)}",
        f"  method:    {cfg.method} v{cfg.method_version} "
        f"(threshold {cfg.threshold}, n_pcs {cfg.n_pcs})",
        f"  plugins:   {plugins}",
        f"  root:      {paths.root}",
    ]
    if "periodic" in cfg.plugins:
        lines.append(
            f"  periodic:  method mod_fit v1, theta_step {cfg.theta_step}, "
            f"gt_mod_vectors {cfg.gt_mod_vectors or 'none'}"
        )
    if z_caches:
        lines.append(f"  z-caches:  {', '.join(z_caches)} (override context)")
    if context is not None:
        lines.append("  context:")
        lines.append(context.describe())
    else:
        lines.append("  context: none (pass --context for a reproducible record)")
    return "\n".join(lines)


def run(
    cfg: Step2Config,
    heads: list[Head],
    paths: ProjectPaths,
    context: AnalysisContext | None = None,
    *,
    head_set: str = "main",
    heads_artifact: str | Path | None = None,
    z_caches: list[str] | None = None,
    out_dir: str | Path | None = None,
    plots: bool = False,
) -> dict:
    """Execute the Step-2 PCA and write (or reuse) one immutable artifact.

    ``z_caches`` (CLI ``--z-cache`` fingerprints/paths) override the context's
    ``activations`` references. ``--plugin periodic`` dispatches to the add-k
    §4 mod-vector fit (:mod:`subspaces.step2.periodic`) instead of the task-generic
    summary; ``causal_projection`` refuses before any write (its causal role
    is the Step-1 headset-eval ``proj_meanab`` arm).
    """
    cfg.validate()
    if context is None:
        raise ContextError(
            "step2 execution requires --context (model, task split, prompt "
            "protocol, sample manifests, activation refs); only --dry-run is "
            "permissive without one."
        )
    context.validate_refs(paths)
    if "causal_projection" in cfg.plugins:
        raise NotImplementedError(
            "step2 plugin 'causal_projection' is a refusing placeholder: the "
            "causal consumer of the subspace artifact is the Step-1 "
            "headset-eval projected arm (evaluate-headset --subspace, arm "
            "proj_meanab; 'Projected intervention') "
            "— it needs the full evaluation context, which lives in Step 1, "
            "not here. No artifacts were created."
        )
    if not heads:
        raise HeadSpecError("no heads to analyze (resolved head set is empty)")

    if z_caches:
        cache_dirs = [pca_mod.resolve_z_cache_dir(ref, paths) for ref in z_caches]
    else:
        cache_dirs = [
            paths.resolve(ref["path"]) for ref in context.activation_refs().values()
        ]
    if not cache_dirs:
        raise ContextError(
            "step2 execution needs at least one z cache: pass --z-cache "
            "<fingerprint|path> or set context.activations."
        )

    if "periodic" in cfg.plugins:
        from subspaces.step2 import periodic as periodic_mod

        manifest, out_path, reused = periodic_mod.run_periodic_analysis(
            cfg,
            heads,
            paths,
            head_set=head_set,
            heads_artifact=heads_artifact,
            z_cache_dirs=cache_dirs,
            out_dir=out_dir,
            plots=plots,
            expected_model=context.model,
        )
        verb = "reused" if reused else "wrote"
        print(f"[step2:periodic] {verb} {paths.relativize(out_path)}")
        mean_r2 = {
            key: round(record["mean_r2"], 4)
            for key, record in manifest["per_head"].items()
        }
        print(
            f"[step2:periodic] mod-fit mean R2 over {manifest['n_tasks']} "
            f"add-k tasks (n_pcs={cfg.n_pcs}): {mean_r2}"
        )
        return manifest

    manifest, out_path, reused = pca_mod.run_subspace_analysis(
        cfg,
        heads,
        paths,
        head_set=head_set,
        heads_artifact=heads_artifact,
        z_cache_dirs=cache_dirs,
        out_dir=out_dir,
        plots=plots,
        expected_model=context.model,
    )
    verb = "reused" if reused else "wrote"
    print(f"[step2] {verb} {paths.relativize(out_path)}")
    print(
        f"[step2] #PCs to reach {cfg.threshold:g} cumulative EVR "
        f"(n_tasks={manifest['n_tasks']}): {manifest['pcs_to_threshold']}"
    )
    return manifest
