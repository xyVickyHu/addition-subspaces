"""Substep 6 — GPU evaluation of a selected head set (selector-dependent).

Under the corrected protocol this is the ONLY substep that touches the five
held-out tasks: it consumes the held-out activation cache (task-conditioned z
for the held-out tasks, prompts distinct from evaluation via a distinct seed)
and the held-out final-evaluation samples, produced only after selection.

Selected-set intervention (legacy math): for each held-out task ``t``

    FV(t) = sum_{h in main} z_heldout[t][h] + sum_{h in sig \\ main} mean_z_train[h]

evaluated on the final-evaluation prompts, alongside clean, full-significant
(held-out z, unit coefficients), and the raw training-time function vector
(``sum_{l,h in all heads} M_final[l,h] * z_heldout[t][l,h]``) on the SAME
prompts. ``M_final`` always comes from the final epoch checkpoint, even when
significant-head selection used a trailing-checkpoint mean. mean_z_train is
TRAIN-ONLY (from the selection activation cache). Every listed example is
evaluated; effective batch sizes recorded.

Artifact (kind ``headset_eval``): metrics + per-task breakdown + references
to main/significant/matrix artifacts, sample manifests, and both caches; the
per-example outcomes live in a sibling npz referenced by the manifest.
"""

from __future__ import annotations

from pathlib import Path

from subspaces.artifacts import ArtifactError, check_file_ref, make_manifest
from subspaces.config import Step1Config, to_dict
from subspaces.paths import ProjectPaths
from subspaces.step1.recovery import outcomes_content_sha

HEADSET_EVAL_SCHEMA_VERSION = 1
# v4: raw_coef is the all-head FV weighted by the FINAL checkpoint, rather
# than v3's significant-only FV weighted by the selection checkpoint.
IMPL = {"module": "subspaces.step1.headset_eval", "algorithm_version": 4}

# The optional projected arm (``proj_meanab``) is its own versioned estimator
# (project convention): the selected heads' task vectors are each projected onto
# the head's top-k PC subspace before summing, k = the step2_subspace
# artifact's pcs-to-threshold. Absent a subspace input the arm set and the
# eval identity are byte-identical to plain v4.
PROJECTION_ARM = "proj_meanab"


def projection_identity(subspace: dict) -> dict:
    """Identity-bearing description of the projected arm.

    The referenced step2_subspace artifact rides in ``inputs.pca_subspace``
    (semantic fingerprint pins caches/heads/method); this block pins the arm
    estimator itself. Paper §4 convention: mu + P P^T (v - mu), re-centered.
    """
    return {
        "name": "pca_recentered_meanab",
        "version": 1,
        "k_source": "pcs_to_threshold",
        "threshold": subspace["config"]["threshold"],
    }


def identity_config(
    cfg: Step1Config, main_heads: dict, batch_size: int, projection: dict | None = None
) -> dict:
    """Identity-bearing headset settings, shared by planning and execution."""
    config = {
        "sites": to_dict(cfg.sites),
        "selector": {
            "name": main_heads["selector_name"],
            "version": main_heads["selector_version"],
        },
        "intervention_mode": 0,
        "batch_size": batch_size,
        "scoring": to_dict(cfg.scoring),
        "raw_coef": to_dict(cfg.raw_coef),
    }
    if projection is not None:
        config["projection"] = projection
    return config


def _fit_head_projector(x, head_key: str, artifact_entry: dict, threshold=None):
    """PCA basis for one head, re-derived from the same merged per-task rows
    the step2_subspace artifact was computed from (the step-2 math verbatim:
    sklearn ``PCA()`` on float32 rows) and verified against the artifact's
    recorded cumulative-EVR curve — a drifted cache or PCA implementation is
    refused instead of silently projecting onto a different subspace.

    Returns ``(mean, components)`` as float32 numpy arrays with ``components``
    of shape ``(d_model, k)``, k = the artifact's ``pcs_to_threshold``.
    """
    import numpy as np
    from sklearn.decomposition import PCA

    pca = PCA().fit(x)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    recorded = np.asarray(artifact_entry["pca_cumvar"], dtype=np.float64)
    # atol 1e-5: ~50x above measured cross-CPU BLAS jitter on these curves,
    # ~1000x below any real fit-set/implementation difference (>= 1e-2).
    if len(recorded) != len(cumvar) or not np.allclose(
        cumvar, recorded, atol=1e-5, rtol=0.0
    ):
        raise ArtifactError(
            f"re-derived PCA for head {head_key} does not reproduce the "
            "step2_subspace artifact's cumulative-EVR curve; the z caches or "
            "the PCA implementation drifted — refusing to project."
        )
    k = int(artifact_entry["pcs_to_threshold"])
    if not 1 <= k <= len(cumvar):
        raise ArtifactError(
            f"step2_subspace pcs_to_threshold for head {head_key} out of "
            f"range: {k} (n_components={len(cumvar)})"
        )
    if threshold is not None:
        from subspaces.step2.pca import pcs_from_cumvar

        # payload-internal consistency (recorded curve, no float jitter):
        # the artifact's k must be what its own threshold implies.
        expected_k = pcs_from_cumvar(recorded, float(threshold))
        if k != expected_k:
            raise ArtifactError(
                f"step2_subspace pcs_to_threshold for head {head_key} ({k}) "
                f"is not what its recorded threshold {threshold} implies on "
                f"the recorded curve ({expected_k}); refusing a tampered or "
                "mislabeled subspace artifact."
            )
    components = pca.components_[:k].T.astype(np.float32)
    return pca.mean_.astype(np.float32), components


def projected_headset_vectors(
    subspace: dict, train_z: dict, z_heldout: dict, selected: list, eval_tasks: list
) -> tuple[dict, dict]:
    """Per-held-out-task sum of the selected heads' PROJECTED task vectors.

    Paper §4 causal-projection convention per head: subtract the head's
    across-task mean, project onto the top-k PC subspace, re-center
    (``mu + P P^T (v - mu)``), unit coefficient. Float32 throughout — the
    centering subtracts two nearly identical vectors, which bf16 cannot
    represent. The PCA fit set must be EXACTLY the artifact's recorded task
    order (merged train + held-out caches), else refuse.

    Returns ``(vectors_by_task, per_head_k)``.
    """
    import numpy as np
    import torch

    if not selected:
        raise ArtifactError(
            "the projected arm needs a non-empty selected head set; an empty "
            "selection has no projected sum (refusing rather than degrading "
            "to the mean-term-only vector)."
        )
    merged = dict(train_z)
    overlap = sorted(set(merged) & set(z_heldout))
    if overlap:
        raise ArtifactError(f"train/held-out z caches share tasks: {overlap}; refusing")
    merged.update(z_heldout)
    fit_order = list(subspace["task_order"])
    missing = [task for task in fit_order if task not in merged]
    extra = sorted(set(merged) - set(fit_order))
    if missing or extra:
        raise ArtifactError(
            "the step2_subspace artifact's fit tasks do not match this "
            f"evaluation's merged z caches (missing: {missing}, extra: "
            f"{extra}); refusing to project against a different fit set."
        )
    per_head = subspace.get("per_head") or {}
    sums: dict[str, object] = {task: None for task in eval_tasks}
    per_head_k: dict[str, int] = {}
    for layer_idx, head_idx in selected:
        key = f"{layer_idx}:{head_idx}"
        entry = per_head.get(key)
        if entry is None:
            raise ArtifactError(
                f"step2_subspace artifact has no PCA entry for selected head "
                f"{key}; recompute step 2 over the full selected set."
            )
        x = (
            torch.stack(
                [merged[task][layer_idx, head_idx] for task in fit_order], dim=0
            )
            .to(torch.float32)
            .numpy()
        )
        mean, components = _fit_head_projector(
            x, key, entry, threshold=(subspace.get("config") or {}).get("threshold")
        )
        per_head_k[key] = int(components.shape[1])
        for task in eval_tasks:
            vector = z_heldout[task][layer_idx, head_idx].to(torch.float32).numpy()
            centered = vector - mean
            projected = mean + components @ (components.T @ centered)
            sums[task] = projected if sums[task] is None else sums[task] + projected
    vectors = {
        task: torch.from_numpy(np.asarray(total, dtype=np.float32))
        for task, total in sums.items()
    }
    return vectors, per_head_k


def all_head_raw_coef_vector(matrix, z):
    """Return ``sum_{l,h} matrix[l,h] * z[l,h]`` over the full head grid."""
    if matrix.ndim != 2:
        raise ArtifactError(
            f"raw_coef checkpoint must be a 2-D head matrix; got {tuple(matrix.shape)}"
        )
    if z.ndim != 3 or tuple(z.shape[:2]) != tuple(matrix.shape):
        raise ArtifactError(
            "raw_coef matrix/z head-grid mismatch: "
            f"matrix={tuple(matrix.shape)}, z={tuple(z.shape)}"
        )
    coefficients = matrix.to(device=z.device, dtype=z.dtype)
    return (coefficients.unsqueeze(-1) * z).sum(dim=(0, 1))


def evaluate_headset(
    main_heads: dict,
    significant: dict,
    matrix_ref: dict,
    train_mean_z,
    heldout_activation_cache: dict,
    final_eval_samples: dict,
    cfg: Step1Config,
    paths: ProjectPaths,
    out_dir: Path,
    *,
    resolved: dict,
    inputs_refs: dict,
    model_loader,
    artifact_stem: str,
    subspace: dict | None = None,
    train_z: dict | None = None,
) -> dict:
    """Evaluate the selected set; returns the ``headset_eval`` manifest.

    ``artifact_stem`` (complete identity hash) names BOTH the manifest and its
    outcomes npz, so variants never clobber each other's outcome files.
    With ``subspace`` (a step2_subspace manifest; requires ``train_z``, the
    train z cache for the PCA fit rows), the projected arm ``proj_meanab``
    joins the arm set.
    """
    import numpy as np

    from subspaces.step1.eval_gpu import (
        effective_batch_size,
        eval_prompts,
        gpu_peak_memory,
        manifest_prompts,
        require_resolved_model,
    )

    if significant is None or matrix_ref is None:
        raise ArtifactError("evaluate-headset requires significant + matrix refs")
    if not heldout_activation_cache or not final_eval_samples:
        raise ArtifactError(
            "evaluate-headset requires the held-out activation cache and the "
            "final-evaluation sample manifest (created only after selection)."
        )
    require_resolved_model(resolved)

    z_heldout = heldout_activation_cache["z"]
    task_order = list(final_eval_samples["task_order"])
    missing = [task for task in task_order if task not in z_heldout]
    if missing:
        raise ArtifactError(f"held-out activation cache lacks z for: {missing}")

    sig_heads = [
        (int(layer_idx), int(head_idx))
        for layer_idx, head_idx, *_ in significant["heads"]
    ]
    raw_checkpoint_ref = inputs_refs.get("raw_coef_checkpoint")
    if not raw_checkpoint_ref:
        raise ArtifactError(
            "evaluate-headset requires an explicit raw_coef_checkpoint input"
        )
    raw_checkpoint = check_file_ref(
        raw_checkpoint_ref, paths, name="raw_coef final checkpoint"
    )
    import torch

    raw_matrix = torch.load(raw_checkpoint, map_location="cpu")
    if not isinstance(raw_matrix, torch.Tensor):
        raise ArtifactError(
            f"raw_coef final checkpoint {raw_checkpoint} did not contain a Tensor"
        )
    raw_matrix = raw_matrix.detach().cpu()
    selected = [
        (int(layer_idx), int(head_idx))
        for layer_idx, head_idx in main_heads["main_heads"]
    ]
    not_selected = [head for head in sig_heads if head not in selected]
    layer_name = cfg.sites.inject_layer
    batch_size = effective_batch_size(cfg)

    projected_vectors: dict | None = None
    per_head_k: dict[str, int] | None = None
    if subspace is not None:
        if train_z is None:
            raise ArtifactError(
                "the projected arm needs the train z cache for its PCA fit "
                "rows; pass train_z alongside subspace."
            )
        projected_vectors, per_head_k = projected_headset_vectors(
            subspace, train_z, z_heldout, selected, task_order
        )

    model = model_loader()
    mean_term = sum(
        train_mean_z[layer_idx, head_idx] for layer_idx, head_idx in not_selected
    )
    # float32 twin for the projected arm (its per-head math is float32; the
    # unprojected arms keep the legacy bf16 accumulation).
    mean_term_f32 = (
        mean_term.to(torch.float32) if torch.is_tensor(mean_term) else mean_term
    )

    outcomes: dict[str, list[int]] = {
        "clean": [],
        "full_sig": [],
        "raw_coef": [],
        "main_meanab": [],
    }
    if projected_vectors is not None:
        outcomes[PROJECTION_ARM] = []
    per_task: dict[str, dict] = {}
    all_batch_sizes: list[int] = []
    for task in task_order:
        # clean on n-shot prompts; interventions on the paired zero-shot prompts
        nshot = manifest_prompts(final_eval_samples, task)
        zeroshot = manifest_prompts(final_eval_samples, task, zero_shot=True)
        vectors = {
            "clean": None,
            "full_sig": sum(
                z_heldout[task][layer_idx, head_idx]
                for layer_idx, head_idx in sig_heads
            ),
            "raw_coef": all_head_raw_coef_vector(raw_matrix, z_heldout[task]),
            "main_meanab": sum(
                z_heldout[task][layer_idx, head_idx] for layer_idx, head_idx in selected
            )
            + mean_term,
        }
        if projected_vectors is not None:
            vectors[PROJECTION_ARM] = projected_vectors[task] + mean_term_f32
        task_metrics = {}
        for variant, vector in vectors.items():
            prompts, targets = nshot if vector is None else zeroshot
            correct, batch_sizes = eval_prompts(
                model,
                prompts,
                targets,
                layer_name=layer_name if vector is not None else None,
                vector=vector,
                batch_size=batch_size,
            )
            outcomes[variant].extend(correct)
            all_batch_sizes.extend(batch_sizes)
            task_metrics[variant] = float(np.mean(correct))
        task_metrics["n"] = len(prompts)
        per_task[task] = task_metrics

    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = out_dir / f"{artifact_stem}-outcomes.npz"
    np.savez_compressed(
        outcomes_path,
        **{key: np.asarray(values, dtype=np.uint8) for key, values in outcomes.items()},
    )

    metrics = {variant: float(np.mean(values)) for variant, values in outcomes.items()}
    # macro (mean of per-task means) alongside pooled micro; identical when
    # per-task counts are equal, distinct for uneven families
    metrics_macro = {
        variant: float(np.mean([per_task[task][variant] for task in task_order]))
        for variant in outcomes
    }
    payload_projection = None
    if subspace is not None:
        payload_projection = {
            **projection_identity(subspace),
            "per_head_k": per_head_k,
            "n_fit_tasks": len(subspace["task_order"]),
        }
    return make_manifest(
        kind="headset_eval",
        schema_version=HEADSET_EVAL_SCHEMA_VERSION,
        paths=paths,
        config=identity_config(
            cfg,
            main_heads,
            batch_size,
            projection=projection_identity(subspace) if subspace is not None else None,
        ),
        inputs=inputs_refs,
        payload={
            "impl": IMPL,
            "protocol": cfg.protocol,
            "main_heads": [list(head) for head in selected],
            "n_significant": len(sig_heads),
            "n_raw_coef_heads": int(raw_matrix.numel()),
            **({"projection": payload_projection} if payload_projection else {}),
            "metrics": metrics,
            "metrics_macro": metrics_macro,
            "per_task": per_task,
            "task_order": task_order,
            "n_eval_total": sum(data["n"] for data in per_task.values()),
            "effective_batch_sizes": sorted(set(all_batch_sizes)),
            "gpu_memory": gpu_peak_memory(),
            "outcomes": {
                "file": outcomes_path.name,
                "content_sha256": outcomes_content_sha(outcomes),
            },
        },
    )
