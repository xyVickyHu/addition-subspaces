"""Substep 4 — per-head recovery scan + selector-independent baselines (GPU).

Inputs (explicit, all fingerprint-validated by the pipeline stage):

- the ``significant_heads`` artifact (which heads to scan; optionally capped
  by ``scan.head_limit`` — recorded in the artifact);
- the SCAN sample manifest (train tasks only under the corrected protocol);
- the SELECTION activation cache (train-task task-conditioned z; the
  train-only mean_z is derived from it — held-out tasks never enter here).

Per head ``(l, h)`` and integer ``c`` in the grid, for each task ``t`` (legacy
``recovery_scan`` math, unchanged):

    FV(c, h, t) = c * z[t][l, h] + (mean_fv - mean_z[l, h])

i.e. the scaled head keeps its task-conditioned value while every other
significant head sits at its cross-task mean. Baselines on the SAME prompts:
clean (no intervention), full-significant unit-coefficient FV, and the global
mean-FV. Every listed example is evaluated (no drops); effective batch sizes
are recorded.

Artifacts:

- ``step1/head_scan.json`` (kind ``head_scan``): curves, baselines, actual
  per-task denominators, c grid, references;
- ``step1/scan_outcomes.npz``: paired per-example 0/1 outcomes per (head, c)
  plus clean and full-significant vectors — internal (referenced by
  ``head_scan.json``), never exposed through the heads artifact.
"""

from __future__ import annotations

from pathlib import Path

from subspaces.artifacts import ArtifactError, make_manifest
from subspaces.config import Step1Config, to_dict
from subspaces.paths import ProjectPaths

HEAD_SCAN_SCHEMA_VERSION = 1
IMPL = {"module": "subspaces.step1.recovery", "algorithm_version": 2}

OUTCOMES_FILE = "scan_outcomes.npz"


def outcomes_content_sha(outcomes: dict) -> str:
    """Deterministic digest of the outcome ARRAYS (npz bytes embed zip
    timestamps and are not reproducible)."""
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(outcomes):
        digest.update(key.encode())
        digest.update(bytes(bytearray(outcomes[key])))
    return digest.hexdigest()


def scanned_heads(significant: dict, head_limit: int | None) -> list[tuple[int, int]]:
    heads = [
        (int(layer_idx), int(head_idx))
        for layer_idx, head_idx, *_ in significant["heads"]
    ]
    return heads[:head_limit] if head_limit else heads


def run_scan(
    significant: dict,
    scan_samples: dict,
    activation_cache: dict,
    cfg: Step1Config,
    paths: ProjectPaths,
    out_dir: Path,
    *,
    resolved: dict,
    inputs_refs: dict,
    model_loader,
) -> dict:
    """Execute the scan; returns the ``head_scan`` manifest (outcomes on disk)."""
    import numpy as np

    from subspaces.step1.eval_gpu import (
        effective_batch_size,
        eval_prompts,
        gpu_peak_memory,
        manifest_prompts,
        mean_z_over_tasks,
        require_resolved_model,
    )

    if not scan_samples or not activation_cache:
        raise ArtifactError(
            "scan requires the scan-sample manifest and the selection "
            "activation cache; run make-samples first (the composed run wires "
            "these automatically)."
        )
    require_resolved_model(resolved)

    z_results = activation_cache["z"]
    task_order = list(scan_samples["task_order"])
    missing = [task for task in task_order if task not in z_results]
    if missing:
        raise ArtifactError(f"activation cache lacks task z for: {missing}")

    sig_heads = [
        (int(layer_idx), int(head_idx))
        for layer_idx, head_idx, *_ in significant["heads"]
    ]
    heads = scanned_heads(significant, cfg.scan.head_limit)
    c_grid = list(range(cfg.scan.c_min, cfg.scan.c_max + 1))
    layer_name = cfg.sites.inject_layer
    batch_size = effective_batch_size(cfg)

    model = model_loader()
    mean_z = mean_z_over_tasks({task: z_results[task] for task in task_order})
    mean_fv = sum(mean_z[layer_idx, head_idx] for layer_idx, head_idx in sig_heads)

    # Legacy paradigm: clean accuracy on the N-SHOT prompts; every FV
    # intervention on the ZERO-SHOT prompts of the SAME examples (paired).
    per_task_nshot = {task: manifest_prompts(scan_samples, task) for task in task_order}
    per_task_zeroshot = {
        task: manifest_prompts(scan_samples, task, zero_shot=True)
        for task in task_order
    }
    counts = {task: len(per_task_nshot[task][0]) for task in task_order}
    n_eval_total = sum(counts.values())
    all_batch_sizes: list[int] = []
    outcomes: dict[str, list[int]] = {}

    def eval_over_tasks(vector_for_task) -> list[int]:
        collected: list[int] = []
        for task in task_order:
            vector = vector_for_task(task)
            if vector is None:
                prompts, targets = per_task_nshot[task]
            else:
                prompts, targets = per_task_zeroshot[task]
            correct, batch_sizes = eval_prompts(
                model,
                prompts,
                targets,
                layer_name=layer_name if vector is not None else None,
                vector=vector,
                batch_size=batch_size,
            )
            collected.extend(correct)
            all_batch_sizes.extend(batch_sizes)
        return collected

    outcomes["clean"] = eval_over_tasks(lambda task: None)
    outcomes["full_sig"] = eval_over_tasks(
        lambda task: sum(
            z_results[task][layer_idx, head_idx] for layer_idx, head_idx in sig_heads
        )
    )
    outcomes["mean_fv"] = eval_over_tasks(lambda task: mean_fv)

    curves: dict[str, dict[str, float]] = {}
    for layer_idx, head_idx in heads:
        others = mean_fv - mean_z[layer_idx, head_idx]
        curve: dict[str, float] = {}
        for c in c_grid:
            key = f"{layer_idx}:{head_idx}:{c}"
            outcomes[key] = eval_over_tasks(
                lambda task, _c=c, _l=layer_idx, _h=head_idx, _others=others: (
                    _c * z_results[task][_l, _h] + _others
                )
            )
            curve[str(c)] = float(np.mean(outcomes[key]))
        curves[f"{layer_idx}:{head_idx}"] = curve

    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = out_dir / OUTCOMES_FILE
    np.savez_compressed(
        outcomes_path,
        **{key: np.asarray(values, dtype=np.uint8) for key, values in outcomes.items()},
    )

    baselines = {
        "clean_acc": float(np.mean(outcomes["clean"])),
        "full_significant_acc": float(np.mean(outcomes["full_sig"])),
        "mean_fv_acc": float(np.mean(outcomes["mean_fv"])),
    }
    # macro (mean of per-task means) alongside the pooled micro baselines;
    # identical when per-task counts are equal, distinct for uneven families
    task_slices, start = {}, 0
    for task in task_order:
        task_slices[task] = slice(start, start + counts[task])
        start += counts[task]
    baselines_macro = {
        key: float(
            np.mean([np.mean(outcomes[src][task_slices[task]]) for task in task_order])
        )
        for key, src in (
            ("clean_acc", "clean"),
            ("full_significant_acc", "full_sig"),
            ("mean_fv_acc", "mean_fv"),
        )
    }
    return make_manifest(
        kind="head_scan",
        schema_version=HEAD_SCAN_SCHEMA_VERSION,
        paths=paths,
        config={
            "scan": to_dict(cfg.scan),
            "sites": to_dict(cfg.sites),
            "intervention_mode": 0,
            "batch_size": batch_size,
            # versioned scoring contract — part of artifact identity
            "scoring": to_dict(cfg.scoring),
        },
        inputs=inputs_refs,
        payload={
            "impl": IMPL,
            "curves": curves,
            "baselines": baselines,
            "baselines_macro": baselines_macro,
            "c_grid": c_grid,
            "scanned_heads": [list(head) for head in heads],
            "head_limit_applied": cfg.scan.head_limit,
            "n_sig_heads": len(sig_heads),
            "n_eval_examples_per_head_per_c": n_eval_total,
            "counts_per_task": counts,
            "task_order": task_order,
            "effective_batch_sizes": sorted(set(all_batch_sizes)),
            "gpu_memory": gpu_peak_memory(),
            "outcomes": {
                "file": OUTCOMES_FILE,
                "content_sha256": outcomes_content_sha(outcomes),
            },
        },
    )
