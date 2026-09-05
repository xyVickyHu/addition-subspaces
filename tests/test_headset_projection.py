"""Projected headset arm (``proj_meanab``): unit math, identity discipline,
lineage refusals, and the e2e path through ``stage_evaluate_headset``.

The projection convention under test is the paper's §4 causal projection:
per head, ``mu + P P^T (v - mu)`` with the PCA re-derived from EXACTLY the
step2_subspace artifact's fit set and verified against its recorded
cumulative-EVR curve.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from conftest import build_fake_repo  # noqa: F401 — fixtures via conftest

from subspaces.artifacts import ArtifactError
from subspaces.config import load_step1_config
from subspaces.paths import ProjectPaths
from subspaces.step1 import pipeline
from subspaces.step1.headset_eval import (
    PROJECTION_ARM,
    identity_config,
    projected_headset_vectors,
    projection_identity,
)

D_MODEL = 8


def _synthetic_z(task_ids, seed_offset=0):
    z = {}
    for index, task in enumerate(task_ids):
        generator = torch.Generator().manual_seed(1000 + seed_offset + index)
        z[task] = torch.randn(4, 4, D_MODEL, generator=generator)
    return z


def _subspace_for(train_z, heldout_z, heads, threshold=0.95):
    """A step2_subspace-shaped manifest computed with the real step-2 math."""
    from subspaces.step2.pca import head_pca_summary

    merged = {**train_z, **heldout_z}
    task_order = sorted(merged)
    per_head = {}
    for layer_idx, head_idx in heads:
        x = (
            torch.stack(
                [merged[task][layer_idx, head_idx] for task in task_order], dim=0
            )
            .to(torch.float32)
            .numpy()
        )
        per_head[f"{layer_idx}:{head_idx}"] = head_pca_summary(
            x, threshold=threshold, n_pcs=6
        )
    return {
        "config": {
            "threshold": threshold,
            "heads": sorted([list(head) for head in heads]),
        },
        "task_order": task_order,
        "per_head": per_head,
        "inputs": {"z_caches": []},
    }


def test_full_rank_projection_reproduces_fit_vectors():
    """k = n_components makes the projection the identity on fit-set rows —
    and the held-out tasks ARE fit rows (merged fit set, paper convention)."""
    train_z = _synthetic_z([f"t{i}" for i in range(5)])
    heldout_z = _synthetic_z(["e1", "e2"], seed_offset=50)
    heads = [(1, 1), (2, 3)]
    # threshold 1.0 -> k = the centered rank (n_tasks - 1): the fit rows'
    # deviations lie exactly in that span, so projection is identity on them
    subspace = _subspace_for(train_z, heldout_z, heads, threshold=1.0)

    vectors, per_head_k = projected_headset_vectors(
        subspace, train_z, heldout_z, heads, ["e1", "e2"]
    )
    for task in ("e1", "e2"):
        plain = sum(
            heldout_z[task][layer_idx, head_idx].to(torch.float32)
            for layer_idx, head_idx in heads
        )
        torch.testing.assert_close(vectors[task], plain, atol=1e-4, rtol=1e-4)
    assert set(per_head_k) == {"1:1", "2:3"}


def test_k1_projection_matches_manual_sklearn():
    from sklearn.decomposition import PCA

    train_z = _synthetic_z([f"t{i}" for i in range(6)])
    heldout_z = _synthetic_z(["e1"], seed_offset=80)
    head = (0, 2)
    subspace = _subspace_for(train_z, heldout_z, [head])
    # force k=1 CONSISTENTLY: the manifest threshold must imply it
    subspace["per_head"]["0:2"]["pcs_to_threshold"] = 1
    subspace["config"]["threshold"] = subspace["per_head"]["0:2"]["pca_cumvar"][0] / 2

    vectors, per_head_k = projected_headset_vectors(
        subspace, train_z, heldout_z, [head], ["e1"]
    )
    merged = {**train_z, **heldout_z}
    x = np.stack(
        [merged[task][head[0], head[1]].float().numpy() for task in sorted(merged)]
    )
    pca = PCA().fit(x)
    basis = pca.components_[:1].T.astype(np.float32)
    mean = pca.mean_.astype(np.float32)
    v = heldout_z["e1"][head[0], head[1]].float().numpy()
    expected = mean + basis @ (basis.T @ (v - mean))
    np.testing.assert_allclose(vectors["e1"].numpy(), expected, atol=1e-5)
    assert per_head_k == {"0:2": 1}


def test_projection_refusals():
    train_z = _synthetic_z([f"t{i}" for i in range(4)])
    heldout_z = _synthetic_z(["e1"], seed_offset=30)
    heads = [(1, 1)]
    subspace = _subspace_for(train_z, heldout_z, heads)

    # evaluated head with no PCA entry
    with pytest.raises(ArtifactError, match="no PCA entry"):
        projected_headset_vectors(subspace, train_z, heldout_z, [(3, 3)], ["e1"])

    # fit-set mismatch: extra task in the caches
    extra = dict(train_z)
    extra["t99"] = torch.randn(
        4, 4, D_MODEL, generator=torch.Generator().manual_seed(7)
    )
    with pytest.raises(ArtifactError, match="different fit set"):
        projected_headset_vectors(subspace, extra, heldout_z, heads, ["e1"])

    # fit-set mismatch: artifact expects a task the caches lack
    smaller = {task: z for task, z in train_z.items() if task != "t0"}
    with pytest.raises(ArtifactError, match="different fit set"):
        projected_headset_vectors(subspace, smaller, heldout_z, heads, ["e1"])

    # overlapping train/heldout tasks
    with pytest.raises(ArtifactError, match="share tasks"):
        projected_headset_vectors(
            subspace, {**train_z, "e1": heldout_z["e1"]}, heldout_z, heads, ["e1"]
        )

    # drifted cumulative-EVR curve (stale/foreign artifact)
    drifted = _subspace_for(train_z, heldout_z, heads)
    drifted["per_head"]["1:1"]["pca_cumvar"] = [
        value * 0.9 for value in drifted["per_head"]["1:1"]["pca_cumvar"]
    ]
    with pytest.raises(ArtifactError, match="does not reproduce"):
        projected_headset_vectors(drifted, train_z, heldout_z, heads, ["e1"])

    # out-of-range component count
    broken = _subspace_for(train_z, heldout_z, heads)
    broken["per_head"]["1:1"]["pcs_to_threshold"] = 0
    with pytest.raises(ArtifactError, match="out of range"):
        projected_headset_vectors(broken, train_z, heldout_z, heads, ["e1"])

    # empty selection must refuse, never degrade to a NaN vector
    with pytest.raises(ArtifactError, match="non-empty evaluated head set"):
        projected_headset_vectors(subspace, train_z, heldout_z, [], ["e1"])

    # payload k inconsistent with the recorded threshold+curve (tamper check)
    mislabeled = _subspace_for(train_z, heldout_z, heads)
    entry = mislabeled["per_head"]["1:1"]
    implied = entry["pcs_to_threshold"]
    entry["pcs_to_threshold"] = (
        implied + 1 if implied < entry["n_components"] else implied - 1
    )
    with pytest.raises(ArtifactError, match="threshold .* implies"):
        projected_headset_vectors(mislabeled, train_z, heldout_z, heads, ["e1"])


def test_identity_config_unchanged_without_projection(fake_repo):
    cfg = load_step1_config(fake_repo / "configs" / "step1_test.yaml")
    main_heads = {"selector_name": "unified", "selector_version": 1}
    base = identity_config(cfg, main_heads, 20)
    assert "projection" not in base
    assert base == identity_config(cfg, main_heads, 20, projection=None)

    block = projection_identity({"config": {"threshold": 0.95}})
    with_projection = identity_config(cfg, main_heads, 20, projection=block)
    assert with_projection["projection"] == {
        "name": "pca_recentered_meanab",
        "version": 1,
        "k_source": "pcs_to_threshold",
        "threshold": 0.95,
    }
    without = {
        key: value for key, value in with_projection.items() if key != "projection"
    }
    assert without == base


def _subspace_artifact_for_run(paths, cfg, summary, heads, *, cache_keys):
    """Run real step 2 over the run's z caches; returns the artifact path."""
    from subspaces.step2.api import Step2Config
    from subspaces.step2.pca import run_subspace_analysis

    main_node = paths.resolve(summary["latest_selection"]["main_node"])
    eval_manifest = json.loads(
        next(main_node.glob("eval-*.json")).read_text(encoding="utf-8")
    )
    z_dirs = [
        paths.root
        / "log"
        / "cache"
        / "z"
        / eval_manifest["inputs"][key]["content_fingerprint"]
        for key in cache_keys
    ]
    _, artifact_path, _ = run_subspace_analysis(
        Step2Config(),
        heads,
        paths,
        head_set="main",
        heads_artifact=main_node / "main_heads.json",
        z_cache_dirs=z_dirs,
    )
    return main_node, artifact_path


def test_projected_arm_e2e_identity_fork_and_baseline_equality(fake_repo, gpu_stubs):
    paths = ProjectPaths.from_root(fake_repo)
    cfg = load_step1_config(fake_repo / "configs" / "step1_test.yaml")
    summary = pipeline.run(cfg, paths)
    main_node = paths.resolve(summary["latest_selection"]["main_node"])
    baseline_path = next(main_node.glob("eval-*.json"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    main_manifest = json.loads(
        (main_node / "main_heads.json").read_text(encoding="utf-8")
    )
    heads = [tuple(head) for head in main_manifest["main_heads"]]

    _, subspace_path = _subspace_artifact_for_run(
        paths, cfg, summary, heads, cache_keys=("z_cache_train", "z_cache_heldout")
    )

    journal_dir, split, resolved = pipeline.ensure_run(cfg, paths)
    matrix_ref, matrix_node = pipeline.stage_matrix(cfg, paths, journal_dir)
    selected, selected_node = pipeline.stage_selected(
        cfg, paths, matrix_node, matrix_ref
    )
    samples = pipeline.stage_selection_samples(cfg, paths, split)
    scan_node = paths.resolve(summary["nodes"]["scan"])
    manifest = pipeline.stage_evaluate_headset(
        cfg,
        paths,
        split,
        matrix_node,
        selected_node,
        scan_node,
        selected,
        matrix_ref,
        main_manifest,
        main_node,
        samples["activation"],
        samples["scan"],
        resolved,
        pipeline.make_model_loader(cfg, resolved),
        subspace_path=subspace_path,
    )

    # new arm present, with per-head component counts recorded
    assert PROJECTION_ARM in manifest["metrics"]
    assert set(manifest["projection"]["per_head_k"]) == {
        f"{layer_idx}:{head_idx}" for layer_idx, head_idx in heads
    }
    assert manifest["config"]["projection"]["name"] == "pca_recentered_meanab"
    assert manifest["inputs"]["pca_subspace"]["semantic_fingerprint"]

    # immutable sibling: the plain v4 eval is untouched, identities fork
    evals = sorted(main_node.glob("eval-*.json"))
    assert len(evals) == 2
    assert baseline_path in evals
    untouched = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert untouched == baseline

    # set-independent arms must reproduce the plain eval bit-exactly
    for arm in ("clean", "full_selected", "raw_coef", "main_meanab"):
        assert manifest["metrics"][arm] == baseline["metrics"][arm]

    # reuse: an identical re-request returns the SAME manifest, no rerun
    again = pipeline.stage_evaluate_headset(
        cfg,
        paths,
        split,
        matrix_node,
        selected_node,
        scan_node,
        selected,
        matrix_ref,
        main_manifest,
        main_node,
        samples["activation"],
        samples["scan"],
        resolved,
        pipeline.make_model_loader(cfg, resolved),
        subspace_path=subspace_path,
    )
    assert again["metrics"] == manifest["metrics"]


def test_projected_arm_refuses_foreign_fit_set_and_uncovered_heads(
    fake_repo, gpu_stubs
):
    paths = ProjectPaths.from_root(fake_repo)
    cfg = load_step1_config(fake_repo / "configs" / "step1_test.yaml")
    summary = pipeline.run(cfg, paths)
    main_node = paths.resolve(summary["latest_selection"]["main_node"])
    main_manifest = json.loads(
        (main_node / "main_heads.json").read_text(encoding="utf-8")
    )
    heads = [tuple(head) for head in main_manifest["main_heads"]]

    def _stage(subspace_path):
        journal_dir, split, resolved = pipeline.ensure_run(cfg, paths)
        matrix_ref, matrix_node = pipeline.stage_matrix(cfg, paths, journal_dir)
        selected, selected_node = pipeline.stage_selected(
            cfg, paths, matrix_node, matrix_ref
        )
        samples = pipeline.stage_selection_samples(cfg, paths, split)
        return pipeline.stage_evaluate_headset(
            cfg,
            paths,
            split,
            matrix_node,
            selected_node,
            paths.resolve(summary["nodes"]["scan"]),
            selected,
            matrix_ref,
            main_manifest,
            main_node,
            samples["activation"],
            samples["scan"],
            resolved,
            pipeline.make_model_loader(cfg, resolved),
            subspace_path=subspace_path,
        )

    # fitted on the train cache only -> foreign fit set
    _, train_only = _subspace_artifact_for_run(
        paths, cfg, summary, heads, cache_keys=("z_cache_train",)
    )
    with pytest.raises(ArtifactError, match="foreign fit set"):
        _stage(train_only)

    # covers only a strict subset of the selected heads
    _, partial = _subspace_artifact_for_run(
        paths, cfg, summary, heads[:1], cache_keys=("z_cache_train", "z_cache_heldout")
    )
    if len(heads) > 1:
        with pytest.raises(ArtifactError, match="covers no PCA"):
            _stage(partial)
