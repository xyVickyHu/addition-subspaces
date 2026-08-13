"""Step-2 task-generic PCA: math exactness, artifact/reuse contract, CLI.

The statistic under test is the legacy stage-3 rule: sklearn ``PCA()`` over a
head's per-task prompt-mean vectors (float32, mean-centered), cumulative
``explained_variance_ratio_``, and ``pcs_to_threshold`` = the 1-indexed first
count whose cumulative ratio reaches the threshold (>= semantics).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from conftest import tree_snapshot

from subspaces.artifacts import (
    ArtifactError,
    make_manifest,
    semantic_fingerprint,
    write_json_atomic,
)
from subspaces.head_sets import HeadSpecError, load_head_set
from subspaces.paths import ProjectPaths
from subspaces.runners import step2 as step2_cli
from subspaces.step1.zcache import load_zcache_dir, store_zcache
from subspaces.step2.pca import (
    head_pca_summary,
    load_and_merge_zcaches,
    pcs_from_cumvar,
)

REAL_REPO = Path(__file__).resolve().parents[1]
# 30-task legacy-protocol llama3/number_add z cache (log/ is gitignored, so
# this anchor only runs where the cache exists).
ANCHOR_CACHE = REAL_REPO / "log" / "cache" / "z" / "96bcb3aad0650c71"


# -- math ---------------------------------------------------------------------


def _spectrum_matrix(variances: list[float], n_tasks: int, d_model: int):
    """Rows with an EXACT centered-PCA spectrum: X = U diag(s) V^T with U
    orthonormal and orthogonal to the all-ones vector, so column means are
    zero and sklearn's centering is the identity."""
    rng = np.random.default_rng(7)
    k = len(variances)
    base = np.column_stack([np.ones(n_tasks), rng.standard_normal((n_tasks, k))])
    q, _ = np.linalg.qr(base)
    u = q[:, 1 : k + 1]
    v, _ = np.linalg.qr(rng.standard_normal((d_model, k)))
    s = np.sqrt(np.asarray(variances))
    return (u * s) @ v.T


def test_pcs_from_cumvar_exact_semantics():
    """The >=-at-equality (searchsorted side='left') and clamp semantics,
    pinned on hand-built exact float64 arrays — an end-to-end PCA can never
    hit the tie exactly (one-ulp noise), so the rule is tested at the unit
    seam."""
    tie = np.array([0.5, 0.8, 0.95, 1.0])
    assert pcs_from_cumvar(tie, 0.95) == 3  # exact tie counts (>=)
    assert pcs_from_cumvar(tie, 0.9500000001) == 4  # just above the tie
    assert pcs_from_cumvar(tie, 0.5) == 1
    assert pcs_from_cumvar(tie, 1.0) == 4
    # clamp: a cumvar that never reaches the threshold (float-rounding
    # undershoot) stays in range instead of legacy's len+1
    assert pcs_from_cumvar(np.array([0.6, 0.99999994]), 1.0) == 2


def test_pcs_to_threshold_on_engineered_spectrum():
    # centered spectrum 0.5 / 0.3 / 0.15 / 0.05 -> cumvar .5 .8 .95 1.0;
    # thresholds bracket the 0.95 boundary by more than float32 ulp noise
    # (the boundary itself is one-ulp sensitive and not seed-stable).
    x = _spectrum_matrix([50.0, 30.0, 15.0, 5.0], n_tasks=8, d_model=16)
    result = head_pca_summary(x.astype(np.float32), threshold=0.9499, n_pcs=6)
    assert result["pcs_to_threshold"] == 3
    assert result["n_components"] == 8
    cumvar = result["pca_cumvar"]
    assert cumvar[:4] == pytest.approx([0.5, 0.8, 0.95, 1.0], abs=1e-5)
    # n_pcs beyond the component count clamps to the last component
    assert result["cumvar_at_n_pcs"] == pytest.approx(1.0, abs=1e-6)
    above = head_pca_summary(x.astype(np.float32), threshold=0.9501, n_pcs=2)
    assert above["pcs_to_threshold"] == 4
    assert above["cumvar_at_n_pcs"] == pytest.approx(0.8, abs=1e-5)


def test_head_pca_summary_matches_independent_svd():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((7, 12)).astype(np.float32)
    result = head_pca_summary(x, threshold=0.95, n_pcs=6)
    centered = x - x.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    ratios = singular**2 / np.sum(singular**2)
    expected = np.cumsum(ratios)[: result["n_components"]]
    assert result["pca_cumvar"] == pytest.approx(expected.tolist(), abs=1e-5)
    assert result["pcs_to_threshold"] == int(np.searchsorted(expected, 0.95) + 1)


def test_head_pca_summary_refuses_degenerate_inputs():
    with pytest.raises(ArtifactError, match="at least 2 task vectors"):
        head_pca_summary(np.ones((1, 8), dtype=np.float32), threshold=0.95, n_pcs=6)
    with pytest.raises(ArtifactError, match="non-finite"):
        head_pca_summary(np.ones((5, 8), dtype=np.float32), threshold=0.95, n_pcs=6)


# -- fixtures: fabricated caches + head artifacts -------------------------------


def _fake_z(task_ids, *, n_layers=4, n_heads=4, d_model=8):
    z_results = {}
    for task_id in task_ids:
        seed = int.from_bytes(hashlib.sha256(task_id.encode()).digest()[:4], "big")
        generator = torch.Generator().manual_seed(seed)
        z_results[task_id] = torch.randn(
            n_layers, n_heads, d_model, generator=generator
        )
    return z_results


def _identity(samples_fp: str, model_name: str = "test-model") -> dict:
    return {
        "model": {"name": model_name, "revision": "abc", "dtype": "float32"},
        "dataset_fingerprint": "d" * 64,
        "samples": samples_fp,
        "hook_site": "attn.hook_result[last_token]",
        "impl": {"module": "subspaces.step1.zcache", "algorithm_version": 1},
    }


@pytest.fixture()
def step2_repo(fake_repo):
    """fake_repo + train/heldout z caches + a composed heads artifact in a
    main-* node + a context YAML referencing both caches."""
    paths = ProjectPaths.from_root(fake_repo)
    train_tasks = ["task-a", "task-b", "task-c", "task-d"]
    heldout_tasks = ["task-e", "task-f"]
    train_fp = store_zcache(paths, _identity("s-train"), _fake_z(train_tasks))
    heldout_fp = store_zcache(paths, _identity("s-heldout"), _fake_z(heldout_tasks))

    node = fake_repo / "log" / "runs" / "cell__abc" / "main-largest_gap-v1-cafe0000"
    node.mkdir(parents=True)
    heads_manifest = make_manifest(
        kind="heads",
        schema_version=1,
        paths=paths,
        payload={
            "significant_heads": [[1, 1], [2, 3], [0, 0]],
            "main_heads": [[1, 1], [2, 3]],
            "minor_heads": [],
            "model_dims": {"n_layers": 4, "n_heads": 4},
        },
    )
    heads_path = node / "heads.json"
    write_json_atomic(heads_path, heads_manifest)

    context = {
        "schema_version": 1,
        "model": {"name": "test-model", "revision": "abc"},
        "task": {"family": "faketasks", "prompt_format": "arrow", "n_shot": 5},
        "activations": {
            "train": {"path": f"log/cache/z/{train_fp}"},
            "heldout": {"path": f"log/cache/z/{heldout_fp}"},
        },
        "heads_artifact": {"path": str(heads_path)},
    }
    context_path = fake_repo / "configs" / "step2_context.yaml"
    context_path.write_text(yaml.safe_dump(context), encoding="utf-8")
    return {
        "repo": fake_repo,
        "paths": paths,
        "node": node,
        "heads_path": heads_path,
        "context_path": context_path,
        "train_fp": train_fp,
        "heldout_fp": heldout_fp,
        "tasks": sorted(train_tasks + heldout_tasks),
    }


def _run(repo, *argv):
    return step2_cli.main(["--root", str(repo["repo"]), *argv])


# -- CLI end-to-end -------------------------------------------------------------


def test_cli_writes_subspace_artifact_with_correct_numbers(step2_repo, capsys):
    rc = _run(step2_repo, "--context", str(step2_repo["context_path"]))
    assert rc == 0
    artifacts = list(step2_repo["node"].glob("subspace-*.json"))
    assert len(artifacts) == 1
    manifest = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert manifest["kind"] == "step2_subspace"
    assert manifest["n_tasks"] == 6
    assert manifest["task_order"] == step2_repo["tasks"]
    assert set(manifest["pcs_to_threshold"]) == {"1:1", "2:3"}
    assert manifest["config"]["heads"] == [[1, 1], [2, 3]]
    assert manifest["config"]["head_set"] == "main"
    assert manifest["impl"]["method"] == {"name": "cumvar_threshold", "version": 1}
    fingerprints = {
        ref["content_fingerprint"] for ref in manifest["inputs"]["z_caches"]
    }
    assert fingerprints == {step2_repo["train_fp"], step2_repo["heldout_fp"]}

    # independent recompute from the stored tensors, one head
    z_train, _ = load_zcache_dir(
        step2_repo["paths"].z_cache_dir / step2_repo["train_fp"]
    )
    z_held, _ = load_zcache_dir(
        step2_repo["paths"].z_cache_dir / step2_repo["heldout_fp"]
    )
    merged = {**z_train, **z_held}
    x = np.stack(
        [merged[t][1, 1].to(torch.float32).numpy() for t in manifest["task_order"]]
    )
    centered = x - x.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    cumvar = np.cumsum(singular**2 / np.sum(singular**2))
    head = manifest["per_head"]["1:1"]
    assert head["pca_cumvar"] == pytest.approx(
        cumvar[: head["n_components"]].tolist(), abs=1e-5
    )
    assert manifest["pcs_to_threshold"]["1:1"] == int(np.searchsorted(cumvar, 0.95) + 1)
    # dtype pin: the impl computes on float32 rows; an identical sklearn
    # recompute on float32 must match EXACTLY (a float64 drift would not)
    from sklearn.decomposition import PCA

    exact = np.cumsum(PCA().fit(x.astype(np.float32)).explained_variance_ratio_)
    assert head["pca_cumvar"] == [float(v) for v in exact[: head["n_components"]]]


def test_cli_reuses_identical_run_and_refuses_nothing(step2_repo, capsys):
    assert _run(step2_repo, "--context", str(step2_repo["context_path"])) == 0
    before = tree_snapshot(step2_repo["repo"])
    first = list(step2_repo["node"].glob("subspace-*.json"))[0].read_bytes()
    assert _run(step2_repo, "--context", str(step2_repo["context_path"])) == 0
    assert "reused" in capsys.readouterr().out
    assert tree_snapshot(step2_repo["repo"]) == before
    assert list(step2_repo["node"].glob("subspace-*.json"))[0].read_bytes() == first


def test_cli_parameter_change_lands_in_sibling_artifact(step2_repo):
    assert _run(step2_repo, "--context", str(step2_repo["context_path"])) == 0
    assert (
        _run(
            step2_repo,
            "--context",
            str(step2_repo["context_path"]),
            "--threshold",
            "0.9",
        )
        == 0
    )
    artifacts = list(step2_repo["node"].glob("subspace-*.json"))
    assert len(artifacts) == 2
    by_threshold = {
        manifest["config"]["threshold"]: manifest
        for manifest in (json.loads(p.read_text(encoding="utf-8")) for p in artifacts)
    }
    assert set(by_threshold) == {0.95, 0.9}
    # the NUMBERS must follow the parameter, not just the label: recompute
    # the 0.9 statistic independently from the stored tensors
    manifest = by_threshold[0.9]
    cumvar = _independent_cumvar(step2_repo, manifest["task_order"], (1, 1))
    assert manifest["pcs_to_threshold"]["1:1"] == int(np.searchsorted(cumvar, 0.9) + 1)


def _independent_cumvar(step2_repo, task_order, head):
    z_train, _ = load_zcache_dir(
        step2_repo["paths"].z_cache_dir / step2_repo["train_fp"]
    )
    z_held, _ = load_zcache_dir(
        step2_repo["paths"].z_cache_dir / step2_repo["heldout_fp"]
    )
    merged = {**z_train, **z_held}
    layer_idx, head_idx = head
    x = np.stack(
        [merged[t][layer_idx, head_idx].to(torch.float32).numpy() for t in task_order]
    )
    centered = x - x.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return np.cumsum(singular**2 / np.sum(singular**2))


def test_explicit_heads_order_does_not_fork_identity(step2_repo, tmp_path, capsys):
    """The same head SET in a different order must reuse the same artifact
    (identity sorts heads; per-head PCAs are independent)."""
    out_dir = tmp_path / "ord"
    base = ["--context", str(step2_repo["context_path"]), "--out-dir", str(out_dir)]
    assert _run(step2_repo, "--heads", "2:3,1:1", *base) == 0
    assert _run(step2_repo, "--heads", "1:1,2:3", *base) == 0
    assert "reused" in capsys.readouterr().out
    assert len(list(out_dir.glob("subspace-*.json"))) == 1


def test_cli_z_cache_flags_override_context(step2_repo):
    """--z-cache (train only) beats the context's train+heldout pair."""
    rc = _run(
        step2_repo,
        "--context",
        str(step2_repo["context_path"]),
        "--z-cache",
        step2_repo["train_fp"],
    )
    assert rc == 0
    manifest = json.loads(
        list(step2_repo["node"].glob("subspace-*.json"))[0].read_text(encoding="utf-8")
    )
    assert manifest["n_tasks"] == 4
    assert [ref["content_fingerprint"] for ref in manifest["inputs"]["z_caches"]] == [
        step2_repo["train_fp"]
    ]


def test_main_heads_artifact_kind_accepted(step2_repo, capsys):
    """Selector nodes carry only main_heads.json — step 2 must accept it."""
    node = step2_repo["node"]
    main_manifest = make_manifest(
        kind="main_heads",
        schema_version=1,
        paths=step2_repo["paths"],
        payload={
            "selector_name": "largest_gap",
            "selector_version": 1,
            "params": {},
            "main_heads": [[2, 3]],
            "minor_heads": [],
        },
    )
    main_path = node / "main_heads.json"
    write_json_atomic(main_path, main_manifest)
    rc = _run(
        step2_repo,
        "--heads-artifact",
        str(main_path),
        "--head-set",
        "main",
        "--context",
        str(step2_repo["context_path"]),
    )
    assert rc == 0
    manifests = [
        json.loads(p.read_text(encoding="utf-8")) for p in node.glob("subspace-*.json")
    ]
    assert any(m["config"]["heads"] == [[2, 3]] for m in manifests)
    # significant set is not available from a main_heads artifact
    with pytest.raises(HeadSpecError, match="no 'significant' set"):
        load_head_set(main_path, which="significant")


def test_explicit_heads_require_out_dir(step2_repo, tmp_path, capsys):
    rc = _run(
        step2_repo,
        "--heads",
        "1:1",
        "--context",
        str(step2_repo["context_path"]),
    )
    assert rc == 2
    assert "--out-dir" in capsys.readouterr().err
    out_dir = tmp_path / "explicit"
    rc = _run(
        step2_repo,
        "--heads",
        "1:1",
        "--context",
        str(step2_repo["context_path"]),
        "--out-dir",
        str(out_dir),
    )
    assert rc == 0
    manifest = json.loads(
        list(out_dir.glob("subspace-*.json"))[0].read_text(encoding="utf-8")
    )
    assert manifest["config"]["head_set"] == "explicit"
    assert manifest["config"]["heads"] == [[1, 1]]


def test_frozen_flat_run_parent_refused(step2_repo, capsys):
    """A heads artifact inside a frozen pre-tree layout must not become the
    default output dir (select-main/compose rule)."""
    frozen = step2_repo["repo"] / "log" / "runs" / "old__run" / "main_heads"
    frozen.mkdir(parents=True)
    heads_manifest = json.loads(step2_repo["heads_path"].read_text(encoding="utf-8"))
    frozen_path = frozen / "heads.json"
    write_json_atomic(frozen_path, heads_manifest)
    before = tree_snapshot(step2_repo["repo"])
    rc = _run(
        step2_repo,
        "--heads-artifact",
        str(frozen_path),
        "--context",
        str(step2_repo["context_path"]),
    )
    assert rc == 2
    assert "frozen" in capsys.readouterr().err
    assert tree_snapshot(step2_repo["repo"]) == before


def test_causal_projection_plugin_refused_before_any_write(step2_repo, capsys):
    """The causal consumer of the subspace artifact is the Step-1 headset-eval
    projected arm (proj_meanab); the causal_projection plugin name stays a
    refusing placeholder. (periodic IS implemented since 2026-08-06 — its
    contract lives in tests/test_step2_periodic.py.)"""
    before = tree_snapshot(step2_repo["repo"])
    rc = _run(
        step2_repo,
        "--context",
        str(step2_repo["context_path"]),
        "--plugin",
        "causal_projection",
    )
    assert rc == 2
    assert "proj_meanab" in capsys.readouterr().err
    assert tree_snapshot(step2_repo["repo"]) == before


def test_missing_z_caches_refused_clearly(step2_repo, capsys):
    context = yaml.safe_load(step2_repo["context_path"].read_text(encoding="utf-8"))
    context.pop("activations")
    bare = step2_repo["repo"] / "configs" / "bare_context.yaml"
    bare.write_text(yaml.safe_dump(context), encoding="utf-8")
    before = tree_snapshot(step2_repo["repo"])
    rc = _run(step2_repo, "--context", str(bare))
    assert rc == 2
    assert "z cache" in capsys.readouterr().err
    assert tree_snapshot(step2_repo["repo"]) == before


def test_incompatible_and_overlapping_caches_refused(step2_repo):
    paths = step2_repo["paths"]
    other_model_fp = store_zcache(
        paths, _identity("s-other", model_name="other-model"), _fake_z(["task-x"])
    )
    train_dir = paths.z_cache_dir / step2_repo["train_fp"]
    with pytest.raises(ArtifactError, match="incompatible"):
        load_and_merge_zcaches([train_dir, paths.z_cache_dir / other_model_fp])
    overlap_fp = store_zcache(
        paths, _identity("s-overlap"), _fake_z(["task-a", "task-z"])
    )
    with pytest.raises(ArtifactError, match="more than one z cache"):
        load_and_merge_zcaches([train_dir, paths.z_cache_dir / overlap_fp])


def test_head_out_of_cache_dims_refused(step2_repo, tmp_path, capsys):
    rc = _run(
        step2_repo,
        "--heads",
        "9:9",
        "--context",
        str(step2_repo["context_path"]),
        "--out-dir",
        str(tmp_path / "oob"),
    )
    assert rc == 2
    assert "out of range" in capsys.readouterr().err


def test_frozen_step1_heads_layout_refused(step2_repo, capsys):
    """Real frozen flat runs keep composed heads at <run>/step1/heads/…;
    the default out-dir must refuse there too (guard bypass finding)."""
    frozen = step2_repo["repo"] / "log" / "runs" / "old__run" / "step1" / "heads"
    frozen.mkdir(parents=True)
    heads_manifest = json.loads(step2_repo["heads_path"].read_text(encoding="utf-8"))
    frozen_path = frozen / "unified-v1-abc.json"
    write_json_atomic(frozen_path, heads_manifest)
    before = tree_snapshot(step2_repo["repo"])
    rc = _run(
        step2_repo,
        "--heads-artifact",
        str(frozen_path),
        "--context",
        str(step2_repo["context_path"]),
    )
    assert rc == 2
    assert "frozen" in capsys.readouterr().err
    assert tree_snapshot(step2_repo["repo"]) == before


def _samples_manifest(prompt_format: str, n_shot: int) -> dict:
    return {
        "schema_version": 1,
        "kind": "samples",
        "config": {
            "sample_kind": "activation",
            "task": {
                "dataset_dir": "faketasks",
                "prompt_format": prompt_format,
                "n_shot": n_shot,
            },
        },
    }


def test_cross_protocol_caches_refused(step2_repo):
    """prompt_format/n_shot live only in the samples fingerprint; merging
    caches from different protocols must refuse when the samples manifests
    are resolvable (protocol-mixing finding: arrow+qa flipped the trio)."""
    paths = step2_repo["paths"]
    fps = {}
    for name, fmt in (("arrow", "arrow"), ("qa", "qa")):
        manifest = _samples_manifest(fmt, 5)
        samples_fp = semantic_fingerprint(manifest)
        write_json_atomic(
            paths.samples_cache_dir / samples_fp[:16] / "activation.json", manifest
        )
        fps[name] = store_zcache(
            paths, _identity(samples_fp), _fake_z([f"proto-{name}"])
        )
    with pytest.raises(ArtifactError, match="DIFFERENT prompt protocols"):
        load_and_merge_zcaches(
            [paths.z_cache_dir / fps["arrow"], paths.z_cache_dir / fps["qa"]], paths
        )


def test_context_model_mismatch_refused(step2_repo, capsys):
    context = yaml.safe_load(step2_repo["context_path"].read_text(encoding="utf-8"))
    context["model"]["name"] = "another-model"
    mislabeled = step2_repo["repo"] / "configs" / "mislabeled_context.yaml"
    mislabeled.write_text(yaml.safe_dump(context), encoding="utf-8")
    before = tree_snapshot(step2_repo["repo"])
    rc = _run(step2_repo, "--context", str(mislabeled))
    assert rc == 2
    assert "does not match" in capsys.readouterr().err
    assert tree_snapshot(step2_repo["repo"]) == before


def test_malformed_activations_shapes_refuse_cleanly(step2_repo, capsys):
    base = yaml.safe_load(step2_repo["context_path"].read_text(encoding="utf-8"))
    for bad in ([{"path": "log/cache/z/x"}], "log/cache/z/x", 42):
        context = dict(base)
        context["activations"] = bad
        path = step2_repo["repo"] / "configs" / "bad_activations.yaml"
        path.write_text(yaml.safe_dump(context), encoding="utf-8")
        rc = _run(step2_repo, "--context", str(path))
        assert rc == 2
        assert "activations" in capsys.readouterr().err


def test_corrupted_z_payload_refused_cleanly(step2_repo, capsys):
    cache_dir = step2_repo["paths"].z_cache_dir / step2_repo["train_fp"]
    (cache_dir / "z_results.pth").write_bytes(b"garbage not a pickle")
    rc = _run(step2_repo, "--context", str(step2_repo["context_path"]))
    assert rc == 2
    assert "unreadable or corrupted" in capsys.readouterr().err


def test_plots_written_and_regenerated_on_reuse(step2_repo):
    ctx = str(step2_repo["context_path"])
    assert _run(step2_repo, "--context", ctx) == 0
    assert not list(step2_repo["node"].glob("subspace-*-plots"))
    # a second (reused) run with --plots must still produce the plots
    assert _run(step2_repo, "--context", ctx, "--plots") == 0
    plot_dirs = list(step2_repo["node"].glob("subspace-*-plots"))
    assert len(plot_dirs) == 1
    assert list(plot_dirs[0].glob("pca_cumvar_L*.png"))


# -- context generation ---------------------------------------------------------


def _stored_samples_manifest(paths, prompt_format, n_shot, kind, seed=42):
    manifest = {
        "schema_version": 1,
        "kind": "samples",
        "config": {
            "sample_kind": kind,
            "spec": {"seed": seed},
            "task": {
                "dataset_dir": "faketasks",
                "prompt_format": prompt_format,
                "n_shot": n_shot,
            },
        },
    }
    samples_fp = semantic_fingerprint(manifest)
    write_json_atomic(
        paths.samples_cache_dir / samples_fp[:16] / f"{kind}.json", manifest
    )
    return samples_fp


def test_build_context_walks_lineage_and_filters_heldout_by_protocol(fake_repo):
    from subspaces.artifacts import manifest_ref
    from subspaces.step2.context_gen import build_context

    paths = ProjectPaths.from_root(fake_repo)
    train_fp = store_zcache(
        paths,
        _identity(_stored_samples_manifest(paths, "arrow", 5, "activation")),
        _fake_z(["task-a", "task-b"]),
    )
    arrow_held_fp = store_zcache(
        paths,
        _identity(_stored_samples_manifest(paths, "arrow", 5, "heldout_activation")),
        _fake_z(["task-c"]),
    )
    # same base identity + disjoint tasks but a DIFFERENT protocol: must be
    # filtered out of discovery, not offered as a candidate
    store_zcache(
        paths,
        _identity(_stored_samples_manifest(paths, "qa", 5, "heldout_activation")),
        _fake_z(["task-d"]),
    )

    node = fake_repo / "log" / "runs" / "cell__abc" / "main-largest_gap-v1-cafe0000"
    scan = make_manifest(
        kind="head_scan",
        schema_version=1,
        paths=paths,
        inputs={"z_cache": {"content_fingerprint": train_fp}},
        payload={"curves": {}},
    )
    scan_path = node / "head_scan.json"
    write_json_atomic(scan_path, scan)
    main = make_manifest(
        kind="main_heads",
        schema_version=1,
        paths=paths,
        inputs={"head_scan": manifest_ref(scan_path, paths, scan)},
        payload={
            "selector_name": "largest_gap",
            "selector_version": 1,
            "params": {},
            "main_heads": [[1, 1]],
            "minor_heads": [],
        },
    )
    main_path = node / "main_heads.json"
    write_json_atomic(main_path, main)

    result = build_context(paths, main_path, include_heldout=True)
    context = result["context"]
    assert result["warnings"] == []
    assert context["model"]["name"] == "test-model"
    assert context["task"] == {
        "family": "faketasks",
        "prompt_format": "arrow",
        "n_shot": 5,
    }
    assert context["activations"]["train"]["path"].endswith(train_fp)
    assert context["activations"]["heldout"]["path"].endswith(arrow_held_fp)
    assert context["heads_artifact"]["semantic_fingerprint"]

    # a second same-protocol disjoint cache makes discovery ambiguous: refuse
    store_zcache(
        paths,
        _identity(
            _stored_samples_manifest(paths, "arrow", 5, "heldout_activation", seed=7)
        ),
        _fake_z(["task-e"]),
    )
    with pytest.raises(ArtifactError, match="same-protocol candidates"):
        build_context(paths, main_path, include_heldout=True)
    # an explicit fingerprint override sidesteps discovery
    explicit = build_context(
        paths, main_path, include_heldout=True, heldout_fingerprint=arrow_held_fp
    )
    assert explicit["context"]["activations"]["heldout"]["path"].endswith(arrow_held_fp)


# -- real-repo anchor -----------------------------------------------------------


@pytest.mark.skipif(
    not ANCHOR_CACHE.is_dir(), reason="30-task legacy-protocol z cache not on disk"
)
def test_trio_pcs_at_95_anchor_on_legacy_protocol_cache():
    """Paper §4 anchor: the three main heads need 6 PCs for 95% cumulative
    variance over the 30 add-k tasks (legacy-protocol extraction)."""
    z_results, meta = load_zcache_dir(ANCHOR_CACHE)
    assert len(meta["tasks"]) == 30
    for layer_idx, head_idx in ((15, 2), (15, 1), (13, 6)):
        x = (
            torch.stack(
                [z_results[t][layer_idx, head_idx] for t in sorted(z_results)], dim=0
            )
            .to(torch.float32)
            .numpy()
        )
        result = head_pca_summary(x, threshold=0.95, n_pcs=6)
        assert result["pcs_to_threshold"] == 6
