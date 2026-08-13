"""Significant-head substep: Lane-A equivalence on the canonical matrix
(exactly 33 heads, ordered set equal to the shipped repro scan) plus contract
behavior on the toy matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subspaces.artifacts import ArtifactError, write_json_atomic
from subspaces.config import load_step1_config
from subspaces.paths import ProjectPaths, find_repo_root
from subspaces.step1 import tree
from subspaces.step1.matrix import resolve_reuse_matrix
from subspaces.step1.pipeline import ensure_run, stage_matrix, stage_significant
from subspaces.step1.significant import select_significant

REPO = find_repo_root(Path(__file__))
FIXTURES = Path(__file__).parent / "fixtures"


def _toy_cfg(fake_paths):
    return load_step1_config(fake_paths.root / "configs" / "step1_test.yaml")


def test_canonical_matrix_exact_33_equivalence():
    """The paper matrix under the committed corrected config reproduces the
    shipped repro scan's significant set EXACTLY: 33 heads, same order,
    same elbow threshold."""
    paths = ProjectPaths.from_root(REPO)
    cfg = load_step1_config(REPO / "configs" / "step1_number_add_llama3.yaml")
    matrix_ref = resolve_reuse_matrix(cfg, paths)
    manifest = select_significant(matrix_ref, cfg, paths)

    fixture = json.loads(
        (FIXTURES / "selector_scan_repro_v1.json").read_text(encoding="utf-8")
    )
    assert manifest["count"] == 33
    got = [[layer_idx, head_idx] for layer_idx, head_idx, _ in manifest["heads"]]
    assert got == fixture["significant_heads"]
    assert manifest["threshold"] == pytest.approx(fixture["threshold"])
    assert manifest["threshold_method"] == fixture["threshold_method"] == "elbow"
    assert manifest["model_dims"] == {"n_layers": 32, "n_heads": 32}
    trio = [[15, 2], [15, 1], [13, 6]]
    assert all(head in got for head in trio)


def test_toy_matrix_elbow_matches_legacy(fake_paths):
    """Elbow on the toy matrix: the knee value IS the threshold and selection
    is strictly-above (legacy semantics) — only the 0.9 spike survives."""
    cfg = _toy_cfg(fake_paths)
    matrix_ref = resolve_reuse_matrix(cfg, fake_paths)
    manifest = select_significant(matrix_ref, cfg, fake_paths)
    heads = [(layer_idx, head_idx) for layer_idx, head_idx, _ in manifest["heads"]]
    assert heads == [(1, 1)]
    assert manifest["threshold"] == pytest.approx(0.8, abs=1e-6)
    assert manifest["heads"][0][2] == pytest.approx(0.9)
    assert manifest["model_dims"] == {"n_layers": 4, "n_heads": 4}


def test_fixed_and_fraction_methods(fake_paths):
    cfg = _toy_cfg(fake_paths)
    matrix_ref = resolve_reuse_matrix(cfg, fake_paths)

    cfg.significant.method = "fixed"
    cfg.significant.fixed_threshold = 0.5
    manifest = select_significant(matrix_ref, cfg, fake_paths)
    # ordering by |coef| descending across both spike heads
    assert [(lh[0], lh[1]) for lh in manifest["heads"]] == [(1, 1), (2, 3)]
    assert manifest["threshold"] == pytest.approx(0.5)

    cfg.significant.method = "fraction"
    cfg.significant.fixed_threshold = None
    cfg.significant.fraction = 0.99
    manifest = select_significant(matrix_ref, cfg, fake_paths)
    # legacy boundary semantics: the threshold equals the smallest coef whose
    # cumulative mass reaches the fraction, and selection is strictly above it
    assert manifest["threshold"] == pytest.approx(0.8, abs=1e-6)
    assert [(lh[0], lh[1]) for lh in manifest["heads"]] == [(1, 1)]


def test_toy_matrix_largest_gap(fake_paths):
    """largest_gap on the toy matrix: the biggest drop is 0.8 -> 0.011
    (epoch 10), so both spike heads are kept and the realized threshold is
    the drop's lower edge (strictly-above selection then includes the
    upper edge)."""
    cfg = _toy_cfg(fake_paths)
    cfg.significant.method = "largest_gap"
    matrix_ref = resolve_reuse_matrix(cfg, fake_paths)
    manifest = select_significant(matrix_ref, cfg, fake_paths)
    heads = [(layer_idx, head_idx) for layer_idx, head_idx, _ in manifest["heads"]]
    assert heads == [(1, 1), (2, 3)]
    assert manifest["threshold"] == pytest.approx(0.011, abs=1e-6)
    assert manifest["threshold_method"] == "largest_gap"
    gap = manifest["gap"]
    assert gap["upper"] == pytest.approx(0.8)
    assert gap["lower"] == pytest.approx(0.011, abs=1e-6)
    assert gap["n_selected"] == manifest["count"] == 2
    # runner-up is the 0.9 -> 0.8 drop at the very top of the curve
    assert gap["runner_up"]["drop"] == pytest.approx(0.1)
    assert gap["runner_up"]["rank_of_upper"] == 1


def test_largest_gap_threshold_edge_cases():
    """Pure-rule contract: tie handling at both edges, the sparsity
    boundary as a candidate cut, |coef| semantics, tie between drops, and
    refusal on degenerate curves."""
    import torch

    from subspaces.step1.significant import largest_gap_threshold

    # ties at the UPPER edge are all kept
    threshold, info = largest_gap_threshold(torch.tensor([[1.0, 1.0], [0.2, 0.1]]))
    assert threshold == pytest.approx(0.2)
    assert info["n_selected"] == 2

    # ties at the LOWER edge are all excluded (strictly-above semantics)
    threshold, info = largest_gap_threshold(torch.tensor([[1.0, 0.5], [0.5, 0.4]]))
    assert threshold == pytest.approx(0.5)
    assert info["n_selected"] == 1

    # the sparsity boundary (last nonzero -> 0) is a legitimate cut
    threshold, info = largest_gap_threshold(torch.tensor([[0.9, 0.8], [0.7, 0.0]]))
    assert threshold == 0.0
    assert info["n_selected"] == 3

    # exact tie between two drops (binary-exact values so float32 keeps the
    # tie exact): the FIRST (highest cut, fewest heads) wins
    threshold, info = largest_gap_threshold(torch.tensor([[0.75, 0.5], [0.25, 0.25]]))
    assert threshold == pytest.approx(0.5)
    assert info["n_selected"] == 1
    assert info["runner_up"]["drop"] == pytest.approx(info["drop"])

    # the rule runs on |coef|
    threshold, info = largest_gap_threshold(torch.tensor([[-0.9, 0.1], [0.0, 0.0]]))
    assert threshold == pytest.approx(0.1)
    assert info["n_selected"] == 1

    # degenerate curves refuse instead of guessing
    with pytest.raises(ArtifactError, match="no drop"):
        largest_gap_threshold(torch.full((2, 2), 0.5))
    with pytest.raises(ArtifactError, match="fewer than two"):
        largest_gap_threshold(torch.tensor([[1.0]]))
    with pytest.raises(ArtifactError, match="NaN"):
        largest_gap_threshold(torch.tensor([[1.0, float("nan")], [0.5, 0.1]]))


def test_canonical_matrix_largest_gap_anchor():
    """largest_gap on the canonical paper matrix cuts at the rank-28 -> 29
    drop (0.669827 -> 0.441219, drop 0.228607): the selected set is exactly
    the top 28 of the 33 elbow heads and contains the paper trio. The
    runner-up drop (rank 33 -> 34, 0.241950 -> 0.015800, drop 0.226151) IS
    the canonical elbow/fixed-0.2 cut and trails by only ~0.0025 — a
    near-tie that is a property of this checkpoint (cf. the elbow/fixed-0.2
    coincidence noted in the step-1 methods record)."""
    paths = ProjectPaths.from_root(REPO)
    cfg = load_step1_config(REPO / "configs" / "step1_number_add_llama3_siggap.yaml")
    matrix_ref = resolve_reuse_matrix(cfg, paths)
    manifest = select_significant(matrix_ref, cfg, paths)

    fixture = json.loads(
        (FIXTURES / "selector_scan_repro_v1.json").read_text(encoding="utf-8")
    )
    assert manifest["count"] == 28
    got = {(layer_idx, head_idx) for layer_idx, head_idx, _ in manifest["heads"]}
    expected = {
        (layer_idx, head_idx)
        for layer_idx, head_idx in fixture["significant_heads"][:28]
    }
    assert got == expected
    assert {(15, 2), (15, 1), (13, 6)} <= got
    gap = manifest["gap"]
    assert manifest["threshold"] == pytest.approx(gap["lower"])
    assert gap["rank_of_upper"] == 28
    assert gap["upper"] == pytest.approx(0.669827, abs=1e-5)
    assert gap["lower"] == pytest.approx(0.441219, abs=1e-5)
    assert gap["runner_up"]["rank_of_upper"] == 33
    assert gap["runner_up"]["upper"] == pytest.approx(0.241950, abs=1e-5)
    assert gap["runner_up"]["lower"] == pytest.approx(0.015800, abs=1e-5)
    assert gap["drop"] - gap["runner_up"]["drop"] == pytest.approx(0.0025, abs=1e-3)


def test_stage_significant_method_swap_forks_sibling_node_never_clobbers(fake_paths):
    """Under the lineage tree, changing significant.method no longer refuses
    in-place: method variants are SIBLING sig-* nodes under the matrix node.
    The surviving guarantee is that a swap can never clobber the existing
    artifact — it lands in its own node dir, and if a foreign artifact
    occupies that node's slot, stage_significant REFUSES instead of
    overwriting."""
    cfg = _toy_cfg(fake_paths)
    journal_dir, _, _ = ensure_run(cfg, fake_paths)
    matrix_ref, matrix_node = stage_matrix(cfg, fake_paths, journal_dir)
    elbow_manifest, elbow_node = stage_significant(
        cfg, fake_paths, matrix_node, matrix_ref
    )

    cfg.significant.method = "largest_gap"
    gap_node_expected = tree.sig_node_dir(matrix_node, cfg.significant)
    assert gap_node_expected != elbow_node  # sibling dir, not the same slot

    # bypass the tree: squat the elbow artifact on the largest_gap node slot
    write_json_atomic(gap_node_expected / "significant_heads.json", elbow_manifest)
    with pytest.raises(ArtifactError, match="refusing to reuse"):
        stage_significant(cfg, fake_paths, matrix_node, matrix_ref)

    # with the slot cleared, the swap produces its own sibling node and the
    # original elbow artifact is untouched
    (gap_node_expected / "significant_heads.json").unlink()
    gap_manifest, gap_node = stage_significant(cfg, fake_paths, matrix_node, matrix_ref)
    assert gap_node == gap_node_expected
    assert gap_manifest["threshold_method"] == "largest_gap"
    assert (elbow_node / "significant_heads.json").is_file()
    assert (gap_node / "significant_heads.json").is_file()


def test_significant_identity_config_is_method_scoped():
    """The identity slice carries method, method_version, and ONLY the
    registered parameters of that method — a future parameter FIELD on
    SignificantConfig (always None for existing methods) can never fork
    their identities, because absent-from-schema fields never enter."""
    from subspaces.config import SignificantConfig
    from subspaces.step1.significant import significant_identity_config

    assert significant_identity_config(SignificantConfig()) == {
        "method": "elbow",
        "method_version": 1,
    }
    assert significant_identity_config(
        SignificantConfig(method="fraction", fraction=0.9)
    ) == {"method": "fraction", "method_version": 1, "fraction": 0.9}
    # simulated hypothetical extra field: the elbow slice equals the slice of
    # a config that (like any future dataclass revision) carries unrelated
    # branch fields — they are not in elbow's schema, so they cannot appear
    assert significant_identity_config(
        SignificantConfig(method="elbow", fixed_threshold=None, fraction=None)
    ) == significant_identity_config(SignificantConfig())


def test_sig_dirname_params_derive_from_registry():
    import re

    from subspaces.config import SignificantConfig
    from subspaces.step1.tree import sig_dirname

    assert sig_dirname(SignificantConfig()) == "sig-elbow-v1"  # no param hash
    name = sig_dirname(SignificantConfig(method="fraction", fraction=0.9))
    assert re.fullmatch(r"sig-fraction-v1-[0-9a-f]{6}", name)


def test_largest_gap_refuses_non_finite_values():
    """Two +inf coefficients make every drop NaN — must refuse cleanly
    (corrupted checkpoint), not crash with an IndexError."""
    import torch

    from subspaces.step1.significant import largest_gap_threshold

    with pytest.raises(ArtifactError, match="non-finite"):
        largest_gap_threshold(torch.tensor([[float("inf"), float("inf")], [0.5, 0.1]]))


def test_largest_gap_runner_up_none_without_second_cut():
    """A single-nonzero matrix has exactly one candidate cut; the runner-up
    slot records None (a zero non-drop is not a candidate cut)."""
    import torch

    from subspaces.step1.significant import largest_gap_threshold

    threshold, info = largest_gap_threshold(torch.tensor([[0.9, 0.0], [0.0, 0.0]]))
    assert threshold == 0.0
    assert info["n_selected"] == 1
    assert info["runner_up"] is None


def test_registered_but_undispatched_method_refuses(fake_paths, monkeypatch):
    """A registry/dispatch mismatch must raise the explicit ArtifactError,
    never an UnboundLocalError on 'threshold'."""
    from subspaces.step1 import significant as significant_mod

    monkeypatch.setitem(significant_mod.METHOD_PARAM_SCHEMAS, ("foo", 1), frozenset())
    cfg = _toy_cfg(fake_paths)
    cfg.significant.method = "foo"  # bypasses load-time validation on purpose
    matrix_ref = resolve_reuse_matrix(cfg, fake_paths)
    with pytest.raises(ArtifactError, match="registered but not dispatched"):
        significant_mod.select_significant(matrix_ref, cfg, fake_paths)


def test_select_refuses_checkpoint_tamper(fake_paths):
    cfg = _toy_cfg(fake_paths)
    matrix_ref = resolve_reuse_matrix(cfg, fake_paths)
    checkpoint = (
        fake_paths.root / "matrices" / "toy" / "checkpoints" / "matrix_epoch10.pth"
    )
    checkpoint.write_bytes(b"tampered-after-ref")
    with pytest.raises(ArtifactError, match="checkpoint content mismatch"):
        select_significant(matrix_ref, cfg, fake_paths)


def test_stage_significant_end_to_end_and_reuse(fake_paths, monkeypatch):
    cfg = _toy_cfg(fake_paths)
    journal_dir, _, _ = ensure_run(cfg, fake_paths)
    matrix_ref, matrix_node = stage_matrix(cfg, fake_paths, journal_dir)
    first, sig_node = stage_significant(cfg, fake_paths, matrix_node, matrix_ref)
    assert sig_node == tree.sig_node_dir(matrix_node, cfg.significant)
    assert (sig_node / "significant_heads.json").is_file()

    import subspaces.step1.significant as significant_mod

    def _must_not_run(*_a, **_k):
        raise AssertionError("must reuse, not recompute")

    monkeypatch.setattr(significant_mod, "select_significant", _must_not_run)
    # also monkeypatch the pipeline's imported alias
    import subspaces.step1.pipeline as pipeline_mod

    monkeypatch.setattr(
        pipeline_mod.significant_mod, "select_significant", _must_not_run
    )
    second, sig_node_again = stage_significant(cfg, fake_paths, matrix_node, matrix_ref)
    assert second == first
    assert sig_node_again == sig_node
