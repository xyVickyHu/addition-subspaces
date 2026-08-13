"""Derived-checkpoint selection (``matrix.checkpoint_select: mean_last_k``):
config contract, derivation math, contiguity/adoption refusals, locator
invariance of the derived identity, and the CLI --matrix override semantics."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from subspaces.artifacts import ArtifactError, identity_of, semantic_fingerprint
from subspaces.config import (
    CheckpointSelectConfig,
    ConfigError,
    MatrixConfig,
    load_step1_config,
)
from subspaces.step1.matrix import (
    resolve_final_coefficient_checkpoint,
    resolve_reuse_matrix,
)
from subspaces.step1.pipeline import _MATRIX_IDENTITY_KEYS

MATRIX = {"mode": "reuse", "reuse_path": "matrices/toy"}


def _write_cfg(tmp_path: Path, matrix: dict) -> Path:
    path = tmp_path / "cfg.yaml"
    payload = {"protocol": "legacy_reproduction", "matrix": matrix}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _toy_cfg(fake_paths):
    return load_step1_config(fake_paths.root / "configs" / "step1_test.yaml")


def _build_contiguous_matrix(root: Path, epochs=(0, 1, 2, 3)) -> Path:
    """Matrix dir whose trailing checkpoints are contiguous (real tensors)."""
    import torch

    matrix_dir = root / "matrices" / "contig"
    checkpoints = matrix_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    for epoch in epochs:
        matrix = torch.zeros(4, 4)
        matrix[1, 1] = 0.5 + 0.1 * epoch
        matrix[2, 3] = 0.8
        matrix[0, 0] = 0.001 * (epoch + 1)
        torch.save(matrix, checkpoints / f"matrix_epoch{epoch}.pth")
    return matrix_dir


def _select_cfg(fake_paths, matrix_dir: Path, k: int = 3):
    cfg = _toy_cfg(fake_paths)
    cfg.matrix.reuse_path = str(matrix_dir)
    cfg.matrix.checkpoint_select = CheckpointSelectConfig(rule="mean_last_k", k=k)
    return cfg


def _derived_checkpoint_path(fake_paths, manifest) -> Path:
    selected = manifest["selected_checkpoint"]
    return fake_paths.resolve(manifest["matrix_dir"]) / "checkpoints" / selected["file"]


# -- config contract ----------------------------------------------------------


def test_checkpoint_select_xor_checkpoint_epoch(tmp_path):
    matrix = {
        **MATRIX,
        "checkpoint_epoch": 5,
        "checkpoint_select": {"rule": "mean_last_k", "k": 2},
    }
    with pytest.raises(ConfigError, match="mutually exclusive"):
        load_step1_config(_write_cfg(tmp_path, matrix))


def test_checkpoint_select_unknown_rule_refused(tmp_path):
    matrix = {**MATRIX, "checkpoint_select": {"rule": "mean_first_k", "k": 2}}
    with pytest.raises(ConfigError, match="must be mean_last_k"):
        load_step1_config(_write_cfg(tmp_path, matrix))


def test_checkpoint_select_k_zero_refused(tmp_path):
    matrix = {**MATRIX, "checkpoint_select": {"rule": "mean_last_k", "k": 0}}
    with pytest.raises(ConfigError, match="k must be >= 1"):
        load_step1_config(_write_cfg(tmp_path, matrix))


def test_checkpoint_select_float_k_refused(tmp_path):
    matrix = {**MATRIX, "checkpoint_select": {"rule": "mean_last_k", "k": 2.5}}
    with pytest.raises(ConfigError, match="expected int"):
        load_step1_config(_write_cfg(tmp_path, matrix))


# -- derivation ---------------------------------------------------------------


def test_mean_last_k_matches_numpy_recomputation(fake_paths):
    import numpy as np
    import torch

    matrix_dir = _build_contiguous_matrix(fake_paths.root)
    cfg = _select_cfg(fake_paths, matrix_dir, k=3)
    manifest = resolve_reuse_matrix(cfg, fake_paths)
    selected = manifest["selected_checkpoint"]

    derived = torch.load(_derived_checkpoint_path(fake_paths, manifest))
    stacked = np.stack(
        [
            torch.load(matrix_dir / "checkpoints" / f"matrix_epoch{epoch}.pth").numpy()
            for epoch in (1, 2, 3)  # trailing 3 of epochs 0..3
        ]
    )
    np.testing.assert_allclose(derived.numpy(), stacked.mean(axis=0), rtol=1e-6)

    assert selected["selection_rule"] == "mean_last_k"
    assert selected["k"] == 3
    assert selected["epoch"] is None
    assert len(selected["content_digest"]) == 64
    assert len(selected["file_sha256"]) == 64
    assert "sha256" not in selected  # the byte sha must not carry identity
    sources = selected["source_checkpoints"]
    assert [source["epoch"] for source in sources] == [1, 2, 3]
    for source in sources:
        assert source["file"] == f"matrix_epoch{source['epoch']}.pth"
        assert len(source["sha256"]) == 64


def test_raw_coef_resolves_final_source_not_trailing_mean(fake_paths):
    import torch

    matrix_dir = _build_contiguous_matrix(fake_paths.root)
    cfg = _select_cfg(fake_paths, matrix_dir, k=3)
    manifest = resolve_reuse_matrix(cfg, fake_paths)

    source, source_path = resolve_final_coefficient_checkpoint(
        cfg, manifest, fake_paths
    )
    assert source["selection_rule"] == "final_checkpoint"
    assert source["epoch"] == 3
    assert source_path == matrix_dir / "checkpoints" / "matrix_epoch3.pth"
    final = torch.load(source_path)
    trailing_mean = torch.load(_derived_checkpoint_path(fake_paths, manifest))
    assert float(final[1, 1]) == pytest.approx(0.8)
    assert float(trailing_mean[1, 1]) == pytest.approx(0.7)
    assert not torch.equal(final, trailing_mean)


def test_raw_coef_refuses_source_extended_after_mean_resolution(fake_paths):
    """A checkpoint appended to the SOURCE dir after the trailing-mean
    matrix_ref was resolved makes the directory's final checkpoint disagree
    with the recorded window — mixing provenance; must refuse."""
    import torch

    matrix_dir = _build_contiguous_matrix(fake_paths.root)
    cfg = _select_cfg(fake_paths, matrix_dir, k=3)
    manifest = resolve_reuse_matrix(cfg, fake_paths)

    torch.save(torch.ones(4, 4), matrix_dir / "checkpoints" / "matrix_epoch4.pth")
    with pytest.raises(ArtifactError, match="mean_last_k provenance"):
        resolve_final_coefficient_checkpoint(cfg, manifest, fake_paths)


def test_raw_coef_refuses_mean_ref_without_source_checkpoints(fake_paths):
    matrix_dir = _build_contiguous_matrix(fake_paths.root)
    cfg = _select_cfg(fake_paths, matrix_dir, k=3)
    manifest = resolve_reuse_matrix(cfg, fake_paths)

    del manifest["selected_checkpoint"]["source_checkpoints"]
    with pytest.raises(ArtifactError, match="lacks source_checkpoints"):
        resolve_final_coefficient_checkpoint(cfg, manifest, fake_paths)


def test_raw_coef_refuses_source_extended_under_latest_rule(fake_paths):
    """`latest` records the final checkpoint itself; a later-extended source
    directory must refuse rather than silently rebind raw_coef to the new
    final epoch."""
    import torch

    matrix_dir = _build_contiguous_matrix(fake_paths.root)
    cfg = _toy_cfg(fake_paths)
    cfg.matrix.reuse_path = str(matrix_dir)
    manifest = resolve_reuse_matrix(cfg, fake_paths)
    assert manifest["selected_checkpoint"]["selection_rule"] == "latest"

    torch.save(torch.ones(4, 4), matrix_dir / "checkpoints" / "matrix_epoch4.pth")
    with pytest.raises(ArtifactError, match="latest provenance"):
        resolve_final_coefficient_checkpoint(cfg, manifest, fake_paths)


def test_raw_coef_uses_final_even_when_selection_pinned_earlier(fake_paths):
    """A pinned NON-final selection epoch is legitimate (selection substrate);
    raw_coef still resolves the directory's final checkpoint, unguarded by
    design (the input reference records epoch+sha for audit)."""
    matrix_dir = _build_contiguous_matrix(fake_paths.root)
    cfg = _toy_cfg(fake_paths)
    cfg.matrix.reuse_path = str(matrix_dir)
    cfg.matrix.checkpoint_epoch = 2
    manifest = resolve_reuse_matrix(cfg, fake_paths)
    assert manifest["selected_checkpoint"]["epoch"] == 2

    source, source_path = resolve_final_coefficient_checkpoint(
        cfg, manifest, fake_paths
    )
    assert source["epoch"] == 3
    assert source_path.name == "matrix_epoch3.pth"


def test_non_consecutive_trailing_epochs_refused(fake_paths):
    """The toy fixture's epochs (0, 2, 10) are gapped: averaging the trailing
    2 would span epochs 2..10 — refuse like the train-path contiguity rule."""
    cfg = _toy_cfg(fake_paths)
    cfg.matrix.checkpoint_select = CheckpointSelectConfig(rule="mean_last_k", k=2)
    with pytest.raises(ArtifactError, match="non-consecutive"):
        resolve_reuse_matrix(cfg, fake_paths)


def test_k_beyond_available_checkpoints_refused(fake_paths):
    cfg = _toy_cfg(fake_paths)
    cfg.matrix.checkpoint_select = CheckpointSelectConfig(rule="mean_last_k", k=4)
    with pytest.raises(ArtifactError, match="only 3 checkpoint"):
        resolve_reuse_matrix(cfg, fake_paths)


# -- adoption of an existing derived file --------------------------------------


def test_adoption_refuses_content_divergent_derived_file(fake_paths):
    import torch

    matrix_dir = _build_contiguous_matrix(fake_paths.root)
    cfg = _select_cfg(fake_paths, matrix_dir)
    manifest = resolve_reuse_matrix(cfg, fake_paths)
    out_path = _derived_checkpoint_path(fake_paths, manifest)

    torch.save(torch.ones(4, 4), out_path)  # different but valid tensor
    with pytest.raises(ArtifactError, match="content digest"):
        resolve_reuse_matrix(cfg, fake_paths)


def test_adoption_refuses_unreadable_derived_file(fake_paths):
    matrix_dir = _build_contiguous_matrix(fake_paths.root)
    cfg = _select_cfg(fake_paths, matrix_dir)
    manifest = resolve_reuse_matrix(cfg, fake_paths)
    out_path = _derived_checkpoint_path(fake_paths, manifest)

    out_path.write_bytes(b"garbage, not a checkpoint")
    with pytest.raises(ArtifactError, match="unreadable"):
        resolve_reuse_matrix(cfg, fake_paths)


def test_adoption_refuses_non_tensor_derived_file(fake_paths):
    import torch

    matrix_dir = _build_contiguous_matrix(fake_paths.root)
    cfg = _select_cfg(fake_paths, matrix_dir)
    manifest = resolve_reuse_matrix(cfg, fake_paths)
    out_path = _derived_checkpoint_path(fake_paths, manifest)

    torch.save({"not": 1}, out_path)
    with pytest.raises(ArtifactError, match="did not contain a torch.Tensor"):
        resolve_reuse_matrix(cfg, fake_paths)


# -- identity: locator invariance + regeneration stability ---------------------


def test_derived_identity_survives_source_move(fake_paths, tmp_path):
    """Moving the SOURCE matrix dir (locator change only) must not change the
    derived node's identity or the manifest's semantic fingerprint."""
    matrix_dir = _build_contiguous_matrix(fake_paths.root)
    cfg = _select_cfg(fake_paths, matrix_dir)
    before = resolve_reuse_matrix(cfg, fake_paths)

    moved = tmp_path / "elsewhere" / "contig"  # same basename: same node name
    moved.parent.mkdir(parents=True)
    shutil.move(str(matrix_dir), str(moved))
    cfg.matrix.reuse_path = str(moved)
    after = resolve_reuse_matrix(cfg, fake_paths)

    assert identity_of(before, _MATRIX_IDENTITY_KEYS) == identity_of(
        after, _MATRIX_IDENTITY_KEYS
    )
    assert semantic_fingerprint(before) == semantic_fingerprint(after)
    assert (
        before["selected_checkpoint"]["source_matrix_dir"]
        != after["selected_checkpoint"]["source_matrix_dir"]
    )


def test_derived_regeneration_keeps_identity(fake_paths):
    """Deleting the materialized mean and re-deriving must reproduce the same
    content digest and identity (the file byte sha is free to differ)."""
    matrix_dir = _build_contiguous_matrix(fake_paths.root)
    cfg = _select_cfg(fake_paths, matrix_dir)
    first = resolve_reuse_matrix(cfg, fake_paths)

    _derived_checkpoint_path(fake_paths, first).unlink()
    second = resolve_reuse_matrix(cfg, fake_paths)

    assert (
        first["selected_checkpoint"]["content_digest"]
        == second["selected_checkpoint"]["content_digest"]
    )
    assert identity_of(first, _MATRIX_IDENTITY_KEYS) == identity_of(
        second, _MATRIX_IDENTITY_KEYS
    )


# -- CLI --matrix override ------------------------------------------------------


def test_matrix_override_preserves_checkpoint_select(fake_paths):
    from subspaces.runners.step1 import _matrix_override

    base = MatrixConfig(
        mode="reuse",
        reuse_path="matrices/original",
        checkpoint_select=CheckpointSelectConfig(rule="mean_last_k", k=2),
        node_name="mynode",
    )
    effective = _matrix_override("matrices/toy", base, fake_paths)
    assert effective.mode == "reuse"
    assert effective.reuse_path == "matrices/toy"
    assert effective.train is None
    assert effective.checkpoint_select is not None
    assert effective.checkpoint_select.k == 2  # NOT silently discarded
    assert effective.node_name == "mynode"


def test_matrix_override_refuses_derived_matrix_ref(fake_paths):
    from subspaces.artifacts import write_json_atomic
    from subspaces.runners.step1 import _matrix_override

    matrix_dir = _build_contiguous_matrix(fake_paths.root)
    cfg = _select_cfg(fake_paths, matrix_dir)
    manifest = resolve_reuse_matrix(cfg, fake_paths)
    ref_path = fake_paths.runs_dir / "derived-node" / "matrix_ref.json"
    write_json_atomic(ref_path, manifest)

    with pytest.raises(ArtifactError) as excinfo:
        _matrix_override(str(ref_path), cfg.matrix, fake_paths)
    message = str(excinfo.value)
    assert "source_matrix_dir" in message
    assert "checkpoint_select" in message
    assert "mean_last_k" in message
    # actionable: the recorded source dir and k are printed
    assert json.dumps(manifest["selected_checkpoint"]["k"]) in message
