"""Dependency-chain enforcement: stale artifacts refuse; compositions verify
that main-head scans descend from the supplied selected artifact."""

import pytest

from subspaces.artifacts import (
    ArtifactError,
    make_manifest,
    manifest_ref,
    write_json_atomic,
)
from subspaces.config import load_step1_config
from subspaces.step1 import selected as selected_mod
from subspaces.step1 import tree
from subspaces.step1.pipeline import (
    compose_heads,
    ensure_run,
    stage_matrix,
    stage_selected,
)


def _cfg(fake_paths):
    return load_step1_config(fake_paths.root / "configs" / "step1_test.yaml")


def _selected_manifest(fake_paths, matrix_ref, ref_path, heads):
    cfg = _cfg(fake_paths)
    return make_manifest(
        kind="selected_heads",
        schema_version=selected_mod.SELECTED_SCHEMA_VERSION,
        paths=fake_paths,
        config={"selected": selected_mod.selected_identity_config(cfg.selected)},
        inputs={"matrix_ref": manifest_ref(ref_path, fake_paths, matrix_ref)},
        payload={
            "impl": selected_mod.impl_for(cfg.selected),
            "heads": heads,
            "model_dims": {"n_layers": 32, "n_heads": 32},
        },
    )


def test_stale_selected_artifact_refused(fake_paths):
    """A selected_heads.json recorded against a DIFFERENT matrix identity
    must refuse instead of silently flowing downstream."""
    cfg = _cfg(fake_paths)
    journal_dir, _, _ = ensure_run(cfg, fake_paths)
    matrix_ref, matrix_node = stage_matrix(cfg, fake_paths, journal_dir)

    stale = _selected_manifest(
        fake_paths,
        matrix_ref,
        matrix_node / "matrix_ref.json",
        heads=[[15, 2, 0.5]],
    )
    stale["inputs"]["matrix_ref"]["semantic_fingerprint"] = "0" * 64  # wrong matrix
    selected_node = tree.selected_node_dir(matrix_node, cfg.selected)
    write_json_atomic(selected_node / "selected_heads.json", stale)

    with pytest.raises(ArtifactError, match="refusing to reuse"):
        stage_selected(cfg, fake_paths, matrix_node, matrix_ref)


def test_matching_selected_artifact_reused_without_recompute(fake_paths, monkeypatch):
    cfg = _cfg(fake_paths)
    journal_dir, _, _ = ensure_run(cfg, fake_paths)
    matrix_ref, matrix_node = stage_matrix(cfg, fake_paths, journal_dir)

    valid = _selected_manifest(
        fake_paths,
        matrix_ref,
        matrix_node / "matrix_ref.json",
        heads=[[15, 2, 0.5], [13, 6, 0.3]],
    )
    selected_node = tree.selected_node_dir(matrix_node, cfg.selected)
    write_json_atomic(selected_node / "selected_heads.json", valid)

    def _must_not_run(*_a, **_k):
        raise AssertionError("extract_selected must not run on valid reuse")

    monkeypatch.setattr(selected_mod, "extract_selected", _must_not_run)
    reused, reused_node = stage_selected(cfg, fake_paths, matrix_node, matrix_ref)
    assert reused["heads"] == [[15, 2, 0.5], [13, 6, 0.3]]
    assert reused_node == selected_node


@pytest.fixture()
def lineage_chain(fake_paths, tmp_path):
    """selected -> head_scan -> main_heads, with correct reference chain."""
    selected = make_manifest(
        kind="selected_heads",
        schema_version=1,
        paths=fake_paths,
        payload={
            "impl": selected_mod.IMPL,
            "heads": [[15, 2, 0.51], [15, 1, 0.44], [13, 6, 0.31]],
            "model_dims": {"n_layers": 32, "n_heads": 32},
        },
    )
    selected_path = tmp_path / "selected_heads.json"
    write_json_atomic(selected_path, selected)

    scan = make_manifest(
        kind="head_scan",
        schema_version=1,
        paths=fake_paths,
        inputs={"selected_heads": manifest_ref(selected_path, fake_paths, selected)},
        payload={"curves": {}, "c_grid": list(range(21))},
    )
    scan_path = tmp_path / "head_scan.json"
    write_json_atomic(scan_path, scan)

    main = make_manifest(
        kind="main_heads",
        schema_version=1,
        paths=fake_paths,
        inputs={"head_scan": manifest_ref(scan_path, fake_paths, scan)},
        payload={
            "impl": {"module": "subspaces.step1.selectors", "algorithm_version": 1},
            "selector_name": "unified",
            "selector_version": 1,
            "params": {"beta": 0.25},
            "main_heads": [[15, 2], [15, 1], [13, 6]],
            "minor_heads": [],
            "decisions": [],
            "verdict": "localized",
        },
    )
    main_path = tmp_path / "main_heads.json"
    write_json_atomic(main_path, main)
    return selected_path, main_path


def test_compose_heads_accepts_valid_lineage(fake_paths, tmp_path, lineage_chain):
    selected_path, main_path = lineage_chain
    out_dir = tmp_path / "heads"
    out_dir.mkdir()
    manifest = compose_heads(
        selected_path=selected_path,
        main_path=main_path,
        paths=fake_paths,
        out_dir=out_dir,
    )
    assert manifest["main_heads"] == [[15, 2], [15, 1], [13, 6]]


def test_compose_heads_refuses_foreign_selected(fake_paths, tmp_path, lineage_chain):
    """The scan descends from selected A; composing with selected B
    (different content) must refuse."""
    _, main_path = lineage_chain
    other = make_manifest(
        kind="selected_heads",
        schema_version=1,
        paths=fake_paths,
        payload={
            "impl": selected_mod.IMPL,
            "heads": [[0, 0, 0.9]],
            "model_dims": {"n_layers": 32, "n_heads": 32},
        },
    )
    other_path = tmp_path / "other_selected.json"
    write_json_atomic(other_path, other)

    with pytest.raises(ArtifactError, match="lineage mismatch"):
        compose_heads(
            selected_path=other_path,
            main_path=main_path,
            paths=fake_paths,
            out_dir=tmp_path / "heads2",
        )


def test_compose_heads_refuses_unlineaged_main(fake_paths, tmp_path, lineage_chain):
    selected_path, _ = lineage_chain
    orphan = make_manifest(
        kind="main_heads",
        schema_version=1,
        paths=fake_paths,
        payload={
            "selector_name": "unified",
            "selector_version": 1,
            "params": {},
            "main_heads": [[15, 2]],
            "minor_heads": [],
        },
    )
    orphan_path = tmp_path / "orphan_main.json"
    write_json_atomic(orphan_path, orphan)

    with pytest.raises(ArtifactError, match="no head_scan input"):
        compose_heads(
            selected_path=selected_path,
            main_path=orphan_path,
            paths=fake_paths,
            out_dir=tmp_path / "heads3",
        )


def test_split_family_must_match_dataset_dir(fake_paths):
    from subspaces.config import ConfigError
    from subspaces.step1.pipeline import load_split

    cfg = _cfg(fake_paths)
    cfg.task.dataset_dir = "number_mul"
    with pytest.raises(ConfigError, match="split family"):
        load_split(cfg, fake_paths)
