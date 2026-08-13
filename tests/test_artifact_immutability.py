"""Selector/heads artifacts under output layout v2: single-slot heads.json per
main node, immutable (refuse-on-different-identity), variants as sibling
main-* node dirs."""

import json

import pytest

from subspaces.artifacts import (
    ArtifactError,
    make_manifest,
    manifest_ref,
    write_json_atomic,
)
from subspaces.step1 import tree
from subspaces.step1.pipeline import compose_heads


def _main_manifest(fake_paths, scan_path, scan, params):
    return make_manifest(
        kind="main_heads",
        schema_version=1,
        paths=fake_paths,
        inputs={"head_scan": manifest_ref(scan_path, fake_paths, scan)},
        payload={
            "impl": {"module": "subspaces.step1.selectors", "algorithm_version": 1},
            "selector_name": "unified",
            "selector_version": 1,
            "params": params,
            "main_heads": [[15, 2], [15, 1], [13, 6]],
            "minor_heads": [],
            "decisions": [],
            "verdict": "localized",
        },
    )


@pytest.fixture()
def substep_artifacts(fake_paths, tmp_path):
    significant = make_manifest(
        kind="significant_heads",
        schema_version=1,
        paths=fake_paths,
        payload={
            "impl": {"module": "subspaces.step1.significant", "algorithm_version": 1},
            "heads": [[15, 2, 0.51], [15, 1, 0.44], [13, 6, 0.31], [10, 0, 0.21]],
            "model_dims": {"n_layers": 32, "n_heads": 32},
        },
    )
    significant_path = tmp_path / "significant_heads.json"
    write_json_atomic(significant_path, significant)

    scan = make_manifest(
        kind="head_scan",
        schema_version=1,
        paths=fake_paths,
        inputs={
            "significant_heads": manifest_ref(significant_path, fake_paths, significant)
        },
        payload={"curves": {}, "c_grid": list(range(21))},
    )
    scan_path = tmp_path / "head_scan.json"
    write_json_atomic(scan_path, scan)

    main = _main_manifest(fake_paths, scan_path, scan, {"beta": 0.25, "floor": 0.12})
    main_path = tmp_path / "main_heads.json"
    write_json_atomic(main_path, main)
    return significant_path, scan_path, scan, main_path


def test_compose_heads_writes_single_slot_heads_json(
    fake_paths, tmp_path, substep_artifacts
):
    """Layout v2 replaced the hashed heads filename with a SINGLE SLOT
    ``heads.json`` per main node: the identity is fully determined by the
    significant + main pair fixed in that node, and the slot is guarded by
    refuse-on-different-identity (next tests)."""
    significant_path, _, _, main_path = substep_artifacts
    out_dir = tmp_path / "heads"
    out_dir.mkdir()
    compose_heads(
        significant_path=significant_path,
        main_path=main_path,
        paths=fake_paths,
        out_dir=out_dir,
    )
    files = list(out_dir.iterdir())
    assert [f.name for f in files] == ["heads.json"]
    manifest = json.loads(files[0].read_text(encoding="utf-8"))
    assert manifest["main_heads"] == [[15, 2], [15, 1], [13, 6]]
    assert manifest["significant_heads"][0] == [15, 2]
    assert "scan_outcomes" not in json.dumps(manifest)  # per-example data excluded


def test_compose_heads_reuses_identical_and_refuses_tampered(
    fake_paths, tmp_path, substep_artifacts
):
    significant_path, _, _, main_path = substep_artifacts
    out_dir = tmp_path / "heads"
    out_dir.mkdir()
    compose_heads(
        significant_path=significant_path,
        main_path=main_path,
        paths=fake_paths,
        out_dir=out_dir,
    )
    compose_heads(  # identical inputs -> reuse (still exactly one file)
        significant_path=significant_path,
        main_path=main_path,
        paths=fake_paths,
        out_dir=out_dir,
    )
    files = list(out_dir.iterdir())
    assert len(files) == 1

    tampered = json.loads(files[0].read_text(encoding="utf-8"))
    tampered["main_heads"] = [[0, 0]]
    files[0].write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ArtifactError, match="different semantic identity"):
        compose_heads(
            significant_path=significant_path,
            main_path=main_path,
            paths=fake_paths,
            out_dir=out_dir,
        )


def test_different_selector_params_get_different_sibling_nodes(
    fake_paths, tmp_path, substep_artifacts
):
    """Layout v2 replaced per-params hashed FILES in one dir with sibling
    main-* node DIRS (``tree.main_dirname`` over params + scan fingerprint +
    selector), each holding its own single-slot heads.json; a different
    selection composed into an occupied slot refuses."""
    significant_path, scan_path, scan, main_path = substep_artifacts
    other = _main_manifest(fake_paths, scan_path, scan, {"beta": 0.33, "floor": 0.12})
    other_path = tmp_path / "main_heads_b033.json"
    write_json_atomic(other_path, other)

    node_a = tmp_path / tree.main_dirname(
        "unified", 1, {"beta": 0.25, "floor": 0.12}, scan
    )
    node_b = tmp_path / tree.main_dirname(
        "unified", 1, {"beta": 0.33, "floor": 0.12}, scan
    )
    assert node_a != node_b  # params fork the node dir, not the filename

    compose_heads(
        significant_path=significant_path,
        main_path=main_path,
        paths=fake_paths,
        out_dir=node_a,
    )
    compose_heads(
        significant_path=significant_path,
        main_path=other_path,
        paths=fake_paths,
        out_dir=node_b,
    )
    # immutable side-by-side variants, one single-slot heads.json each
    assert (node_a / "heads.json").is_file()
    assert (node_b / "heads.json").is_file()

    # the single slot is guarded: the other selection cannot land in node_a
    with pytest.raises(ArtifactError, match="different semantic identity"):
        compose_heads(
            significant_path=significant_path,
            main_path=other_path,
            paths=fake_paths,
            out_dir=node_a,
        )


def test_atomic_write_leaves_no_temp_files(tmp_path):
    target = tmp_path / "artifact.json"
    write_json_atomic(target, {"a": 1})
    write_json_atomic(target, {"a": 1})  # idempotent rewrite is fine
    leftovers = [p for p in tmp_path.iterdir() if p.name != "artifact.json"]
    assert leftovers == []
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
