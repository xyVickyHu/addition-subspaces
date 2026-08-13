import pytest

from subspaces.artifacts import (
    ArtifactError,
    check_file_ref,
    dataset_fingerprint,
    file_ref,
    make_manifest,
    read_manifest,
    semantic_fingerprint,
    write_json_atomic,
)


def test_write_read_round_trip(fake_paths, tmp_path):
    manifest = make_manifest(
        kind="matrix_ref", schema_version=1, paths=fake_paths, payload={"x": 1}
    )
    out = tmp_path / "m.json"
    write_json_atomic(out, manifest)
    loaded = read_manifest(out, expect_kind="matrix_ref", max_schema_version=1)
    assert loaded == manifest


def test_read_refuses_wrong_kind_and_version(fake_paths, tmp_path):
    out = tmp_path / "m.json"
    write_json_atomic(
        out,
        make_manifest(kind="matrix_ref", schema_version=1, paths=fake_paths),
    )
    with pytest.raises(ArtifactError, match="expected kind"):
        read_manifest(out, expect_kind="heads", max_schema_version=1)
    write_json_atomic(
        out, make_manifest(kind="matrix_ref", schema_version=9, paths=fake_paths)
    )
    with pytest.raises(ArtifactError, match="schema_version"):
        read_manifest(out, expect_kind="matrix_ref", max_schema_version=1)


def test_semantic_fingerprint_ignores_volatile_fields():
    base = {
        "schema_version": 1,
        "kind": "heads",
        "created_at": "2026-07-20T00:00:00+00:00",
        "code": {"git_commit": "abc", "dirty": True},
        "inputs": {"scan": {"path": "log/runs/a/head_scan.json", "sha256": "aa"}},
        "main_heads": [[15, 2]],
    }
    moved = dict(base)
    moved["created_at"] = "2027-01-01T00:00:00+00:00"
    moved["code"] = {"git_commit": "def", "dirty": False}
    moved["inputs"] = {"scan": {"path": "elsewhere/head_scan.json", "sha256": "aa"}}
    assert semantic_fingerprint(base) == semantic_fingerprint(moved)

    changed = dict(base)
    changed["main_heads"] = [[15, 2], [13, 6]]
    assert semantic_fingerprint(changed) != semantic_fingerprint(base)

    changed_input = dict(base)
    changed_input["inputs"] = {
        "scan": {"path": "log/runs/a/head_scan.json", "sha256": "bb"}
    }
    assert semantic_fingerprint(changed_input) != semantic_fingerprint(base)


def test_semantic_fingerprint_ignores_gpu_memory():
    """Peak-VRAM observability is device/history-dependent: two identical
    regenerations may differ, so it must not fork fingerprints or lineage."""
    base = {
        "schema_version": 1,
        "kind": "head_scan",
        "curves": {"15:2": {"0": 0.03}},
        "gpu_memory": None,
    }
    other = dict(base)
    other["gpu_memory"] = {
        "device_name": "NVIDIA A100-PCIE-40GB",
        "max_allocated_gib": 17.2,
        "max_reserved_gib": 18.0,
        "total_gib": 40.0,
    }
    assert semantic_fingerprint(base) == semantic_fingerprint(other)


def test_file_ref_validation_refuses_mismatch(fake_paths):
    target = fake_paths.root / "configs" / "step1_test.yaml"
    ref = file_ref(target, fake_paths)
    assert not ref["path"].startswith("/")
    assert check_file_ref(ref, fake_paths) == target

    target.write_text(target.read_text() + "\n# changed\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="content mismatch"):
        check_file_ref(ref, fake_paths)


def test_dataset_fingerprint_tracks_content(fake_paths):
    task_dir = fake_paths.task_dir("number_add")
    before = dataset_fingerprint(task_dir)
    assert before == dataset_fingerprint(task_dir)  # deterministic

    victim = task_dir / "number-add1.json"
    victim.write_text(victim.read_text().replace('"2"', '"999"', 1), encoding="utf-8")
    assert dataset_fingerprint(task_dir) != before
