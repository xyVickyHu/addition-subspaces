import pytest

from subspaces.artifacts import (
    ArtifactError,
    canonicalize,
    check_file_ref,
    content_hash,
    dataset_fingerprint,
    file_ref,
    identity_content_hash,
    identity_of,
    make_manifest,
    modernize,
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


# -- head-set terminology: legacy spelling <-> paper spelling ------------------


def _pkg() -> str:
    import subspaces.artifacts as artifacts

    return artifacts.__name__.split(".")[0]


def test_head_set_spelling_round_trip_keeps_identity():
    pkg = _pkg()
    legacy = {
        "kind": "significant_heads",
        "schema_version": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "config": {"significant": {"method": "largest_gap", "method_version": 1}},
        "inputs": {"matrix_ref": {"path": "log/runs/x", "semantic_fingerprint": "abc"}},
        "impl": {
            "module": f"{pkg}.step1.significant",
            "algorithm_version": 2,
            "method": {"name": "largest_gap", "version": 1},
        },
        "heads": [[15, 2, 0.9]],
        "count": 1,
        "gap": {"n_selected": 1},
        "verdict": {"n_selected": 3, "n_sig": 33},
        "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
        "metrics": {"full_sig": 0.7},
        "n_significant": 33,
        "n_sig_heads": 33,
        "decisions": {"10:6": {"headroom_vs_sumSig": 0.1}},
        "recpos_main_heads": {"path": "log/runs/y"},
    }
    modern = modernize(legacy)
    assert modern["kind"] == "selected_heads"
    assert set(modern["config"]) == {"selected"}
    assert modern["impl"]["module"] == f"{pkg}.step1.selected"
    assert modern["gap"] == {"n_selected": 1}  # a pre-existing key: untouched
    assert modern["verdict"] == {"n_selected": 3, "n_scanned": 33}
    assert modern["n_selected_set"] == 33
    assert modern["n_selected_heads"] == 33
    assert modern["baselines"]["full_selected_acc"] == 0.8
    assert modern["metrics"]["full_selected"] == 0.7
    assert modern["decisions"]["10:6"] == {"headroom_vs_sumSelected": 0.1}
    assert "significant_main_heads" in modern
    # exact inverse, idempotent on both sides, and identity-preserving
    assert canonicalize(modern) == legacy
    assert canonicalize(legacy) == legacy
    assert modernize(modern) == modern
    assert semantic_fingerprint(modern) == semantic_fingerprint(legacy)
    assert identity_of(modern) == identity_of(legacy)


def test_identity_content_hash_is_spelling_independent():
    pkg = _pkg()
    legacy = {
        "requested": {"significant": {"method": "elbow", "method_version": 1}},
        "algorithm_versions": {
            f"{pkg}.step1.significant": 2,
            f"{pkg}.step1.recovery": 3,
        },
    }
    modern = {
        "requested": {"selected": {"method": "elbow", "method_version": 1}},
        "algorithm_versions": {f"{pkg}.step1.selected": 2, f"{pkg}.step1.recovery": 3},
    }
    assert identity_content_hash(modern) == identity_content_hash(legacy)
    assert identity_content_hash(legacy) == content_hash(
        legacy
    )  # legacy is the hashed form


def test_outcomes_digest_is_spelling_independent():
    import numpy as np

    from subspaces.step1.recovery import outcomes_content_sha

    clean = np.array([1, 0, 1], dtype=np.uint8)
    full = np.array([0, 1, 1], dtype=np.uint8)
    legacy = outcomes_content_sha({"clean": clean, "full_sig": full})
    modern = outcomes_content_sha({"full_selected": full, "clean": clean})
    assert modern == legacy
    assert outcomes_content_sha({"clean": clean, "full_sig": clean}) != legacy


def test_respell_refuses_ambiguous_spelling():
    with pytest.raises(ArtifactError, match="ambiguous head-set spelling"):
        canonicalize({"selected": 1, "significant": 2})
    with pytest.raises(ArtifactError, match="ambiguous head-set spelling"):
        modernize({"n_sig": 1, "n_scanned": 2})


def test_read_manifest_modernizes_legacy_kind(tmp_path):
    path = tmp_path / "selected_heads.json"
    write_json_atomic(
        path,
        {
            "kind": "significant_heads",
            "schema_version": 1,
            "significant_heads": [[1, 2]],
        },
    )
    manifest = read_manifest(path, expect_kind="selected_heads", max_schema_version=2)
    assert manifest["kind"] == "selected_heads"
    assert manifest["selected_heads"] == [[1, 2]]


def test_committed_legacy_fixtures_round_trip():
    import json
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures"
    for name in ("selector_scan_repro_v1.json", "selector_expected_repro_v1.json"):
        legacy = json.loads((fixtures / name).read_text(encoding="utf-8"))
        assert canonicalize(modernize(legacy)) == legacy, name
        assert semantic_fingerprint(modernize(legacy)) == semantic_fingerprint(legacy)
