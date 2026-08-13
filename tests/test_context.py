import pytest
import yaml

from subspaces.context import ContextError, load_context
from subspaces.runners import step2 as step2_cli


def _write_context(tmp_path, payload):
    path = tmp_path / "context.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


VALID = {
    "model": {"name": "test-model", "revision": "abc"},
    "task": {"family": "number_add", "prompt_format": "arrow", "n_shot": 5},
    "samples": {"final_eval": {"path": "log/runs/x/step1/samples/final_eval.json"}},
    "activations": {"path": "log/cache/z/deadbeef"},
}


def test_load_valid_context(tmp_path):
    context = load_context(_write_context(tmp_path, VALID))
    assert context.model["name"] == "test-model"
    description = context.describe()
    assert "number_add" in description and "arrow" in description


def test_context_requires_model_and_task(tmp_path):
    with pytest.raises(ContextError, match="model.name"):
        load_context(_write_context(tmp_path, {**VALID, "model": {}}))
    broken_task = {**VALID, "task": {"family": "number_add"}}
    with pytest.raises(ContextError, match="task.prompt_format"):
        load_context(_write_context(tmp_path, broken_task))


def test_context_rejects_unknown_keys(tmp_path):
    with pytest.raises(ContextError, match="unknown key"):
        load_context(_write_context(tmp_path, {**VALID, "nope": 1}))


def test_step2_dry_run_with_context(fake_repo, tmp_path, capsys):
    context_path = _write_context(tmp_path, VALID)
    rc = step2_cli.main(
        [
            "--root",
            str(fake_repo),
            "--heads",
            "15:2",
            "--context",
            str(context_path),
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "context:" in out and "test-model" in out


def _heads_artifact(fake_repo, tmp_path):
    from subspaces.artifacts import make_manifest, write_json_atomic
    from subspaces.paths import ProjectPaths

    heads_manifest = make_manifest(
        kind="heads",
        schema_version=1,
        paths=ProjectPaths.from_root(fake_repo),
        payload={
            "significant_heads": [[15, 2], [13, 6]],
            "main_heads": [[15, 2]],
            "minor_heads": [],
            "model_dims": {"n_layers": 32, "n_heads": 32},
        },
    )
    heads_path = tmp_path / "heads.json"
    write_json_atomic(heads_path, heads_manifest)
    return heads_path


def test_step2_heads_from_context(fake_repo, tmp_path, capsys):
    heads_path = _heads_artifact(fake_repo, tmp_path)
    context_path = _write_context(
        tmp_path, {**VALID, "heads_artifact": str(heads_path)}
    )
    rc = step2_cli.main(
        ["--root", str(fake_repo), "--context", str(context_path), "--dry-run"]
    )
    assert rc == 0
    assert "15:2" in capsys.readouterr().out


def test_explicit_heads_override_context_heads_artifact(fake_repo, tmp_path, capsys):
    """--heads plus a context that also names a heads artifact must NOT
    collide: explicit heads win (arbitrary heads + shared context is a core
    requirement)."""
    heads_path = _heads_artifact(fake_repo, tmp_path)
    context_path = _write_context(
        tmp_path, {**VALID, "heads_artifact": str(heads_path)}
    )
    rc = step2_cli.main(
        [
            "--root",
            str(fake_repo),
            "--heads",
            "31:7",
            "--context",
            str(context_path),
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "31:7" in out and "15:2" not in out


def test_step2_run_with_valid_context_but_no_z_cache_refuses(
    fake_repo, tmp_path, capsys
):
    """Execution with a sufficient, existing-refs context passes validation
    but refuses clearly when no z cache is referenced (still zero artifacts).
    Full execution is covered in tests/test_step2_pca.py."""
    from conftest import tree_snapshot

    heads_path = _heads_artifact(fake_repo, tmp_path)
    sample_file = tmp_path / "final_eval.json"
    sample_file.write_text("{}", encoding="utf-8")
    payload = {
        **VALID,
        "heads_artifact": str(heads_path),
        "samples": {"final_eval": {"path": str(sample_file)}},
        "activations": {},
    }
    context_path = _write_context(tmp_path, payload)
    before = tree_snapshot(fake_repo)
    rc = step2_cli.main(["--root", str(fake_repo), "--context", str(context_path)])
    assert rc == 2
    assert "z cache" in capsys.readouterr().err.lower()
    assert tree_snapshot(fake_repo) == before


def test_step2_run_refuses_missing_context_ref(fake_repo, tmp_path, capsys):
    payload = {
        **VALID,
        "samples": {"final_eval": {"path": str(tmp_path / "nope.json")}},
    }
    context_path = _write_context(tmp_path, payload)
    rc = step2_cli.main(
        [
            "--root",
            str(fake_repo),
            "--heads",
            "15:2",
            "--context",
            str(context_path),
        ]
    )
    assert rc == 2
    assert "does not exist" in capsys.readouterr().err


def test_context_ref_fingerprint_mismatch_refused(fake_repo, tmp_path, capsys):
    from subspaces.context import ContextError, load_context
    from subspaces.paths import ProjectPaths

    heads_path = _heads_artifact(fake_repo, tmp_path)
    payload = {
        **VALID,
        "heads_artifact": {
            "path": str(heads_path),
            "semantic_fingerprint": "0" * 64,
        },
    }
    context = load_context(_write_context(tmp_path, payload))
    with pytest.raises(ContextError, match="fingerprint mismatch"):
        context.validate_refs(ProjectPaths.from_root(fake_repo))
