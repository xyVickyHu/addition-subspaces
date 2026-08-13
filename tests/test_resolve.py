from pathlib import Path

import pytest

from subspaces.config import ConfigError, load_step1_config
from subspaces.paths import ProjectPaths, find_repo_root
from subspaces.step1.pipeline import ensure_run, load_split
from subspaces.step1.resolve import (
    resolve_model_identity,
    resolve_runtime,
    run_id_for,
)

REPO = find_repo_root(Path(__file__))


def _cfg(fake_paths):
    return load_step1_config(fake_paths.root / "configs" / "step1_test.yaml")


def _fake_hf_cache(tmp_path, model_name="test-model", revision="abc123"):
    slug = "models--" + model_name.replace("/", "--")
    model_dir = tmp_path / slug
    (model_dir / "refs").mkdir(parents=True)
    (model_dir / "refs" / "main").write_text(revision + "\n", encoding="utf-8")
    snapshot = model_dir / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def _no_ambient_hf_cache(monkeypatch):
    """Tests control cache visibility explicitly."""
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.delenv(var, raising=False)


def test_identity_and_validation_split(fake_paths):
    cfg = _cfg(fake_paths)
    split = load_split(cfg, fake_paths)
    resolved = resolve_runtime(cfg, fake_paths, split)

    identity = resolved["identity"]
    assert identity["dataset_fingerprint"] == split.dataset_fingerprint
    assert identity["checkpoint"]["epoch"] == 10
    assert len(identity["split_sha256"]) == 64
    assert identity["requested"]["task"]["prompt_format"] == "arrow"
    assert identity["requested"]["samples"]["scan"]["seed"] == 42
    assert "reuse_path" not in identity["requested"]["matrix"]  # locator excluded
    assert "revision" not in identity["requested"]["model"]
    assert identity["model_revision"] is None  # unpinned, no cache
    # v2 = largest_gap method added (intentional identity fork)
    assert identity["algorithm_versions"]["subspaces.step1.significant"] == 2

    validation = resolved["validation"]
    assert validation["model"]["status"] == "unresolved"
    assert validation["matrix_provenance"]["status"] == "ok"
    assert validation["environment"]["python"]
    # status/cache fields are NOT part of identity
    assert "status" not in str(identity)


def test_run_id_invariant_to_matrix_relocation(fake_paths):
    cfg_rel = _cfg(fake_paths)
    split = load_split(cfg_rel, fake_paths)
    id_rel = run_id_for(cfg_rel, resolve_runtime(cfg_rel, fake_paths, split))

    cfg_abs = _cfg(fake_paths)
    cfg_abs.matrix.reuse_path = str(fake_paths.root / "matrices" / "toy")
    id_abs = run_id_for(cfg_abs, resolve_runtime(cfg_abs, fake_paths, split))
    assert id_rel == id_abs  # relative vs absolute locator: same run


def test_run_id_invariant_to_split_file_relocation(fake_paths):
    """A byte-identical split file at a different path must not fork the run
    (audit finding: task_split is a locator; content identity is
    split_sha256)."""
    import shutil

    cfg = _cfg(fake_paths)
    split = load_split(cfg, fake_paths)
    before = run_id_for(cfg, resolve_runtime(cfg, fake_paths, split))

    original = fake_paths.root / "configs" / "task_splits" / "number_add_test.yaml"
    moved = original.with_name("number_add_test_MOVED.yaml")
    shutil.copyfile(original, moved)
    cfg.task.task_split = "configs/task_splits/number_add_test_MOVED.yaml"
    split2 = load_split(cfg, fake_paths)
    after = run_id_for(cfg, resolve_runtime(cfg, fake_paths, split2))
    assert before == after


def test_run_id_changes_with_checkpoint_content(fake_paths):
    cfg = _cfg(fake_paths)
    split = load_split(cfg, fake_paths)
    before = run_id_for(cfg, resolve_runtime(cfg, fake_paths, split))

    checkpoint = (
        fake_paths.root / "matrices" / "toy" / "checkpoints" / "matrix_epoch10.pth"
    )
    checkpoint.write_bytes(b"different-weights")
    after = run_id_for(cfg, resolve_runtime(cfg, fake_paths, split))
    assert before != after  # content change -> new identity


def test_run_id_changes_with_dataset_content(fake_paths):
    cfg = _cfg(fake_paths)
    split = load_split(cfg, fake_paths)
    before = run_id_for(cfg, resolve_runtime(cfg, fake_paths, split))

    victim = fake_paths.task_dir("number_add") / "number-add1.json"
    victim.write_text(victim.read_text().replace('"2"', '"999"', 1), encoding="utf-8")
    # split fingerprint validation notices first — rebuild split file to match
    import yaml

    from subspaces.artifacts import dataset_fingerprint

    split_path = fake_paths.root / "configs" / "task_splits" / "number_add_test.yaml"
    raw = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    raw["dataset_fingerprint"] = dataset_fingerprint(fake_paths.task_dir("number_add"))
    split_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    split2 = load_split(cfg, fake_paths)
    after = run_id_for(cfg, resolve_runtime(cfg, fake_paths, split2))
    assert before != after


def test_unresolved_to_resolved_continues_same_run(fake_paths, tmp_path, monkeypatch):
    """With a PINNED revision, resolving the cache later must neither fork the
    run nor trip the drift refusal; validation status is updated in place."""
    cfg = _cfg(fake_paths)
    cfg.model.revision = "abc123"

    run_dir1, _, resolved1 = ensure_run(cfg, fake_paths)
    assert resolved1["validation"]["model"]["status"] == "unresolved"
    assert resolved1["identity"]["model_revision"] == "abc123"

    cache = _fake_hf_cache(tmp_path / "hf")
    monkeypatch.setenv("HF_HOME", str(cache))
    run_dir2, _, resolved2 = ensure_run(cfg, fake_paths)  # must not raise
    assert run_dir2 == run_dir1
    assert resolved2["validation"]["model"]["status"] == "resolved"
    resolved_text = (run_dir2 / "resolved_config.yaml").read_text(encoding="utf-8")
    assert "status: resolved" in resolved_text  # validation updated on disk


def test_unpinned_revision_participates_in_identity(fake_paths, tmp_path, monkeypatch):
    cfg = _cfg(fake_paths)
    split = load_split(cfg, fake_paths)
    id_without = run_id_for(cfg, resolve_runtime(cfg, fake_paths, split))
    monkeypatch.setenv("HF_HOME", str(_fake_hf_cache(tmp_path / "hf")))
    id_with = run_id_for(cfg, resolve_runtime(cfg, fake_paths, split))
    assert id_without != id_with  # unpinned: cache resolution changes identity


def test_pinned_revision_conflicts_with_cache(fake_paths, tmp_path, monkeypatch):
    cfg = _cfg(fake_paths)
    cfg.model.revision = "not-the-cached-one"
    monkeypatch.setenv("HF_HOME", str(_fake_hf_cache(tmp_path / "hf")))
    split = load_split(cfg, fake_paths)
    with pytest.raises(ConfigError, match="local cache resolves"):
        resolve_runtime(cfg, fake_paths, split)


def test_prompt_format_env_contradiction_refused(fake_paths, monkeypatch):
    cfg = _cfg(fake_paths)
    split = load_split(cfg, fake_paths)
    monkeypatch.setenv("FV_PROMPT_FORMAT", "qa")
    with pytest.raises(ConfigError, match="FV_PROMPT_FORMAT"):
        resolve_runtime(cfg, fake_paths, split)


def test_ensure_run_writes_requested_and_resolved(fake_paths):
    cfg = _cfg(fake_paths)
    run_dir, _split, _resolved = ensure_run(cfg, fake_paths)
    assert (run_dir / "requested_config.yaml").is_file()
    resolved_text = (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    assert "identity:" in resolved_text and "validation:" in resolved_text
    ensure_run(cfg, fake_paths)  # idempotent when nothing changed


def test_model_identity_from_fake_hf_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(_fake_hf_cache(tmp_path, "test/tiny")))
    identity = resolve_model_identity("test/tiny")
    assert identity["status"] == "resolved"
    assert identity["revision"] == "abc123"
    assert identity["tokenizer"]["file"] == "tokenizer_config.json"


def test_model_identity_unresolved_without_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty"))
    identity = resolve_model_identity("nobody/nothing")
    assert identity["status"] == "unresolved"
    assert identity["revision"] is None


def test_real_repo_matrix_provenance_ok():
    """The shipped canonical matrix passes strict provenance against the
    committed corrected config (pinned epoch 49, number_add_1018 mapping)."""
    paths = ProjectPaths.from_root(REPO)
    cfg = load_step1_config(REPO / "configs" / "step1_number_add_llama3.yaml")
    from subspaces.step1.matrix import resolve_reuse_matrix

    manifest = resolve_reuse_matrix(cfg, paths)
    check = manifest["provenance_check"]
    assert check["status"] == "ok"
    assert check["recorded"]["task_dir_archived"] == "number_add_1018"
    assert check["checked"]["model_name"] == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert check["checked"]["prompt_format"] == "arrow"
    assert check["checked"]["inject_layer"] == "blocks.10.hook_resid_mid"
    assert len(check["args_sha256"]) == 64
    selected = manifest["selected_checkpoint"]
    assert selected["epoch"] == 49 and selected["selection_rule"] == "pinned"


def test_all_committed_configs_load():
    for name in (
        "step1_number_add_llama3.yaml",
        "step1_number_add_llama3_legacy_repro.yaml",
        "step1_number_add_llama3_tinysmoke.yaml",
    ):
        cfg = load_step1_config(REPO / "configs" / name)
        assert cfg.model.revision == "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
