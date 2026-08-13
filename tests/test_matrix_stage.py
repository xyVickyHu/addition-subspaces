import json

import pytest

from subspaces.artifacts import ArtifactError
from subspaces.config import load_step1_config
from subspaces.step1 import tree
from subspaces.step1.matrix import (
    check_provenance,
    find_checkpoints,
    resolve_reuse_matrix,
)
from subspaces.step1.pipeline import ensure_run, stage_matrix


def _cfg(fake_paths):
    return load_step1_config(fake_paths.root / "configs" / "step1_test.yaml")


def _toy_dir(fake_paths):
    return fake_paths.root / "matrices" / "toy"


def _staging_matrix_dir(fake_paths, journal_dir):
    """Where matrix.mode=train stages its work before the move to the node."""
    return (
        fake_paths.runs_dir / f".train-staging-{journal_dir.name}" / "step1" / "matrix"
    )


def _edit_args(fake_paths, **updates):
    args_path = _toy_dir(fake_paths) / "args.json"
    archived = json.loads(args_path.read_text(encoding="utf-8"))
    archived.update(updates)
    args_path.write_text(json.dumps(archived), encoding="utf-8")


def test_find_checkpoints_sorts_numerically(fake_paths):
    found = find_checkpoints(_toy_dir(fake_paths))
    assert [epoch for epoch, _ in found] == [0, 2, 10]  # not lexicographic


def test_resolve_reuse_matrix_defaults_to_latest(fake_paths):
    manifest = resolve_reuse_matrix(_cfg(fake_paths), fake_paths)
    selected = manifest["selected_checkpoint"]
    assert selected["epoch"] == 10
    assert selected["selection_rule"] == "latest"
    assert len(selected["sha256"]) == 64
    assert manifest["impl"]["module"] == "subspaces.step1.matrix"
    assert manifest["provenance_check"]["status"] == "ok"
    assert len(manifest["provenance_check"]["args_sha256"]) == 64


def test_checkpoint_epoch_pinning(fake_paths):
    cfg = _cfg(fake_paths)
    cfg.matrix.checkpoint_epoch = 2
    manifest = resolve_reuse_matrix(cfg, fake_paths)
    selected = manifest["selected_checkpoint"]
    assert selected["epoch"] == 2 and selected["selection_rule"] == "pinned"

    cfg.matrix.checkpoint_epoch = 7  # not present
    with pytest.raises(ArtifactError, match="checkpoint_epoch=7 not found"):
        resolve_reuse_matrix(cfg, fake_paths)


def test_provenance_model_mismatch_refused(fake_paths):
    _edit_args(fake_paths, model_name="some-other-model")
    with pytest.raises(ArtifactError, match="provenance mismatch"):
        resolve_reuse_matrix(_cfg(fake_paths), fake_paths)


def test_provenance_prompt_format_mismatch_refused(fake_paths):
    """An arrow-trained matrix (archived default) with prompt_format=qa must
    refuse, not pass with a note."""
    cfg = _cfg(fake_paths)
    cfg.task.prompt_format = "qa"
    with pytest.raises(ArtifactError, match="prompt_format"):
        resolve_reuse_matrix(cfg, fake_paths)


def test_provenance_site_mismatch_refused(fake_paths):
    cfg = _cfg(fake_paths)
    cfg.sites.inject_layer = "blocks.31.hook_resid_post"  # bogus site
    with pytest.raises(ArtifactError, match="inject_layer"):
        resolve_reuse_matrix(cfg, fake_paths)


def test_provenance_unmappable_archived_layer_refused(fake_paths):
    _edit_args(fake_paths, layer="late")  # no known mapping
    with pytest.raises(ArtifactError, match="inject_layer"):
        resolve_reuse_matrix(_cfg(fake_paths), fake_paths)


def test_provenance_override_is_explicit_and_recorded(fake_paths):
    cfg = _cfg(fake_paths)
    cfg.task.prompt_format = "qa"
    cfg.matrix.override_provenance = ["prompt_format"]
    manifest = resolve_reuse_matrix(cfg, fake_paths)
    check = manifest["provenance_check"]
    assert check["status"] == "ok_with_overrides"
    assert check["overridden"] == ["prompt_format"]


def test_provenance_task_alias_accepted(fake_paths):
    _edit_args(fake_paths, task_dir="number_add_1018")  # historical alias
    check = check_provenance(_toy_dir(fake_paths), _cfg(fake_paths))
    assert check["status"] == "ok"
    assert check["recorded"]["task_dir_archived"] == "number_add_1018"


def test_provenance_missing_args_json_recorded_unverified(fake_paths):
    (_toy_dir(fake_paths) / "args.json").unlink()
    check = check_provenance(_toy_dir(fake_paths), _cfg(fake_paths))
    assert check["status"] == "unverified"
    manifest = resolve_reuse_matrix(_cfg(fake_paths), fake_paths)
    assert manifest["provenance_check"]["status"] == "unverified"


def test_stage_matrix_persists_and_reuses(fake_paths):
    cfg = _cfg(fake_paths)
    journal_dir, _, _ = ensure_run(cfg, fake_paths)
    assert (journal_dir / "requested_config.yaml").is_file()
    assert "__" in journal_dir.name  # <run_name>__<identity-hash>

    first, node_dir = stage_matrix(cfg, fake_paths, journal_dir)
    assert node_dir == tree.matrix_node_dir(fake_paths, cfg, first)
    assert (node_dir / "matrix_ref.json").is_file()
    assert (journal_dir / "matrix_ref.json").is_file()  # journal memo
    second, node_again = stage_matrix(cfg, fake_paths, journal_dir)
    assert second == first  # reused, not regenerated
    assert node_again == node_dir


def test_stage_matrix_refuses_identity_change_in_place(fake_paths):
    """If a checkpoint is swapped under an EXISTING journal entry, the shared
    reuse rule refuses (the run ID would normally change first; this guards
    manual tampering)."""
    cfg = _cfg(fake_paths)
    journal_dir, _, _ = ensure_run(cfg, fake_paths)
    stage_matrix(cfg, fake_paths, journal_dir)

    checkpoint = _toy_dir(fake_paths) / "checkpoints" / "matrix_epoch10.pth"
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="refusing to reuse"):
        stage_matrix(cfg, fake_paths, journal_dir)


def test_matrix_identity_ignores_location(fake_paths, tmp_path):
    import shutil

    from subspaces.artifacts import semantic_fingerprint

    cfg = _cfg(fake_paths)
    before = semantic_fingerprint(resolve_reuse_matrix(cfg, fake_paths))

    moved = tmp_path / "elsewhere" / "toy"
    shutil.copytree(_toy_dir(fake_paths), moved)
    cfg.matrix.reuse_path = str(moved)
    after = semantic_fingerprint(resolve_reuse_matrix(cfg, fake_paths))
    assert before == after


def test_missing_checkpoints_refused(fake_paths, tmp_path):
    cfg = _cfg(fake_paths)
    cfg.matrix.reuse_path = str(tmp_path / "nowhere")
    with pytest.raises(ArtifactError, match="not found"):
        resolve_reuse_matrix(cfg, fake_paths)


# -- matrix.mode=train (legacy trainer driven in-process, stubbed here) -------


def _train_cfg(fake_paths):
    from subspaces.config import MatrixTrainConfig

    cfg = _cfg(fake_paths)
    cfg.matrix.mode = "train"
    cfg.matrix.reuse_path = None
    cfg.matrix.train = MatrixTrainConfig(
        epochs=3, bs=2, n_example=5, indist_limit=8, ood_limit=4
    )
    return cfg


def _parse_stub_argv(argv):
    opts, flags = {}, []
    i = 1  # skip program name
    while i < len(argv):
        assert argv[i].startswith("--"), argv[i]
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            opts[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            flags.append(argv[i][2:])
            i += 1
    return opts, flags


def _stub_trainer(
    root,
    calls,
    *,
    epochs_to_write=None,
    rebind_stdout=False,
    crash=False,
    attach_logging=None,
):
    """Mimics the legacy trainer's side effects for the argv it receives."""
    import sys
    from types import SimpleNamespace

    def main():
        import os

        opts, flags = _parse_stub_argv(sys.argv)
        calls.append(
            {"opts": opts, "flags": flags, "wandb_mode": os.environ.get("WANDB_MODE")}
        )
        version_dir = root / "log" / opts["version"]
        checkpoint_dir = version_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        version_dir.joinpath("args.json").write_text(
            json.dumps(
                {
                    "model_name": opts["model_name"],
                    "task_dir": opts["task_dir"],
                    "n_shot": int(opts["n_shot"]),
                    "layer_name": opts["layer_name"],
                    "seed": int(opts["seed"]),
                    "lambdaL1": float(opts["lambdaL1"]),
                    "zero_one_interval": "zero_one_interval" in flags,
                    "indist_keys": json.loads(opts["indist_keys"]),
                    "ood_keys": json.loads(opts["ood_keys"]),
                }
            ),
            encoding="utf-8",
        )
        if rebind_stdout:
            sys.stdout = open(version_dir / "output.txt", "a")  # noqa: SIM115
            sys.stderr = sys.stdout
        if attach_logging is not None:
            # simulate a library creating a handler DURING training, bound to
            # the redirected stream (the mid-training handler-rebind logging-error mechanism)
            import logging

            handler = logging.StreamHandler(sys.stderr)
            logging.getLogger().addHandler(handler)
            attach_logging.append(handler)
        if crash:
            raise RuntimeError("boom mid-training")
        existing = {
            int(p.name[len("matrix_epoch") : -len(".pth")])
            for p in checkpoint_dir.glob("matrix_epoch*.pth")
        }
        start = max(existing) + 1 if "resume" in opts else 0
        end = int(opts["epoch_num"]) if epochs_to_write is None else epochs_to_write
        import torch

        for epoch in range(start, end):
            torch.save(
                torch.full((2, 2), float(epoch)),
                checkpoint_dir / f"matrix_epoch{epoch}.pth",
            )

    return SimpleNamespace(PROJECT_ROOT=str(root), main=main)


def _install_stub(monkeypatch, stub):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "subspaces.runners.train_matrix", stub)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=None))


def test_train_fresh_builds_trained_matrix_ref(fake_paths, monkeypatch):
    cfg = _train_cfg(fake_paths)
    journal_dir, _, _ = ensure_run(cfg, fake_paths)
    calls = []
    _install_stub(monkeypatch, _stub_trainer(fake_paths.root, calls))

    manifest, node_dir = stage_matrix(cfg, fake_paths, journal_dir)

    assert len(calls) == 1
    opts, flags = calls[0]["opts"], calls[0]["flags"]
    # training runs in a journal-scoped staging dir (the node key — the
    # checkpoint sha — is unknowable pre-training), then moves to the node
    assert opts["version"] == f"runs/.train-staging-{journal_dir.name}/step1/matrix"
    assert json.loads(opts["indist_keys"]) == ["number-add1", "number-add3"]
    assert json.loads(opts["ood_keys"]) == ["number-add2"]
    assert opts["ood_task_num"] == "1"
    assert opts["epoch_num"] == "3" and opts["bs"] == "2"
    assert opts["lambdaL1"] == "0.05" and opts["anneal"] == "1"
    assert "zero_one_interval" in flags
    assert "resume" not in opts and "savevar" not in flags

    assert manifest["provenance"] == "trained"
    selected = manifest["selected_checkpoint"]
    assert selected["epoch"] == 2
    assert selected["selection_rule"] == "trained_final_epoch"
    assert manifest["provenance_check"]["status"] == "ok"
    assert node_dir == tree.matrix_node_dir(fake_paths, cfg, manifest)
    assert (node_dir / "matrix_ref.json").is_file()
    assert (journal_dir / "matrix_ref.json").is_file()  # journal memo
    # the staging dir is gone: the matrix was moved under the node
    assert not _staging_matrix_dir(fake_paths, journal_dir).exists()
    # effective prompt format annotated into the trainer's args.json
    archived = json.loads(
        (node_dir / "matrix" / "args.json").read_text(encoding="utf-8")
    )
    assert archived["prompt_format"] == "arrow"
    # re-invocation returns the journal memo without retraining
    again, node_again = stage_matrix(cfg, fake_paths, journal_dir)
    assert len(calls) == 1
    assert again == manifest and node_again == node_dir


def test_train_resumes_from_partial_checkpoints(fake_paths, monkeypatch):
    cfg = _train_cfg(fake_paths)
    journal_dir, _, _ = ensure_run(cfg, fake_paths)
    import torch

    # an interrupted training left partial checkpoints in the staging dir
    checkpoint_dir = _staging_matrix_dir(fake_paths, journal_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for epoch in (0, 1):
        torch.save(torch.zeros(1), checkpoint_dir / f"matrix_epoch{epoch}.pth")
    epoch0_bytes = (checkpoint_dir / "matrix_epoch0.pth").read_bytes()
    calls = []
    _install_stub(monkeypatch, _stub_trainer(fake_paths.root, calls))

    manifest, node_dir = stage_matrix(cfg, fake_paths, journal_dir)
    assert "resume" in calls[0]["opts"]
    # earlier checkpoints kept, only the missing epoch trained (checkpoints
    # now live under the node after the move)
    moved = node_dir / "matrix" / "checkpoints"
    assert (moved / "matrix_epoch0.pth").read_bytes() == epoch0_bytes
    assert torch.load(moved / "matrix_epoch2.pth")[0, 0].item() == 2.0
    assert manifest["selected_checkpoint"]["epoch"] == 2


def test_train_skips_when_fully_trained(fake_paths, monkeypatch):
    cfg = _train_cfg(fake_paths)
    journal_dir, _, _ = ensure_run(cfg, fake_paths)
    import torch

    matrix_dir = _staging_matrix_dir(fake_paths, journal_dir)
    (matrix_dir / "checkpoints").mkdir(parents=True)
    for epoch in range(3):
        torch.save(
            torch.zeros(1), matrix_dir / "checkpoints" / f"matrix_epoch{epoch}.pth"
        )
    (matrix_dir / "args.json").write_text(
        json.dumps(
            {
                "model_name": cfg.model.name,
                "task_dir": "number_add",
                "n_shot": cfg.task.n_shot,
                "layer_name": "blocks.10.hook_resid_mid",
            }
        ),
        encoding="utf-8",
    )
    calls = []
    _install_stub(monkeypatch, _stub_trainer(fake_paths.root, calls))

    manifest, _node_dir = stage_matrix(cfg, fake_paths, journal_dir)
    assert calls == []  # trainer never invoked
    assert manifest["provenance"] == "trained"


def test_train_restores_stdio_and_argv_on_crash(fake_paths, monkeypatch):
    import sys

    cfg = _train_cfg(fake_paths)
    run_dir, _, _ = ensure_run(cfg, fake_paths)
    calls = []
    _install_stub(
        monkeypatch,
        _stub_trainer(fake_paths.root, calls, rebind_stdout=True, crash=True),
    )
    argv_before, stdout_before, stderr_before = sys.argv, sys.stdout, sys.stderr
    with pytest.raises(RuntimeError, match="boom"):
        stage_matrix(cfg, fake_paths, run_dir)
    assert sys.argv is argv_before
    assert sys.stdout is stdout_before
    assert sys.stderr is stderr_before


def test_train_incomplete_epochs_refused(fake_paths, monkeypatch):
    cfg = _train_cfg(fake_paths)
    run_dir, _, _ = ensure_run(cfg, fake_paths)
    _install_stub(monkeypatch, _stub_trainer(fake_paths.root, [], epochs_to_write=1))
    with pytest.raises(ArtifactError, match="without epoch checkpoint"):
        stage_matrix(cfg, fake_paths, run_dir)


def test_train_requires_bfloat16(fake_paths):
    cfg = _train_cfg(fake_paths)
    cfg.model.dtype = "float32"
    with pytest.raises(ArtifactError, match="bfloat16"):
        stage_matrix(cfg, fake_paths, fake_paths.root / "log" / "runs" / "x")


def test_train_refuses_mismatched_prompt_format_env(fake_paths, monkeypatch):
    cfg = _train_cfg(fake_paths)
    monkeypatch.setenv("FV_PROMPT_FORMAT", "qa")
    with pytest.raises(ArtifactError, match="FV_PROMPT_FORMAT"):
        stage_matrix(cfg, fake_paths, fake_paths.root / "log" / "runs" / "x")


def test_train_variant_config_loads():
    from pathlib import Path

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "step1_number_add_llama3_train.yaml"
    )
    cfg = load_step1_config(config_path)
    assert cfg.matrix.mode == "train"
    assert cfg.matrix.train.epochs == 50
    assert cfg.matrix.train.lambda_l1 == 0.05
    assert cfg.matrix.train.anneal == 1
    assert cfg.protocol == "corrected_holdout"


def test_train_enforces_wandb_mode_from_config(fake_paths, monkeypatch):
    cfg = _train_cfg(fake_paths)
    run_dir, _, _ = ensure_run(cfg, fake_paths)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    calls = []
    _install_stub(monkeypatch, _stub_trainer(fake_paths.root, calls))
    stage_matrix(cfg, fake_paths, run_dir)
    assert calls[0]["wandb_mode"] == "offline"  # config field enforced via env


def test_train_refuses_stray_files_in_checkpoint_dir(fake_paths, monkeypatch):
    import torch

    cfg = _train_cfg(fake_paths)
    journal_dir, _, _ = ensure_run(cfg, fake_paths)
    checkpoint_dir = _staging_matrix_dir(fake_paths, journal_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    torch.save(torch.zeros(1), checkpoint_dir / "matrix_epoch0.pth")
    (checkpoint_dir / "notes.txt").write_text("stray", encoding="utf-8")
    _install_stub(monkeypatch, _stub_trainer(fake_paths.root, []))
    with pytest.raises(ArtifactError, match="unexpected entries"):
        stage_matrix(cfg, fake_paths, journal_dir)


def test_train_refuses_non_contiguous_checkpoints(fake_paths, monkeypatch):
    import torch

    cfg = _train_cfg(fake_paths)
    journal_dir, _, _ = ensure_run(cfg, fake_paths)
    checkpoint_dir = _staging_matrix_dir(fake_paths, journal_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for epoch in (0, 2):  # gap at 1
        torch.save(torch.zeros(1), checkpoint_dir / f"matrix_epoch{epoch}.pth")
    _install_stub(monkeypatch, _stub_trainer(fake_paths.root, []))
    with pytest.raises(ArtifactError, match="non-contiguous"):
        stage_matrix(cfg, fake_paths, journal_dir)


def test_train_refuses_corrupt_resume_checkpoint(fake_paths, monkeypatch):
    import torch

    cfg = _train_cfg(fake_paths)
    journal_dir, _, _ = ensure_run(cfg, fake_paths)
    checkpoint_dir = _staging_matrix_dir(fake_paths, journal_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    torch.save(torch.zeros(1), checkpoint_dir / "matrix_epoch0.pth")
    (checkpoint_dir / "matrix_epoch1.pth").write_bytes(b"partial-write")
    _install_stub(monkeypatch, _stub_trainer(fake_paths.root, []))
    with pytest.raises(ArtifactError, match="unreadable"):
        stage_matrix(cfg, fake_paths, journal_dir)


def test_train_config_bounds_rejected(fake_paths):
    from subspaces.config import ConfigError

    for field, bad in (
        ("epochs", 0),
        ("test_gap", 0),
        ("bs", 0),
        ("train_ratio", 1.0),
        ("lr", 0.0),
        ("anneal", 4),
    ):
        cfg = _train_cfg(fake_paths)
        setattr(cfg.matrix.train, field, bad)
        with pytest.raises(ConfigError, match=f"matrix.train.{field}"):
            cfg.validate()


def test_train_retargets_logging_handlers_before_closing_log(fake_paths, monkeypatch):
    """Handlers created during training (bound to the redirected output.txt)
    must be pointed back at the real stderr before the log file closes —
    otherwise every later logging call in the process errors on a closed
    stream (observed in a GPU run)."""
    import logging
    import sys

    cfg = _train_cfg(fake_paths)
    run_dir, _, _ = ensure_run(cfg, fake_paths)
    handlers = []
    _install_stub(
        monkeypatch,
        _stub_trainer(fake_paths.root, [], rebind_stdout=True, attach_logging=handlers),
    )
    try:
        stage_matrix(cfg, fake_paths, run_dir)
        (handler,) = handlers
        assert handler.stream is sys.stderr  # retargeted to the restored stream
        assert not handler.stream.closed
        logging.getLogger().warning("post-training logging must not raise")
    finally:
        for handler in handlers:
            logging.getLogger().removeHandler(handler)


def test_stage_matrix_refuses_node_name_divergence(fake_paths):
    """The SAME matrix identity under a different node_name must refuse
    instead of silently duplicating the whole subtree (and its GPU scans)."""
    cfg = _cfg(fake_paths)
    journal_dir, _, _ = ensure_run(cfg, fake_paths)
    _, node_dir = stage_matrix(cfg, fake_paths, journal_dir)

    cfg2 = _cfg(fake_paths)
    cfg2.matrix.node_name = "renamed-toy"
    journal2, _, _ = ensure_run(cfg2, fake_paths)
    with pytest.raises(ArtifactError, match="already materialized under") as excinfo:
        stage_matrix(cfg2, fake_paths, journal2)
    message = str(excinfo.value)
    assert node_dir.name in message  # names the existing node
    assert "renamed-toy" in message  # and the would-be duplicate
    node_key = node_dir.name.rsplit("__", 1)[1]
    assert not (fake_paths.runs_dir / f"renamed-toy__{node_key}").exists()


def test_train_refuses_orphan_trained_node(fake_paths, monkeypatch):
    """A node dir with matrix/checkpoints/ but no matrix_ref.json (crash
    between the staging rename and the ref write) must surface, not be
    silently shadowed by a retrain."""
    import torch

    cfg = _train_cfg(fake_paths)
    journal_dir, _, _ = ensure_run(cfg, fake_paths)
    orphan_dir = fake_paths.runs_dir / "orphan__deadbeef1234"
    (orphan_dir / "matrix" / "checkpoints").mkdir(parents=True)
    torch.save(
        torch.zeros(1), orphan_dir / "matrix" / "checkpoints" / "matrix_epoch0.pth"
    )
    calls = []
    _install_stub(monkeypatch, _stub_trainer(fake_paths.root, calls))
    with pytest.raises(ArtifactError, match="orphan") as excinfo:
        stage_matrix(cfg, fake_paths, journal_dir)
    assert str(orphan_dir) in str(excinfo.value)
    assert calls == []  # refused BEFORE any retraining


def test_train_adopts_usable_node_despite_orphan(fake_paths, monkeypatch):
    """An adoptable trained node wins; unrelated orphan dirs do not block."""
    import torch

    cfg = _train_cfg(fake_paths)
    journal_dir, _, _ = ensure_run(cfg, fake_paths)
    calls = []
    _install_stub(monkeypatch, _stub_trainer(fake_paths.root, calls))
    manifest, node_dir = stage_matrix(cfg, fake_paths, journal_dir)
    assert len(calls) == 1

    orphan_dir = fake_paths.runs_dir / "orphan__deadbeef1234"
    (orphan_dir / "matrix" / "checkpoints").mkdir(parents=True)
    torch.save(
        torch.zeros(1), orphan_dir / "matrix" / "checkpoints" / "matrix_epoch0.pth"
    )
    (journal_dir / "matrix_ref.json").unlink()  # force the adoption scan
    adopted, node_again = stage_matrix(cfg, fake_paths, journal_dir)
    assert len(calls) == 1  # adopted, not retrained
    assert node_again == node_dir


def test_cli_adopts_config_prompt_format_no_manual_export(fake_paths, monkeypatch):
    """A non-arrow train config must work through the generic CLI with NO
    manually synchronized FV_PROMPT_FORMAT export (sweep-readiness item 2);
    empty string counts as unset."""
    import os

    import yaml

    import subspaces.runners.step1 as cli

    config_path = fake_paths.root / "configs" / "step1_test.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["run_name"] = "testtrainqa"
    raw["task"]["prompt_format"] = "qa"
    raw["matrix"] = {
        "mode": "train",
        "train": {
            "epochs": 3,
            "bs": 2,
            "n_example": 5,
            "indist_limit": 8,
            "ood_limit": 4,
        },
    }
    qa_path = fake_paths.root / "configs" / "step1_test_train_qa.yaml"
    qa_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    monkeypatch.setenv("FV_PROMPT_FORMAT", "")  # empty == unset
    calls = []
    _install_stub(monkeypatch, _stub_trainer(fake_paths.root, calls))
    rc = cli.main(["--root", str(fake_paths.root), "train", "--config", str(qa_path)])
    assert rc == 0
    assert os.environ["FV_PROMPT_FORMAT"] == "qa"  # adopted from the config
    assert len(calls) == 1  # trainer ran without a manual export
    # the trained matrix's provenance records the effective format (node
    # named from the config's run_name)
    node_dirs = list((fake_paths.root / "log" / "runs").glob("testtrainqa__*"))
    (node_dir,) = node_dirs
    archived = json.loads(
        (node_dir / "matrix" / "args.json").read_text(encoding="utf-8")
    )
    assert archived["prompt_format"] == "qa"


def test_adoption_is_provenance_gated(fake_paths, monkeypatch):
    """Adoption of a trained node must check args.json provenance, not only
    the matrix request block: the four format matrices share ONE matrix
    block and differ only in task.prompt_format (GPU runs — the
    fmt-fx evaluation adopted the fmt-ab node; the divergence guard aborted
    the wave)."""
    from subspaces.step1.pipeline import _find_trained_node

    cfg = _train_cfg(fake_paths)
    journal_dir, _, _ = ensure_run(cfg, fake_paths)
    calls = []
    _install_stub(monkeypatch, _stub_trainer(fake_paths.root, calls))
    manifest, _node_dir = stage_matrix(cfg, fake_paths, journal_dir)

    # same train request + same format, fresh run identity -> adopts
    same_format = _train_cfg(fake_paths)
    same_format.scan.c_max = 5  # forks the run, not the train request
    adopted = _find_trained_node(same_format, fake_paths)
    assert adopted is not None
    assert adopted["matrix_dir"] == manifest["matrix_dir"]

    # same train request, DIFFERENT prompt format -> a different training
    other_format = _train_cfg(fake_paths)
    other_format.task.prompt_format = "qa"
    assert _find_trained_node(other_format, fake_paths) is None


def test_split_provenance_guard(fake_paths):
    """A reused matrix whose archived TRAINING split disagrees with the
    configured task split must refuse (2026-07-27 phi-4 add finding: its
    historical matrix trained on 4 of the committed 5 eval tasks); the
    named task_split override accepts it explicitly; a matching archive
    passes; archives without keys are skipped (pre-schema)."""
    import json as _json

    from subspaces.step1.matrix import resolve_reuse_matrix

    cfg = _cfg(fake_paths)
    args_path = fake_paths.resolve("matrices/toy") / "args.json"
    archived = _json.loads(args_path.read_text(encoding="utf-8"))

    # matching split (the committed 2/1 fixture split) -> passes
    archived["indist_keys"] = ["number-add1", "number-add3"]
    archived["ood_keys"] = ["number-add2"]
    args_path.write_text(_json.dumps(archived), encoding="utf-8")
    assert resolve_reuse_matrix(cfg, fake_paths)["provenance"] == "reuse"

    # the configured EVAL task inside the archived TRAINING set -> refuse
    archived["indist_keys"] = ["number-add1", "number-add2"]
    archived["ood_keys"] = ["number-add3"]
    args_path.write_text(_json.dumps(archived), encoding="utf-8")
    with pytest.raises(ArtifactError, match="TRAINING-split provenance"):
        resolve_reuse_matrix(cfg, fake_paths)

    # named override accepts it explicitly
    cfg.matrix.override_provenance = ["task_split"]
    assert resolve_reuse_matrix(cfg, fake_paths)["provenance"] == "reuse"
