"""Steps 2/3 scaffolds and unimplemented Step-1 substeps must fail clearly and
never create artifacts; --dry-run validates without writing anything."""

from conftest import tree_snapshot

from subspaces.runners import run_all as run_all_cli
from subspaces.runners import step1 as step1_cli
from subspaces.runners import step2 as step2_cli
from subspaces.runners import step3 as step3_cli
from subspaces.step1.selectors import SelectorError, get_selector


def test_step2_dry_run_ok_and_writes_nothing(fake_repo, capsys):
    before = tree_snapshot(fake_repo)
    rc = step2_cli.main(["--root", str(fake_repo), "--heads", "15:2,13:6", "--dry-run"])
    assert rc == 0
    assert "dry run" in capsys.readouterr().out
    assert tree_snapshot(fake_repo) == before


def test_step2_run_without_context_fails_clearly_no_artifacts(fake_repo, capsys):
    before = tree_snapshot(fake_repo)
    rc = step2_cli.main(["--root", str(fake_repo), "--heads", "15:2"])
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "requires --context" in err
    assert tree_snapshot(fake_repo) == before


def test_step2_rejects_unknown_plugin(fake_repo, capsys):
    rc = step2_cli.main(
        ["--root", str(fake_repo), "--heads", "15:2", "--plugin", "nope", "--dry-run"]
    )
    assert rc == 2
    assert "unknown step2 plugin" in capsys.readouterr().err


def test_step3_dry_run_and_run(fake_repo, capsys):
    before = tree_snapshot(fake_repo)
    assert (
        step3_cli.main(["--root", str(fake_repo), "--heads", "15:2", "--dry-run"]) == 0
    )
    rc = step3_cli.main(["--root", str(fake_repo), "--heads", "15:2"])
    assert rc == 2  # execution requires a context
    assert tree_snapshot(fake_repo) == before


def test_heads_required_error_is_clear(fake_repo, capsys):
    rc = step2_cli.main(["--root", str(fake_repo), "--dry-run"])
    assert rc == 2
    assert "--heads" in capsys.readouterr().err


def test_step1_select_significant_succeeds(fake_repo, capsys):
    from subspaces.config import load_step1_config
    from subspaces.step1 import tree

    config = str(fake_repo / "configs" / "step1_test.yaml")
    rc = step1_cli.main(
        ["--root", str(fake_repo), "select-significant", "--config", config]
    )
    assert rc == 0
    # journal entry (per-invocation config record) + one matrix node
    journal_dirs = list((fake_repo / "log" / "journal").iterdir())
    assert len(journal_dirs) == 1
    assert (journal_dirs[0] / "requested_config.yaml").is_file()
    node_dirs = list((fake_repo / "log" / "runs").iterdir())
    assert len(node_dirs) == 1
    assert (node_dirs[0] / "matrix_ref.json").is_file()
    cfg = load_step1_config(fake_repo / "configs" / "step1_test.yaml")
    sig_node = tree.sig_node_dir(node_dirs[0], cfg.significant)
    assert (sig_node / "significant_heads.json").is_file()


def test_step1_gpu_substep_refuses_unresolved_model(fake_repo, capsys):
    """Without a pinned+resolvable model identity, the GPU-touching scan
    refuses clearly and leaves no scan/downstream artifacts."""
    config = str(fake_repo / "configs" / "step1_test.yaml")
    rc = step1_cli.main(["--root", str(fake_repo), "scan", "--config", config])
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "model identity is unresolved" in err
    node_dirs = list((fake_repo / "log" / "runs").iterdir())
    assert len(node_dirs) == 1
    # the CPU substeps may have staged their nodes, but no scan node (or any
    # downstream artifact) may exist after the refusal
    assert not list(node_dirs[0].glob("sig-*/scan-*"))
    assert not list(node_dirs[0].rglob("heads.json"))


def test_run_all_refuses_unresolved_model_before_gpu_work(fake_repo, capsys):
    config = str(fake_repo / "configs" / "step1_test.yaml")
    rc = run_all_cli.main(
        ["--root", str(fake_repo), "--config", config, "--through", "step1"]
    )
    assert rc == 2
    assert "model identity is unresolved" in capsys.readouterr().err.lower()


def test_step1_matrix_flag_overrides_config(fake_repo, capsys):
    """--matrix is wired: the matrix_ref must point at the override, and the
    override participates in the node identity (node named from the override,
    keyed by the override's checkpoint)."""
    import shutil

    import torch

    toy2 = fake_repo / "matrices" / "toy2"
    shutil.copytree(fake_repo / "matrices" / "toy", toy2)
    other = torch.zeros(4, 4)
    other[3, 3] = 0.7
    torch.save(other, toy2 / "checkpoints" / "matrix_epoch10.pth")

    config = str(fake_repo / "configs" / "step1_test.yaml")
    rc = step1_cli.main(
        [
            "--root",
            str(fake_repo),
            "select-significant",
            "--config",
            config,
            "--matrix",
            "matrices/toy2",
        ]
    )
    assert rc == 0
    node_dirs = list((fake_repo / "log" / "runs").iterdir())
    assert len(node_dirs) == 1
    assert node_dirs[0].name.startswith("toy2__")  # named from the override
    ref = (node_dirs[0] / "matrix_ref.json").read_text(encoding="utf-8")
    assert "matrices/toy2" in ref


def test_step1_select_main_param_wiring(fake_repo, tmp_path, capsys):
    import json

    from subspaces.artifacts import make_manifest, write_json_atomic
    from subspaces.paths import ProjectPaths

    scan = make_manifest(
        kind="head_scan",
        schema_version=1,
        paths=ProjectPaths.from_root(fake_repo),
        payload={
            "curves": {"15:2": {"0": 0.1, "1": 0.5, "2": 0.6}},
            "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
            "n_eval_examples_per_head_per_c": 300,
            "c_grid": [0, 1, 2],
        },
    )
    scan_path = tmp_path / "head_scan.json"
    write_json_atomic(scan_path, scan)

    base = ["--root", str(fake_repo), "select-main", "--scan", str(scan_path)]
    # valid typed override runs the selector and writes an immutable artifact
    # into its main-* node dir (default parent: the scan's directory)
    rc = step1_cli.main([*base, "--selector", "unified", "--param", "beta=0.3"])
    assert rc == 0
    main_nodes = list(tmp_path.glob("main-*"))
    assert len(main_nodes) == 1
    assert main_nodes[0].name.startswith("main-unified-v1-")
    written = json.loads(
        (main_nodes[0] / "main_heads.json").read_text(encoding="utf-8")
    )
    assert written["params"]["beta"] == 0.3
    assert written["main_heads"] == [[15, 2]]
    # re-running with identical params reuses the node (still one main-* dir)
    assert step1_cli.main([*base, "--selector", "unified", "--param", "beta=0.3"]) == 0
    assert len(list(tmp_path.glob("main-*"))) == 1
    # a different parameterization lands in a SECOND sibling main-* node dir
    assert step1_cli.main([*base, "--selector", "unified"]) == 0
    assert len(list(tmp_path.glob("main-*"))) == 2
    # unknown parameter names are rejected before any selector runs
    rc = step1_cli.main([*base, "--selector", "unified", "--param", "nope=1"])
    assert rc == 2
    assert "unknown parameter" in capsys.readouterr().err
    # malformed --param
    rc = step1_cli.main([*base, "--selector", "unified", "--param", "beta"])
    assert rc == 2
    assert "NAME=VALUE" in capsys.readouterr().err
    # --pin-heads only valid for the pin selector
    rc = step1_cli.main([*base, "--selector", "unified", "--pin-heads", "15:2,13:6"])
    assert rc == 2
    assert "pin" in capsys.readouterr().err


def test_step1_scan_validates_significant_artifact(fake_repo, tmp_path, capsys):
    from subspaces.artifacts import make_manifest, write_json_atomic
    from subspaces.paths import ProjectPaths

    wrong_kind = make_manifest(
        kind="heads",
        schema_version=1,
        paths=ProjectPaths.from_root(fake_repo),
        payload={"main_heads": []},
    )
    wrong_path = tmp_path / "not_significant.json"
    write_json_atomic(wrong_path, wrong_kind)

    config = str(fake_repo / "configs" / "step1_test.yaml")
    rc = step1_cli.main(
        [
            "--root",
            str(fake_repo),
            "scan",
            "--config",
            config,
            "--significant",
            str(wrong_path),
        ]
    )
    assert rc == 2
    assert "expected kind" in capsys.readouterr().err


def test_scan_refuses_foreign_significant_artifact(fake_repo, tmp_path, capsys):
    """--significant must descend from THIS run's matrix (audit finding)."""
    from subspaces.artifacts import make_manifest, write_json_atomic
    from subspaces.paths import ProjectPaths

    foreign = make_manifest(
        kind="significant_heads",
        schema_version=1,
        paths=ProjectPaths.from_root(fake_repo),
        inputs={
            "matrix_ref": {
                "path": "elsewhere/matrix_ref.json",
                "semantic_fingerprint": "e" * 64,
            }
        },
        payload={
            "impl": {"module": "subspaces.step1.significant", "algorithm_version": 1},
            "heads": [[0, 0, 0.9]],
            "model_dims": {"n_layers": 4, "n_heads": 4},
        },
    )
    foreign_path = tmp_path / "foreign_significant.json"
    write_json_atomic(foreign_path, foreign)

    config = str(fake_repo / "configs" / "step1_test.yaml")
    rc = step1_cli.main(
        [
            "--root",
            str(fake_repo),
            "scan",
            "--config",
            config,
            "--significant",
            str(foreign_path),
        ]
    )
    assert rc == 2
    assert "foreign head list" in capsys.readouterr().err


def test_compose_cli_validates_lineage(fake_repo, tmp_path, capsys):
    """The compose subcommand exists and enforces the descent chain."""
    from subspaces.artifacts import make_manifest, manifest_ref, write_json_atomic
    from subspaces.paths import ProjectPaths

    paths = ProjectPaths.from_root(fake_repo)
    significant = make_manifest(
        kind="significant_heads",
        schema_version=1,
        paths=paths,
        payload={
            "impl": {"module": "subspaces.step1.significant", "algorithm_version": 1},
            "heads": [[1, 1, 0.9], [2, 3, 0.8]],
            "model_dims": {"n_layers": 4, "n_heads": 4},
        },
    )
    significant_path = tmp_path / "significant_heads.json"
    write_json_atomic(significant_path, significant)
    scan = make_manifest(
        kind="head_scan",
        schema_version=1,
        paths=paths,
        inputs={
            "significant_heads": manifest_ref(significant_path, paths, significant)
        },
        payload={"curves": {}, "c_grid": [0, 1]},
    )
    scan_path = tmp_path / "head_scan.json"
    write_json_atomic(scan_path, scan)
    main = make_manifest(
        kind="main_heads",
        schema_version=1,
        paths=paths,
        inputs={"head_scan": manifest_ref(scan_path, paths, scan)},
        payload={
            "impl": {"module": "subspaces.step1.selectors", "algorithm_version": 1},
            "selector_name": "unified",
            "selector_version": 1,
            "params": {},
            "main_heads": [[1, 1]],
            "minor_heads": [],
            "decisions": {},
            "verdict": {},
        },
    )
    main_path = tmp_path / "main_heads" / "main.json"
    write_json_atomic(main_path, main)

    rc = step1_cli.main(
        [
            "--root",
            str(fake_repo),
            "compose",
            "--significant",
            str(significant_path),
            "--main",
            str(main_path),
            "--out-dir",
            str(tmp_path / "heads"),
        ]
    )
    assert rc == 0
    # single-slot heads artifact (identity is fixed by the sig + main pair)
    assert (tmp_path / "heads" / "heads.json").is_file()

    # foreign significant refuses
    other = dict(significant)
    other["heads"] = [[3, 3, 0.7]]
    other_path = tmp_path / "other_significant.json"
    write_json_atomic(other_path, other)
    rc = step1_cli.main(
        [
            "--root",
            str(fake_repo),
            "compose",
            "--significant",
            str(other_path),
            "--main",
            str(main_path),
            "--out-dir",
            str(tmp_path / "heads2"),
        ]
    )
    assert rc == 2
    assert "lineage mismatch" in capsys.readouterr().err


def test_selector_registry_unknown_message(capsys):
    # implemented-selector behavior is covered in tests/test_selectors.py
    try:
        get_selector("nope", 1)
        raise AssertionError("expected SelectorError")
    except SelectorError as err:
        assert "unified v1" in str(err)
