"""End-to-end Step-1 orchestration with GPU primitives stubbed (deterministic
fakes): composed run completeness, sequential == composed equivalence,
selector-change reuse, and runtime holdout ordering.

Artifacts live in the lineage tree (``subspaces.step1.tree``):
``log/runs/<node>__<key12>/sig-*/scan-*/main-*/`` with per-invocation config
records in ``log/journal/<run>__<id12>/`` and content-addressed sample
manifests in ``log/cache/samples/<fp16>/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from conftest import (  # noqa: F401 — gpu_stubs fixture lives in conftest
    _fake_compute_z,
    _fake_eval_prompts,
    build_fake_repo,
)

from subspaces.artifacts import semantic_fingerprint
from subspaces.config import load_step1_config
from subspaces.paths import ProjectPaths
from subspaces.step1 import pipeline, tree


def _cfg(root):
    return load_step1_config(root / "configs" / "step1_test.yaml")


def _runs_tree_fps(root) -> dict:
    """Semantic fingerprints of every lineage-tree artifact, keyed by the
    path relative to log/runs — pins content identity AND node placement."""
    runs_dir = root / "log" / "runs"
    return {
        str(path.relative_to(runs_dir)): semantic_fingerprint(
            json.loads(path.read_text(encoding="utf-8"))
        )
        for path in sorted(runs_dir.rglob("*.json"))
    }


def test_scan_head_limit_bounds_scanned_heads():
    from subspaces.step1.recovery import scanned_heads

    significant = {"heads": [[1, 1, 0.9], [2, 3, 0.8], [0, 0, 0.1]]}
    assert scanned_heads(significant, None) == [(1, 1), (2, 3), (0, 0)]
    assert scanned_heads(significant, 2) == [(1, 1), (2, 3)]


def test_raw_coef_vector_uses_the_full_head_grid():
    import torch

    from subspaces.step1.headset_eval import all_head_raw_coef_vector

    matrix = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    z = torch.arange(1, 13, dtype=torch.float32).reshape(2, 2, 3)
    expected = sum(
        matrix[layer_idx, head_idx] * z[layer_idx, head_idx]
        for layer_idx in range(2)
        for head_idx in range(2)
    )
    torch.testing.assert_close(all_head_raw_coef_vector(matrix, z), expected)


def test_composed_run_completes_with_all_artifacts(fake_repo, gpu_stubs):
    paths = ProjectPaths.from_root(fake_repo)
    cfg = _cfg(fake_repo)
    summary = pipeline.run(cfg, paths)

    journal_dir = paths.journal_dir(summary["run_id"])
    assert (journal_dir / "requested_config.yaml").is_file()
    assert (journal_dir / "nodes.json").is_file()

    matrix_node = paths.resolve(summary["nodes"]["matrix"])
    sig_node = paths.resolve(summary["nodes"]["significant"])
    scan_node = paths.resolve(summary["nodes"]["scan"])
    main_node = paths.resolve(summary["latest_selection"]["main_node"])
    # lineage-tree placement: matrix node > sig-* > scan-* > main-*
    assert matrix_node.parent == paths.runs_dir
    assert sig_node == tree.sig_node_dir(matrix_node, cfg.significant)
    assert scan_node.parent == sig_node
    assert scan_node.name.startswith(
        f"scan-{cfg.samples.scan.examples_per_task}x{cfg.scan.c_max}-"
    )
    assert main_node.parent == scan_node
    assert main_node.name.startswith("main-")

    assert (matrix_node / "matrix_ref.json").is_file()
    assert (sig_node / "significant_heads.json").is_file()
    assert (scan_node / "head_scan.json").is_file()
    assert (scan_node / "scan_outcomes.npz").is_file()
    assert (main_node / "main_heads.json").is_file()
    assert list(main_node.glob("eval-*-outcomes.npz"))
    assert (main_node / "heads.json").is_file()
    assert set(summary["metrics"]) == {"clean", "full_sig", "raw_coef", "main_meanab"}

    scan = json.loads((scan_node / "head_scan.json").read_text(encoding="utf-8"))
    c_grid = list(range(cfg.scan.c_min, cfg.scan.c_max + 1))
    assert scan["c_grid"] == c_grid
    for curve in scan["curves"].values():
        assert sorted(int(c) for c in curve) == c_grid
    # denominators: every listed example evaluated
    expected_n = sum(scan["counts_per_task"].values())
    assert scan["n_eval_examples_per_head_per_c"] == expected_n

    # GPU-memory observability recorded (null on CPU), outside identity keys
    assert "gpu_memory" in scan
    headset_path = next(main_node.glob("eval-*.json"))
    headset = json.loads(headset_path.read_text(encoding="utf-8"))
    assert "gpu_memory" in headset
    assert headset["config"]["raw_coef"] == {
        "coefficient_source": "final_checkpoint",
        "head_scope": "all",
    }
    assert headset["inputs"]["raw_coef_checkpoint"]["epoch"] == 10
    assert headset["n_raw_coef_heads"] == 16

    import numpy as np

    outcomes = np.load(scan_node / "scan_outcomes.npz")
    assert len(outcomes["clean"]) == expected_n
    for key in outcomes.files:
        assert len(outcomes[key]) == expected_n


def test_raw_coef_sources_final_checkpoint_not_selection_substrate(
    tmp_path, gpu_stubs, monkeypatch
):
    """Composed run on a mean_last_k node: the tensor actually fed to the
    raw_coef math must be the SOURCE directory's final checkpoint — not the
    derived selection mean — and the input reference must pin that file by
    content. A revert to selection-substrate coefficients fails here even if
    every identity field is kept intact."""
    import torch

    from subspaces.artifacts import sha256_file
    from subspaces.config import CheckpointSelectConfig
    from subspaces.step1 import headset_eval

    repo = build_fake_repo(tmp_path / "meansub")
    # the fixture ships epochs (0, 2, 10); add 8..9 so the trailing 3 are
    # contiguous and their mean differs from the final checkpoint
    ckpt_dir = repo / "matrices" / "toy" / "checkpoints"
    for epoch in (8, 9):
        matrix = torch.zeros(4, 4)
        matrix[1, 1] = 0.9
        matrix[2, 3] = 0.8
        matrix[0, 0] = 0.001 * (epoch + 1)
        torch.save(matrix, ckpt_dir / f"matrix_epoch{epoch}.pth")
    cfg = _cfg(repo)
    cfg.matrix.checkpoint_select = CheckpointSelectConfig(rule="mean_last_k", k=3)

    seen: list = []
    real_vector = headset_eval.all_head_raw_coef_vector

    def spy(matrix, z):
        seen.append(matrix.clone())
        return real_vector(matrix, z)

    monkeypatch.setattr(headset_eval, "all_head_raw_coef_vector", spy)

    paths = ProjectPaths.from_root(repo)
    summary = pipeline.run(cfg, paths)

    final_path = ckpt_dir / "matrix_epoch10.pth"
    final = torch.load(final_path)
    assert seen  # one call per held-out task
    for used in seen:
        assert torch.equal(used, final)

    matrix_node = paths.resolve(summary["nodes"]["matrix"])
    matrix_ref = json.loads(
        (matrix_node / "matrix_ref.json").read_text(encoding="utf-8")
    )
    derived = torch.load(
        paths.resolve(matrix_ref["matrix_dir"])
        / "checkpoints"
        / matrix_ref["selected_checkpoint"]["file"]
    )
    assert not torch.equal(derived, final)  # the substrate is a real decoy

    main_node = paths.resolve(summary["latest_selection"]["main_node"])
    headset = json.loads(
        next(main_node.glob("eval-*.json")).read_text(encoding="utf-8")
    )
    ref = headset["inputs"]["raw_coef_checkpoint"]
    assert ref["epoch"] == 10
    assert ref["selection_rule"] == "final_checkpoint"
    assert ref["sha256"] == sha256_file(final_path)


def test_sequential_subcommands_equal_composed_run(tmp_path, gpu_stubs):
    from subspaces.runners import step1 as step1_cli

    repo_a = build_fake_repo(tmp_path / "composed")
    pipeline.run(_cfg(repo_a), ProjectPaths.from_root(repo_a))
    fps_composed = _runs_tree_fps(repo_a)

    repo_b = build_fake_repo(tmp_path / "sequential")
    config = str(repo_b / "configs" / "step1_test.yaml")
    root = ["--root", str(repo_b)]
    assert step1_cli.main([*root, "select-significant", "--config", config]) == 0
    assert step1_cli.main([*root, "make-samples", "--config", config]) == 0
    # explicit artifact handoff: scan consumes the persisted significant
    significant_path = next(
        (repo_b / "log" / "runs").glob("*/sig-*/significant_heads.json")
    )
    assert (
        step1_cli.main(
            [
                *root,
                "scan",
                "--config",
                config,
                "--significant",
                str(significant_path),
            ]
        )
        == 0
    )
    scan_path = next((repo_b / "log" / "runs").glob("*/sig-*/scan-*/head_scan.json"))
    assert (
        step1_cli.main(
            [*root, "select-main", "--scan", str(scan_path), "--selector", "unified"]
        )
        == 0
    )
    main_path = next(scan_path.parent.glob("main-*/main_heads.json"))
    assert (
        step1_cli.main(
            [
                *root,
                "evaluate-headset",
                "--config",
                config,
                "--main",
                str(main_path),
            ]
        )
        == 0
    )
    # compose the heads artifact exactly as the composed run does (the CLI
    # default out-dir is the main node)
    assert (
        step1_cli.main(
            [
                *root,
                "compose",
                "--significant",
                str(significant_path),
                "--main",
                str(main_path),
            ]
        )
        == 0
    )

    fps_sequential = _runs_tree_fps(repo_b)
    assert fps_sequential == fps_composed
    names = {Path(rel).name for rel in fps_composed}
    assert {
        "matrix_ref.json",
        "significant_heads.json",
        "head_scan.json",
        "main_heads.json",
        "heads.json",
    } <= names
    assert any(name.startswith("eval-") for name in names)


def test_selector_change_composed_run_reuses_scan(fake_repo, gpu_stubs, monkeypatch):
    """PLAN invariant: changing ONLY the selector reuses matrix, significant,
    caches, and scan — through the COMPOSED flow (same journal, same scan
    node), with the GPU primitives booby-trapped on the second run."""
    paths = ProjectPaths.from_root(fake_repo)
    cfg = _cfg(fake_repo)
    summary = pipeline.run(cfg, paths)
    scan_node = paths.resolve(summary["nodes"]["scan"])
    scan_before = (scan_node / "head_scan.json").read_bytes()

    import subspaces.step1.eval_gpu as eval_gpu
    import subspaces.step1.recovery as recovery_mod

    def _boom(*_a, **_k):
        raise AssertionError("GPU work must be reused, not recomputed")

    monkeypatch.setattr(recovery_mod, "run_scan", _boom)
    monkeypatch.setattr(eval_gpu, "compute_z_for_tasks", _boom)

    cfg2 = _cfg(fake_repo)
    cfg2.main_selector.name = "recovery_weak"
    cfg2.main_selector.params = {"rel_floor": 0.2, "abs_floor": 0.01, "eps": 0.01}
    summary2 = pipeline.run(cfg2, paths)  # headset eval still runs (stubbed)

    assert summary2["run_id"] == summary["run_id"]  # selector not in identity
    assert summary2["nodes"]["scan"] == summary["nodes"]["scan"]
    assert (scan_node / "head_scan.json").read_bytes() == scan_before
    # selector variants are SIBLING main-* node dirs under the shared scan
    assert len(list(scan_node.glob("main-*"))) == 2
    assert len(list(scan_node.glob("main-*/eval-*.json"))) == 2
    assert len(list(scan_node.glob("main-*/eval-*-outcomes.npz"))) == 2
    nodes_doc = json.loads(
        (paths.journal_dir(summary["run_id"]) / "nodes.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(nodes_doc["selections"]) == 2  # both selections accumulated


def test_second_identical_run_reuses_scan_and_headset(
    fake_repo, gpu_stubs, monkeypatch
):
    paths = ProjectPaths.from_root(fake_repo)
    cfg = _cfg(fake_repo)
    pipeline.run(cfg, paths)

    import subspaces.step1.eval_gpu as eval_gpu
    import subspaces.step1.recovery as recovery_mod

    def _boom(*_a, **_k):
        raise AssertionError("everything must be reused on an identical rerun")

    monkeypatch.setattr(recovery_mod, "run_scan", _boom)
    monkeypatch.setattr(eval_gpu, "compute_z_for_tasks", _boom)
    monkeypatch.setattr(eval_gpu, "eval_prompts", _boom)
    summary = pipeline.run(cfg, paths)
    assert summary["metrics"]


def test_scan_outcomes_integrity_enforced_on_reuse(fake_repo, gpu_stubs):
    import numpy as np

    from subspaces.artifacts import ArtifactError

    paths = ProjectPaths.from_root(fake_repo)
    cfg = _cfg(fake_repo)
    summary = pipeline.run(cfg, paths)
    scan_node = paths.resolve(summary["nodes"]["scan"])
    npz_path = scan_node / "scan_outcomes.npz"

    # tampered outcome arrays -> digest mismatch refusal
    loaded = {key: np.load(npz_path)[key] for key in np.load(npz_path).files}
    loaded["clean"] = 1 - loaded["clean"]
    np.savez_compressed(npz_path, **loaded)
    with pytest.raises(ArtifactError, match="digest mismatch"):
        pipeline.run(cfg, paths)

    # missing npz -> refusal
    npz_path.unlink()
    with pytest.raises(ArtifactError, match="missing"):
        pipeline.run(cfg, paths)


def test_zcache_round_trip_and_identity_refusal(fake_repo, gpu_stubs):
    import torch

    from subspaces.artifacts import ArtifactError
    from subspaces.step1 import zcache
    from subspaces.step1.pipeline import ensure_run

    paths = ProjectPaths.from_root(fake_repo)
    cfg = _cfg(fake_repo)
    pipeline.run(cfg, paths)
    _, split, resolved = ensure_run(cfg, paths)
    # content-addressed sample manifest (log/cache/samples/<fp16>/), reused
    activation, activation_path = pipeline.stage_selection_samples(cfg, paths, split)[
        "activation"
    ]
    assert activation_path.is_relative_to(paths.samples_cache_dir)
    identity = zcache.cache_identity(cfg, resolved, activation)

    cached = zcache.load_zcache(paths, identity)
    assert cached is not None
    regenerated = _fake_compute_z(None, activation, list(activation["task_order"]))
    for task, tensor in regenerated.items():
        assert torch.equal(cached[task], tensor)  # lossless round-trip

    # tampered meta identity under the same fingerprint dir -> refusal
    meta_path = (
        zcache.cache_dir(paths, zcache.cache_fingerprint(identity)) / "meta.json"
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["identity"]["dataset_fingerprint"] = "0" * 64
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ArtifactError, match="identity does not match"):
        zcache.load_zcache(paths, identity)


def test_outcome_rows_align_with_manifest_prompts(fake_repo, gpu_stubs):
    """Row i of every outcomes array corresponds to the i-th prompt of the
    task_order concatenation — clean rows from n-shot prompts, intervention
    rows from the paired zero-shot prompts (recomputed via the deterministic
    fake)."""
    import numpy as np

    paths = ProjectPaths.from_root(fake_repo)
    cfg = _cfg(fake_repo)
    summary = pipeline.run(cfg, paths)
    scan_node = paths.resolve(summary["nodes"]["scan"])
    sig_node = paths.resolve(summary["nodes"]["significant"])
    scan = json.loads((scan_node / "head_scan.json").read_text(encoding="utf-8"))
    _, split, _ = pipeline.ensure_run(cfg, paths)
    samples = pipeline.stage_selection_samples(cfg, paths, split)["scan"][0]
    outcomes = np.load(scan_node / "scan_outcomes.npz")

    def expected_rows(prompt_key, signature):
        rows = []
        for task in scan["task_order"]:
            for sample in samples["tasks"][task]["samples"]:
                digest = hashlib.sha256(
                    f"{sample[prompt_key]}|{signature}".encode()
                ).digest()
                rows.append(1 if digest[0] < 192 else 0)
        return rows

    assert list(outcomes["clean"]) == expected_rows("prompt", "clean")

    # one curve key: rebuild the fake vector signature for (head, c)
    z = _fake_compute_z(None, samples, list(scan["task_order"]))
    import torch

    sig_heads = [
        (h[0], h[1])
        for h in json.loads(
            (sig_node / "significant_heads.json").read_text(encoding="utf-8")
        )["heads"]
    ]
    mean_z = torch.stack([z[t] for t in scan["task_order"]]).mean(0)
    mean_fv = sum(mean_z[layer, head] for layer, head in sig_heads)
    layer, head = scan["scanned_heads"][0]
    c = 1
    rows = []
    for task in scan["task_order"]:
        vector = c * z[task][layer, head] + (mean_fv - mean_z[layer, head])
        signature = f"{float(vector.sum()):.4f}"
        for sample in samples["tasks"][task]["samples"]:
            digest = hashlib.sha256(
                f"{sample['zero_shot_prompt']}|{signature}".encode()
            ).digest()
            rows.append(1 if digest[0] < 192 else 0)
    assert list(outcomes[f"{layer}:{head}:{c}"]) == rows


def _paired_scan_node(fake_repo, paths):
    """A tree-layout scan node with a digest-consistent outcomes npz: one
    planted head (1,1) whose c=1 vector fixes 10 examples and breaks none."""
    import numpy as np

    from subspaces.artifacts import make_manifest, write_json_atomic
    from subspaces.step1.recovery import outcomes_content_sha

    y0 = np.zeros(20, dtype=np.uint8)
    y0[:2] = 1
    y1 = np.zeros(20, dtype=np.uint8)
    y1[:12] = 1
    arrays = {
        "clean": np.ones(20, dtype=np.uint8),
        "1:1:0": y0,
        "1:1:1": y1,
    }
    scan_dir = (
        fake_repo
        / "log"
        / "runs"
        / "node__abc123def456"
        / "sig-elbow-v1"
        / "scan-5x1-aaaaaa"
    )
    scan_dir.mkdir(parents=True)
    np.savez_compressed(scan_dir / "scan_outcomes.npz", **arrays)
    scan = make_manifest(
        kind="head_scan",
        schema_version=1,
        paths=paths,
        payload={
            "curves": {"1:1": {"0": float(y0.mean()), "1": float(y1.mean())}},
            "baselines": {"clean_acc": 1.0, "full_significant_acc": 0.6},
            "n_eval_examples_per_head_per_c": 20,
            "c_grid": [0, 1],
            "outcomes": {
                "file": "scan_outcomes.npz",
                "content_sha256": outcomes_content_sha(arrays),
            },
        },
    )
    write_json_atomic(scan_dir / "head_scan.json", scan)
    return scan_dir, arrays


def test_apply_selector_paired_family_loads_verified_outcomes(fake_repo):
    """The paired selectors receive the digest-verified npz vectors through
    apply_selector; curve-only selectors on the same scan never need it."""
    paths = ProjectPaths.from_root(fake_repo)
    scan_dir, _ = _paired_scan_node(fake_repo, paths)
    manifest, node = pipeline.apply_selector(
        scan_dir / "head_scan.json",
        selector_name="paired_bh",
        selector_version=1,
        params={"q": 0.05},
        paths=paths,
    )
    assert node.name.startswith("recpos-paired_bh-v1-")
    assert (node / "main_heads.json").is_file()
    assert manifest["main_heads"] == [[1, 1]]
    assert manifest["params"] == {"q": 0.05}
    decision = manifest["decisions"]["1:1"]
    assert decision["b"] == 10 and decision["d"] == 0
    assert decision["p_raw_min"] == 2.0**-10
    assert decision["per_coef"]["1"]["n_discordant"] == 10
    checks = manifest["verdict"]["outcomes_check"]
    assert checks["n_vectors_checked"] == 2
    # the verdict quotes the npz content digest recorded in the scan manifest
    scan_manifest = json.loads(
        (scan_dir / "head_scan.json").read_text(encoding="utf-8")
    )
    assert checks["content_sha256"] == scan_manifest["outcomes"]["content_sha256"]
    # a different q is a sibling node (params enter the node identity)
    _, node_q10 = pipeline.apply_selector(
        scan_dir / "head_scan.json",
        selector_name="paired_bh",
        selector_version=1,
        params={"q": 0.1},
        paths=paths,
    )
    assert node_q10 != node and node_q10.parent == node.parent


def test_apply_selector_paired_family_outcome_refusals(fake_repo):
    import numpy as np

    from subspaces.artifacts import ArtifactError, make_manifest, write_json_atomic

    paths = ProjectPaths.from_root(fake_repo)
    scan_dir, arrays = _paired_scan_node(fake_repo, paths)
    scan_path = scan_dir / "head_scan.json"

    def apply(selector="paired_per_head", params=None):
        return pipeline.apply_selector(
            scan_path,
            selector_name=selector,
            selector_version=1,
            params={"alpha": 0.05} if params is None else params,
            paths=paths,
        )

    # tampered npz (digest mismatch)
    tampered = dict(arrays)
    tampered["1:1:1"] = arrays["1:1:0"]
    np.savez_compressed(scan_dir / "scan_outcomes.npz", **tampered)
    with pytest.raises(ArtifactError, match="digest mismatch"):
        apply()
    # missing npz
    (scan_dir / "scan_outcomes.npz").unlink()
    with pytest.raises(ArtifactError, match="missing but the manifest references"):
        apply()
    # curve-only selectors still work without the npz
    _, pin_node = apply(selector="pin", params={"heads": [[1, 1]]})
    assert (pin_node / "main_heads.json").is_file()
    # a scan manifest without an outcomes reference cannot serve the family
    bare_dir = (
        fake_repo / "log" / "runs" / "node__abc123def456" / "sig-elbow-v1" / "scan-bare"
    )
    bare_dir.mkdir(parents=True)
    bare = make_manifest(
        kind="head_scan",
        schema_version=1,
        paths=paths,
        payload={
            "curves": {"1:1": {"0": 0.1, "1": 0.6}},
            "baselines": {"clean_acc": 1.0, "full_significant_acc": 0.6},
            "n_eval_examples_per_head_per_c": 20,
            "c_grid": [0, 1],
        },
    )
    write_json_atomic(bare_dir / "head_scan.json", bare)
    with pytest.raises(ArtifactError, match="records no per-example outcomes"):
        pipeline.apply_selector(
            bare_dir / "head_scan.json",
            selector_name="paired_bh",
            selector_version=1,
            params={"q": 0.05},
            paths=paths,
        )


def test_select_main_cli_paired_selector(fake_repo):
    """CLI path: preset q, typed --param override, unknown-param refusal."""
    from subspaces.runners import step1 as step1_cli

    paths = ProjectPaths.from_root(fake_repo)
    scan_dir, _ = _paired_scan_node(fake_repo, paths)
    base = [
        "--root",
        str(fake_repo),
        "select-main",
        "--scan",
        str(scan_dir / "head_scan.json"),
        "--selector",
        "paired_bh",
    ]
    assert step1_cli.main(base) == 0  # preset q=0.05
    assert step1_cli.main([*base, "--param", "q=0.01"]) == 0
    assert step1_cli.main([*base, "--param", "alpha=0.05"]) == 2  # unknown name
    nodes = sorted(p.name for p in scan_dir.glob("recpos-paired_bh-v1-*"))
    assert len(nodes) == 2  # q=0.05 and q=0.01 siblings


def test_select_main_refuses_default_write_into_frozen_flat_run(fake_repo):
    """Pre-tree (flat) runs keep their scan directly in <run>/step1/; the
    DEFAULT parent (the scan's directory) must refuse to write into them —
    an explicit --out-dir redirects the main-* node elsewhere."""
    from subspaces.artifacts import ArtifactError, make_manifest, write_json_atomic
    from subspaces.runners import step1 as step1_cli

    paths = ProjectPaths.from_root(fake_repo)
    scan = make_manifest(
        kind="head_scan",
        schema_version=1,
        paths=paths,
        payload={"curves": {"1:1": {"0": 0.5, "1": 0.7}}},
    )
    flat_scan = (
        fake_repo / "log" / "runs" / "frozen__abc123def456" / "step1" / "head_scan.json"
    )
    write_json_atomic(flat_scan, scan)

    with pytest.raises(ArtifactError, match="frozen pre-tree"):
        pipeline.apply_selector(
            flat_scan,
            selector_name="pin",
            selector_version=1,
            params={"heads": [[1, 1]]},
            paths=paths,
        )
    cli_base = [
        "--root",
        str(fake_repo),
        "select-main",
        "--scan",
        str(flat_scan),
        "--selector",
        "pin",
        "--pin-heads",
        "1:1",
    ]
    assert step1_cli.main(cli_base) == 2  # clean CLI refusal, same guard
    assert not list(flat_scan.parent.glob("main-*"))  # nothing written

    out_dir = fake_repo / "log" / "explicit-out"
    assert step1_cli.main([*cli_base, "--out-dir", str(out_dir)]) == 0
    assert list(out_dir.glob("main-pin-v1-*/main_heads.json"))


def test_compose_refuses_default_out_dir_inside_frozen_run(fake_repo, capsys):
    """Analogous guard for compose: a defaulted out-dir pointing inside a
    frozen flat run (main_heads/ grouping dir or step1/) refuses before
    reading any manifest."""
    from subspaces.runners import step1 as step1_cli

    for parent_name in ("main_heads", "step1"):
        main_path = (
            fake_repo / "log" / "runs" / "frozen__abc" / parent_name / "main_heads.json"
        )
        main_path.parent.mkdir(parents=True, exist_ok=True)
        main_path.write_text("{}", encoding="utf-8")
        rc = step1_cli.main(
            [
                "--root",
                str(fake_repo),
                "compose",
                "--significant",
                str(fake_repo / "nowhere.json"),
                "--main",
                str(main_path),
            ]
        )
        assert rc == 2
        assert "frozen pre-tree" in capsys.readouterr().err
        assert not (main_path.parent / "heads.json").exists()


def test_headset_refuses_foreign_main(fake_repo, gpu_stubs):
    from subspaces.artifacts import ArtifactError, make_manifest, write_json_atomic

    paths = ProjectPaths.from_root(fake_repo)
    cfg = _cfg(fake_repo)
    summary = pipeline.run(cfg, paths)
    scan_node = paths.resolve(summary["nodes"]["scan"])

    foreign = make_manifest(
        kind="main_heads",
        schema_version=1,
        paths=paths,
        inputs={
            "head_scan": {
                "path": "elsewhere/head_scan.json",
                "semantic_fingerprint": "f" * 64,
            }
        },
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
    foreign_node = fake_repo / "log" / "elsewhere" / "main-foreign"
    write_json_atomic(foreign_node / "main_heads.json", foreign)

    journal_dir, split, resolved = pipeline.ensure_run(cfg, paths)
    matrix_ref, matrix_node = pipeline.stage_matrix(cfg, paths, journal_dir)
    significant, sig_node = pipeline.stage_significant(
        cfg, paths, matrix_node, matrix_ref
    )
    samples = pipeline.stage_selection_samples(cfg, paths, split)
    with pytest.raises(ArtifactError, match="different scan"):
        pipeline.stage_evaluate_headset(
            cfg,
            paths,
            split,
            matrix_node,
            sig_node,
            scan_node,
            significant,
            matrix_ref,
            foreign,
            foreign_node,
            samples["activation"],
            samples["scan"],
            resolved,
            pipeline.make_model_loader(cfg, resolved),
        )


def test_headset_refuses_foreign_or_stale_scan_on_disk(fake_repo, gpu_stubs):
    """Audit residual 7b/7c closed: a scan swapped in from another run (or
    hand-edited) is refused by identity, not trusted from disk."""
    from subspaces.artifacts import ArtifactError

    paths = ProjectPaths.from_root(fake_repo)
    cfg = _cfg(fake_repo)
    summary = pipeline.run(cfg, paths)
    scan_node = paths.resolve(summary["nodes"]["scan"])
    scan_path = scan_node / "head_scan.json"

    tampered = json.loads(scan_path.read_text(encoding="utf-8"))
    tampered["config"]["scan"]["c_max"] = tampered["config"]["scan"]["c_max"] - 1
    scan_path.write_text(json.dumps(tampered), encoding="utf-8")

    journal_dir, split, resolved = pipeline.ensure_run(cfg, paths)
    matrix_ref, matrix_node = pipeline.stage_matrix(cfg, paths, journal_dir)
    significant, sig_node = pipeline.stage_significant(
        cfg, paths, matrix_node, matrix_ref
    )
    samples = pipeline.stage_selection_samples(cfg, paths, split)
    main_node = paths.resolve(summary["latest_selection"]["main_node"])
    main_manifest = json.loads(
        (main_node / "main_heads.json").read_text(encoding="utf-8")
    )
    with pytest.raises(ArtifactError, match="scan identity mismatch"):
        pipeline.stage_evaluate_headset(
            cfg,
            paths,
            split,
            matrix_node,
            sig_node,
            scan_node,
            significant,
            matrix_ref,
            main_manifest,
            main_node,
            samples["activation"],
            samples["scan"],
            resolved,
            pipeline.make_model_loader(cfg, resolved),
        )


def test_headset_outcomes_integrity_enforced_on_reuse(fake_repo, gpu_stubs):
    import numpy as np

    from subspaces.artifacts import ArtifactError

    paths = ProjectPaths.from_root(fake_repo)
    cfg = _cfg(fake_repo)
    pipeline.run(cfg, paths)
    summary = pipeline.run(cfg, paths)  # reuse path exercises the check
    main_node = paths.resolve(summary["latest_selection"]["main_node"])
    npz_path = next(main_node.glob("eval-*-outcomes.npz"))

    loaded = {key: np.load(npz_path)[key] for key in np.load(npz_path).files}
    loaded["main_meanab"] = 1 - loaded["main_meanab"]
    np.savez_compressed(npz_path, **loaded)
    with pytest.raises(ArtifactError, match="digest mismatch"):
        pipeline.run(cfg, paths)
    npz_path.unlink()
    with pytest.raises(ArtifactError, match="missing"):
        pipeline.run(cfg, paths)


def test_heldout_manifests_created_only_by_headset_stage(fake_repo, gpu_stubs):
    paths = ProjectPaths.from_root(fake_repo)
    cfg = _cfg(fake_repo)

    import subspaces.step1.samples as samples_mod

    original = samples_mod.generate_samples
    kinds_seen: list[str] = []

    def recorder(cfg_, spec, kind, split_, paths_, out_dir):
        kinds_seen.append(kind)
        return original(cfg_, spec, kind, split_, paths_, out_dir)

    samples_mod.generate_samples = recorder
    try:
        pipeline.run(cfg, paths)
    finally:
        samples_mod.generate_samples = original

    heldout_positions = [
        i for i, k in enumerate(kinds_seen) if k in ("heldout_activation", "final_eval")
    ]
    selection_positions = [
        i for i, k in enumerate(kinds_seen) if k in ("activation", "scan")
    ]
    assert selection_positions and heldout_positions
    assert max(selection_positions) < min(heldout_positions)


def test_zcache_hit_avoids_model_load(fake_repo, gpu_stubs, monkeypatch):
    paths = ProjectPaths.from_root(fake_repo)
    cfg = _cfg(fake_repo)
    pipeline.run(cfg, paths)

    import subspaces.step1.eval_gpu as eval_gpu

    monkeypatch.setattr(
        eval_gpu,
        "compute_z_for_tasks",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache must hit")),
    )
    # a second composed run reuses every artifact (and hits the z caches)
    summary = pipeline.run(cfg, paths)
    assert summary["metrics"]
