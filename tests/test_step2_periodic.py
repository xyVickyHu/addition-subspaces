"""Step-2 periodic plugin: §4 mod-vector fit math, add-only refusals,
artifact/reuse contract, GT comparison, and the real-repo anchor.

The math under test is the faithful legacy port (subspaces/step2/periodic.py):
per-head PCA -> top-n_pcs subspace, phase-searched least-squares fit of the
six periodic patterns of the add-k task parameter (mod50/mod2/mod5/mod10c/
mod10s/mod25), mod_vectors = pc_subspace @ weights, float64 throughout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from conftest import tree_snapshot

from subspaces.artifacts import ArtifactError, make_manifest, write_json_atomic
from subspaces.paths import ProjectPaths
from subspaces.runners import step2 as step2_cli
from subspaces.step1.zcache import load_zcache_dir, store_zcache
from subspaces.step2.periodic import (
    MAGNITUDE_COLS,
    MOD_VECTOR_COLS,
    UNIT_COLS,
    fit_mod_vectors,
    gt_metrics,
    parse_add_tasks,
    subspace_principal_angles,
)

REAL_REPO = Path(__file__).resolve().parents[1]
# 30-task legacy-protocol llama3/number_add z cache (log/ is gitignored, so
# the anchor test only runs where the cache exists).
ANCHOR_CACHE = REAL_REPO / "log" / "cache" / "z" / "96bcb3aad0650c71"
# git-tracked ground-truth §4 artifact (llama trio + (15,17) + (15,28))
GT_PATH = (
    REAL_REPO
    / "artifacts"
    / "matrix_add_0204_clip_lambda0.05"
    / ("mod_vectors_dict.pth")
)
# reference §4 R² values (frozen from the legacy reference metrics; column
# order mod50, mod2, mod5, mod10c, mod10s, mod25 — MOD_VECTOR_COLS)
REFERENCE_R2 = {
    (15, 2): (0.7262, 0.9506, 0.8904, 0.8968, 0.9063, 0.7840),
    (15, 1): (0.9207, 0.9220, 0.8295, 0.8529, 0.9195, 0.7659),
    (13, 6): (0.9478, 0.9582, 0.8739, 0.8635, 0.9164, 0.8018),
}


# -- planted synthetic data ------------------------------------------------------


def _zero_mean_phase(omega: float, ks: np.ndarray) -> float:
    """Phase phi such that cos(2*pi*omega*ks + phi) has exactly zero mean over
    ks, so the centered PC coords can reproduce the returned (non-centered)
    opt_shift target with R^2 -> 1."""
    total = np.exp(1j * 2 * np.pi * omega * ks).sum()
    return float((np.pi / 2 - np.angle(total)) % (2 * np.pi))


def _planted_case(d_model: int = 32, seed: int = 5):
    """30 add-k rows X[i] = sum_j a_j * pattern_j(k_i) * v_j with orthonormal
    v_j and the six §4 target patterns at known (recoverable) phases."""
    ks = np.arange(1, 31, dtype=np.float64)
    rng = np.random.default_rng(seed)
    v, _ = np.linalg.qr(rng.standard_normal((d_model, 6)))
    patterns = np.zeros((30, 6))
    patterns[:, 0] = np.cos(2 * np.pi * 0.02 * ks + _zero_mean_phase(0.02, ks))
    patterns[:, 1] = np.cos(2 * np.pi * ks / 2)
    # 30 ks = 6 full periods of 5: zero mean for ANY phase; pick an on-grid one
    patterns[:, 2] = np.cos(2 * np.pi * 0.2 * ks + 0.7)
    patterns[:, 3] = np.cos(2 * np.pi * ks / 10)
    patterns[:, 4] = np.sin(2 * np.pi * ks / 10)
    patterns[:, 5] = np.cos(2 * np.pi * 0.04 * ks + _zero_mean_phase(0.04, ks))
    amplitudes = np.array([3.0, 2.6, 2.2, 1.8, 1.5, 1.2])
    x = (patterns * amplitudes) @ v.T  # (30, d_model)
    return x, ks, v


def _cos(a, b) -> float:
    return float(abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))


# -- math: planted recovery -------------------------------------------------------


def test_fit_recovers_planted_directions_and_patterns():
    x, ks, v = _planted_case()
    fit = fit_mod_vectors(x, ks, n_pcs=6, theta_step=0.01)
    assert fit["pca_cumvar_at_npcs"] == pytest.approx(1.0, abs=1e-9)
    for name in MOD_VECTOR_COLS:
        assert fit["target_r2"][name] > 0.98, name
    for j in range(6):
        assert _cos(fit["mod_vectors"][:, j], v[:, j]) > 0.99, MOD_VECTOR_COLS[j]
    # the fitted 6-D span coincides with the planted span
    max_angle, _ = subspace_principal_angles(fit["mod_vectors"], v)
    assert max_angle < 1.0


def test_fit_invariant_to_task_row_order():
    """The fit is over (row, k) PAIRS: permuting rows together with ks must
    not change the result (the plugin feeds rows sorted by ascending k)."""
    x, ks, _ = _planted_case()
    fit = fit_mod_vectors(x, ks, n_pcs=6)
    perm = np.random.default_rng(0).permutation(len(ks))
    fit_perm = fit_mod_vectors(x[perm], ks[perm], n_pcs=6)
    np.testing.assert_allclose(fit_perm["mod_vectors"], fit["mod_vectors"], atol=1e-8)
    for name in MOD_VECTOR_COLS:
        assert fit_perm["target_r2"][name] == pytest.approx(
            fit["target_r2"][name], abs=1e-9
        )


def test_gt_metrics_on_planted_ground_truth():
    x, ks, v = _planted_case()
    fit = fit_mod_vectors(x, ks, n_pcs=6)
    metrics = gt_metrics(fit["mod_vectors"], v)
    for name in MOD_VECTOR_COLS:
        assert metrics["gt_cos"][name] > 0.99
    assert metrics["gt_angle_6d_max_mean"][0] < 1.0
    assert metrics["gt_angle_unit_max_mean"][0] < 1.0
    assert metrics["gt_angle_magnitude_max_mean"][0] < 1.0


def test_parse_add_tasks_orders_by_ascending_k():
    order, ks = parse_add_tasks(["number-add2", "number-add10", "number-add1"])
    assert order == ["number-add1", "number-add2", "number-add10"]
    assert ks == [1, 2, 10]


def test_parse_add_tasks_refusals():
    with pytest.raises(ArtifactError, match="not an add-k task"):
        parse_add_tasks(["number-add1", "number-mul3"])
    with pytest.raises(ArtifactError, match="not an add-k task"):
        parse_add_tasks(["number-add-3"])
    with pytest.raises(ArtifactError, match="duplicate task parameter k=1"):
        parse_add_tasks(["number-add1", "number-add01"])


# -- fixtures: fabricated caches + head artifacts ---------------------------------


def _identity(samples_fp: str, model_name: str = "test-model") -> dict:
    return {
        "model": {"name": model_name, "revision": "abc", "dtype": "float32"},
        "dataset_fingerprint": "d" * 64,
        "samples": samples_fp,
        "hook_site": "attn.hook_result[last_token]",
        "impl": {"module": "subspaces.step1.zcache", "algorithm_version": 1},
    }


def _fake_z(task_ids, *, n_layers=4, n_heads=4, d_model=32, planted=None):
    """Random per-task tensors; ``planted`` maps task id -> row planted at
    head (1, 1)."""
    z_results = {}
    for task_id in task_ids:
        seed = int.from_bytes(hashlib.sha256(task_id.encode()).digest()[:4], "big")
        generator = torch.Generator().manual_seed(seed)
        tensor = torch.randn(n_layers, n_heads, d_model, generator=generator)
        if planted is not None and task_id in planted:
            tensor[1, 1] = torch.tensor(planted[task_id], dtype=torch.float32)
        z_results[task_id] = tensor
    return z_results


@pytest.fixture()
def periodic_repo(fake_repo):
    """fake_repo + a 30-task add-k z cache (head (1,1) carries the planted §4
    patterns) + a composed heads artifact in a main-* node + a context YAML +
    a planted ground-truth mod_vectors .pth covering (1,1) only."""
    paths = ProjectPaths.from_root(fake_repo)
    x, _ks, v = _planted_case()
    tasks = [f"number-add{k}" for k in range(1, 31)]
    planted = {task: x[i] for i, task in enumerate(tasks)}
    cache_fp = store_zcache(paths, _identity("s-add"), _fake_z(tasks, planted=planted))

    node = fake_repo / "log" / "runs" / "cell__abc" / "main-largest_gap-v1-cafe0000"
    node.mkdir(parents=True)
    heads_manifest = make_manifest(
        kind="heads",
        schema_version=1,
        paths=paths,
        payload={
            "significant_heads": [[1, 1], [2, 3], [0, 0]],
            "main_heads": [[1, 1], [2, 3]],
            "minor_heads": [],
            "model_dims": {"n_layers": 4, "n_heads": 4},
        },
    )
    heads_path = node / "heads.json"
    write_json_atomic(heads_path, heads_manifest)

    gt_path = fake_repo / "artifacts" / "mod_vectors_test.pth"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({(1, 1): np.asarray(v, dtype="float64")}, gt_path)

    context = {
        "schema_version": 1,
        "model": {"name": "test-model", "revision": "abc"},
        "task": {"family": "number_add", "prompt_format": "arrow", "n_shot": 5},
        "activations": {"path": f"log/cache/z/{cache_fp}"},
        "heads_artifact": {"path": str(heads_path)},
    }
    context_path = fake_repo / "configs" / "step2_periodic_context.yaml"
    context_path.write_text(yaml.safe_dump(context), encoding="utf-8")
    return {
        "repo": fake_repo,
        "paths": paths,
        "node": node,
        "heads_path": heads_path,
        "context_path": context_path,
        "cache_fp": cache_fp,
        "gt_path": gt_path,
        "tasks": tasks,
        "planted_v": v,
    }


def _run(repo, *argv):
    return step2_cli.main(["--root", str(repo["repo"]), *argv])


def _periodic(repo, *argv):
    return _run(
        repo, "--context", str(repo["context_path"]), "--plugin", "periodic", *argv
    )


# -- CLI end-to-end ---------------------------------------------------------------


def test_cli_periodic_writes_artifact_sidecar_and_gt_comparison(periodic_repo):
    rc = _periodic(periodic_repo, "--gt-mod-vectors", str(periodic_repo["gt_path"]))
    assert rc == 0
    artifacts = list(periodic_repo["node"].glob("periodic-*.json"))
    assert len(artifacts) == 1
    manifest = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert manifest["kind"] == "step2_periodic"
    assert manifest["config"]["method"] == "mod_fit"
    assert manifest["config"]["method_version"] == 1
    assert manifest["config"]["theta_step"] == 0.01
    assert manifest["config"]["heads"] == [[1, 1], [2, 3]]
    assert manifest["config"]["head_set"] == "main"
    assert manifest["impl"]["method"] == {"name": "mod_fit", "version": 1}
    assert manifest["ks"] == list(range(1, 31))
    assert manifest["fit_task_order"] == [f"number-add{k}" for k in range(1, 31)]
    # merged lexical order differs from the fit order (number-add10 < add2)
    assert manifest["task_order"] != manifest["fit_task_order"]
    assert manifest["col_names"] == list(MOD_VECTOR_COLS)
    assert manifest["unit_cols"] == list(UNIT_COLS)
    assert manifest["magnitude_cols"] == list(MAGNITUDE_COLS)
    assert manifest["inputs"]["gt_mod_vectors"]["sha256"]
    assert manifest["gt_heads"] == ["1:1"]

    planted = manifest["per_head"]["1:1"]
    assert planted["pca_cumvar_at_6"] == pytest.approx(1.0, abs=1e-6)
    for name in MOD_VECTOR_COLS:
        assert planted["target_r2"][name] > 0.98
        assert planted["gt_cos"][name] > 0.99
    assert planted["gt_angle_6d_max_mean"][0] < 1.0
    assert planted["unit_r2_mean"] == pytest.approx(
        np.mean([planted["target_r2"][MOD_VECTOR_COLS[i]] for i in UNIT_COLS])
    )
    assert planted["magnitude_r2_mean"] == pytest.approx(
        np.mean([planted["target_r2"][MOD_VECTOR_COLS[i]] for i in MAGNITUDE_COLS])
    )
    # head absent from the GT dict: null GT fields, not a refusal
    other = manifest["per_head"]["2:3"]
    assert other["gt_cos"] is None
    assert other["gt_angle_6d_max_mean"] is None

    # sidecar: float64 (d_model, 6) per head, matching an independent refit
    sidecar = periodic_repo["node"] / manifest["mod_vectors_file"]
    assert sidecar.name == f"{artifacts[0].stem}-modvectors.npz"
    with np.load(sidecar) as npz:
        assert set(npz.files) == {"1:1", "2:3"}
        stored = npz["1:1"]
        assert stored.dtype == np.float64
        assert stored.shape == (32, 6)
        z_results, _ = load_zcache_dir(
            periodic_repo["paths"].z_cache_dir / periodic_repo["cache_fp"]
        )
        x = np.stack(
            [
                z_results[t][1, 1].to(torch.float32).numpy().astype("float64")
                for t in manifest["fit_task_order"]
            ]
        )
        refit = fit_mod_vectors(x, manifest["ks"], n_pcs=6, theta_step=0.01)
        np.testing.assert_allclose(stored, refit["mod_vectors"], atol=1e-12)


def test_cli_periodic_without_gt_omits_gt_fields(periodic_repo):
    assert _periodic(periodic_repo) == 0
    manifest = json.loads(
        list(periodic_repo["node"].glob("periodic-*.json"))[0].read_text(
            encoding="utf-8"
        )
    )
    assert manifest["gt_heads"] is None
    assert "gt_mod_vectors" not in manifest["inputs"]
    assert "gt_cos" not in manifest["per_head"]["1:1"]


# -- artifact contract ------------------------------------------------------------


def test_cli_periodic_identical_rerun_reuses_byte_stable(periodic_repo, capsys):
    gt = str(periodic_repo["gt_path"])
    assert _periodic(periodic_repo, "--gt-mod-vectors", gt) == 0
    before = tree_snapshot(periodic_repo["repo"])
    artifact = list(periodic_repo["node"].glob("periodic-*.json"))[0]
    json_bytes = artifact.read_bytes()
    sidecar_bytes = (
        periodic_repo["node"] / f"{artifact.stem}-modvectors.npz"
    ).read_bytes()
    assert _periodic(periodic_repo, "--gt-mod-vectors", gt) == 0
    assert "reused" in capsys.readouterr().out
    assert tree_snapshot(periodic_repo["repo"]) == before
    assert artifact.read_bytes() == json_bytes
    assert (
        periodic_repo["node"] / f"{artifact.stem}-modvectors.npz"
    ).read_bytes() == sidecar_bytes


def test_cli_periodic_theta_step_change_lands_in_sibling(periodic_repo):
    assert _periodic(periodic_repo) == 0
    assert _periodic(periodic_repo, "--theta-step", "0.02") == 0
    artifacts = list(periodic_repo["node"].glob("periodic-*.json"))
    assert len(artifacts) == 2
    by_step = {
        manifest["config"]["theta_step"]: manifest
        for manifest in (json.loads(p.read_text(encoding="utf-8")) for p in artifacts)
    }
    assert set(by_step) == {0.01, 0.02}


def test_cli_periodic_head_order_does_not_fork_identity(
    periodic_repo, tmp_path, capsys
):
    out_dir = tmp_path / "ord"
    base = ["--out-dir", str(out_dir)]
    assert _periodic(periodic_repo, "--heads", "2:3,1:1", *base) == 0
    assert _periodic(periodic_repo, "--heads", "1:1,2:3", *base) == 0
    assert "reused" in capsys.readouterr().out
    assert len(list(out_dir.glob("periodic-*.json"))) == 1


def test_cli_periodic_missing_sidecar_refuses_reuse(periodic_repo, capsys):
    assert _periodic(periodic_repo) == 0
    artifact = list(periodic_repo["node"].glob("periodic-*.json"))[0]
    (periodic_repo["node"] / f"{artifact.stem}-modvectors.npz").unlink()
    assert _periodic(periodic_repo) == 2
    assert "sidecar" in capsys.readouterr().err


def test_cli_periodic_plots_written_and_regenerated_on_reuse(periodic_repo):
    assert _periodic(periodic_repo) == 0
    assert not list(periodic_repo["node"].glob("periodic-*-plots"))
    assert _periodic(periodic_repo, "--plots") == 0
    plot_dirs = list(periodic_repo["node"].glob("periodic-*-plots"))
    assert len(plot_dirs) == 1
    names = {p.name for p in plot_dirs[0].iterdir()}
    assert "coords_vs_k_L1H1.png" in names
    assert "fit_vs_target_L1H1.png" in names


# -- refusals ---------------------------------------------------------------------


def test_cli_periodic_refuses_non_add_tasks(periodic_repo, capsys):
    paths = periodic_repo["paths"]
    fp = store_zcache(paths, _identity("s-nonadd"), _fake_z(["task-a", "task-b"]))
    before = tree_snapshot(periodic_repo["repo"])
    rc = _periodic(periodic_repo, "--z-cache", fp)
    assert rc == 2
    assert "not an add-k task" in capsys.readouterr().err
    assert tree_snapshot(periodic_repo["repo"]) == before


def test_cli_periodic_refuses_duplicate_k(periodic_repo, capsys):
    paths = periodic_repo["paths"]
    # distinct task id, same k=1 -> merge succeeds, the k-parse must refuse
    fp = store_zcache(paths, _identity("s-dup"), _fake_z(["number-add01"]))
    rc = _periodic(
        periodic_repo, "--z-cache", periodic_repo["cache_fp"], "--z-cache", fp
    )
    assert rc == 2
    assert "duplicate task parameter k=1" in capsys.readouterr().err


def test_cli_periodic_refuses_too_few_tasks(periodic_repo, capsys):
    paths = periodic_repo["paths"]
    few = [f"number-add{k}" for k in range(1, 6)]
    fp = store_zcache(paths, _identity("s-few"), _fake_z(few))
    rc = _periodic(periodic_repo, "--z-cache", fp)
    assert rc == 2
    assert "n_pcs+2" in capsys.readouterr().err


def test_cli_causal_projection_still_refuses_before_any_write(periodic_repo, capsys):
    before = tree_snapshot(periodic_repo["repo"])
    for plugins in (["causal_projection"], ["periodic", "causal_projection"]):
        argv = ["--context", str(periodic_repo["context_path"])]
        for plugin in plugins:
            argv += ["--plugin", plugin]
        rc = _run(periodic_repo, *argv)
        assert rc == 2
        assert "proj_meanab" in capsys.readouterr().err
        assert tree_snapshot(periodic_repo["repo"]) == before


def test_cli_periodic_refuses_missing_or_unreadable_gt(periodic_repo, capsys):
    before = tree_snapshot(periodic_repo["repo"])
    rc = _periodic(periodic_repo, "--gt-mod-vectors", "artifacts/nonexistent.pth")
    assert rc == 2
    assert "not found" in capsys.readouterr().err
    garbage = periodic_repo["repo"] / "artifacts" / "garbage.pth"
    garbage.write_bytes(b"not a torch archive")
    rc = _periodic(periodic_repo, "--gt-mod-vectors", str(garbage))
    assert rc == 2
    assert "cannot read ground-truth" in capsys.readouterr().err
    snapshot = tree_snapshot(periodic_repo["repo"])
    snapshot.discard("artifacts/garbage.pth")
    assert snapshot == before


def test_cli_periodic_explicit_heads_require_out_dir(periodic_repo, capsys):
    rc = _periodic(periodic_repo, "--heads", "1:1")
    assert rc == 2
    assert "--out-dir" in capsys.readouterr().err


def test_cli_periodic_refuses_frozen_flat_run_parent(periodic_repo, capsys):
    frozen = periodic_repo["repo"] / "log" / "runs" / "old__run" / "main_heads"
    frozen.mkdir(parents=True)
    heads_manifest = json.loads(periodic_repo["heads_path"].read_text(encoding="utf-8"))
    frozen_path = frozen / "heads.json"
    write_json_atomic(frozen_path, heads_manifest)
    before = tree_snapshot(periodic_repo["repo"])
    rc = _run(
        periodic_repo,
        "--heads-artifact",
        str(frozen_path),
        "--context",
        str(periodic_repo["context_path"]),
        "--plugin",
        "periodic",
    )
    assert rc == 2
    assert "frozen" in capsys.readouterr().err
    assert tree_snapshot(periodic_repo["repo"]) == before


def test_cli_theta_step_must_be_positive(periodic_repo, capsys):
    rc = _periodic(periodic_repo, "--theta-step", "0")
    assert rc == 2
    assert "theta_step" in capsys.readouterr().err


def test_cli_gt_mod_vectors_requires_periodic_plugin(periodic_repo, capsys):
    # generic run (no --plugin periodic) with --gt-mod-vectors must refuse,
    # not silently ignore the flag (review finding 2026-08-06)
    rc = _run(
        periodic_repo,
        "--context",
        str(periodic_repo["context_path"]),
        "--gt-mod-vectors",
        str(periodic_repo["gt_path"]),
    )
    assert rc == 2
    assert "periodic" in capsys.readouterr().err


def test_cli_tampered_sidecar_refuses_reuse(periodic_repo, capsys):
    # the manifest pins the sidecar's sha256; a modified npz must refuse
    assert _periodic(periodic_repo) == 0
    (sidecar,) = periodic_repo["node"].glob("periodic-*-modvectors.npz")
    with open(sidecar, "ab") as fh:
        fh.write(b"tamper")
    rc = _periodic(periodic_repo)
    assert rc == 2
    assert "mod_vectors_sha256" in capsys.readouterr().err


# -- real-repo anchor -------------------------------------------------------------


def _reference_r2() -> dict[tuple[int, int], tuple[float, ...]]:
    return REFERENCE_R2


@pytest.mark.skipif(
    not (ANCHOR_CACHE.is_dir() and GT_PATH.is_file()),
    reason="30-task legacy-protocol z cache or GT mod vectors not on disk",
)
def test_trio_mod_fit_anchor_on_legacy_protocol_cache():
    """Paper §4 anchor, tolerance-band (NOT bit-exact) by necessity: the exact
    legacy z file the reference numbers were computed from
    (``z_results_dict_number_add_1018.pth``) no longer exists anywhere; the
    closest surviving input is the 30-task legacy-protocol re-extraction
    ``96bcb3aad0650c71``, on which the recomputed per-direction R² lands
    within ±0.033 of the frozen legacy reference metrics.
    Bands asserted: per-direction R² within ±0.05 of the reference,
    per-direction |cos| vs the git-tracked GT mod vectors >= 0.97, and 6-D
    principal-angle max <= 15 degrees."""
    reference = _reference_r2()
    gt = torch.load(GT_PATH, map_location="cpu", weights_only=False)
    z_results, meta = load_zcache_dir(ANCHOR_CACHE)
    order, ks = parse_add_tasks(meta["tasks"])
    assert ks == list(range(1, 31))
    for head in ((15, 2), (15, 1), (13, 6)):
        layer_idx, head_idx = head
        x = (
            torch.stack([z_results[t][layer_idx, head_idx] for t in order], dim=0)
            .to(torch.float32)
            .numpy()
            .astype("float64")
        )
        fit = fit_mod_vectors(x, ks, n_pcs=6, theta_step=0.01)
        for name, expected in zip(MOD_VECTOR_COLS, reference[head], strict=True):
            assert fit["target_r2"][name] == pytest.approx(expected, abs=0.05), (
                head,
                name,
            )
        metrics = gt_metrics(fit["mod_vectors"], np.asarray(gt[head], dtype="float64"))
        for name in MOD_VECTOR_COLS:
            assert metrics["gt_cos"][name] >= 0.97, (head, name)
        assert metrics["gt_angle_6d_max_mean"][0] <= 15.0, head
