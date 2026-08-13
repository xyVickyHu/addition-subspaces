"""AIE chain orchestration: config validation, artifact identity (reuse /
fork / refuse), npz digest verification, CLI overrides, and run-identity
stability of non-AIE configs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import build_fake_repo
from test_aie_samples import write_aie_config

from subspaces.artifacts import ArtifactError
from subspaces.config import ConfigError, load_step1_config, requested_semantics
from subspaces.paths import ProjectPaths
from subspaces.step1 import aie as aie_mod
from subspaces.step1 import pipeline


def _fake_method_scores(model_loader, method, samples_manifest, z_results, cfg):
    """Deterministic, method-sensitive stand-in for the GPU scorer over the
    fake repo's 4x4 grid (matches conftest's fake z shape)."""
    model_loader()
    task_order = list(samples_manifest["task_order"])
    heads = aie_mod.head_grid(4, 4, cfg.aie.head_limit)
    offset = 3 if method == "cie_replace" else 5
    per_task_cie, per_task_base = {}, {}
    for shift, task in enumerate(task_order):
        cie = np.full((4, 4), np.nan, dtype=np.float32)
        for layer_idx, head_idx in heads:
            cie[layer_idx, head_idx] = (
                layer_idx * 4 + head_idx - offset + 0.1 * shift
            ) / 10.0
        per_task_cie[task] = cie
        n_prompts = len(samples_manifest["tasks"][task]["samples"])
        per_task_base[task] = np.full(n_prompts, 0.25, dtype=np.float32)
    aie = np.mean(np.stack([per_task_cie[t] for t in task_order]), axis=0).astype(
        np.float32
    )
    return {
        "per_task_cie": per_task_cie,
        "aie": aie,
        "per_task_base": per_task_base,
        "task_order": task_order,
        "heads": heads,
        "grid": {"n_layers": 4, "n_heads": 4},
    }


@pytest.fixture()
def aie_repo(tmp_path, gpu_stubs, monkeypatch):
    """Fake repo + aie config + stubbed GPU scorer (call-counted)."""
    root = build_fake_repo(tmp_path / "repo")
    config_path = write_aie_config(root, k_values=[2, 3])
    calls: list[str] = []

    def counting_scores(model_loader, method, samples_manifest, z_results, cfg):
        calls.append(method)
        return _fake_method_scores(
            model_loader, method, samples_manifest, z_results, cfg
        )

    monkeypatch.setattr(aie_mod, "compute_method_scores", counting_scores)
    return root, config_path, calls


def _run(root: Path, config_path: Path, **overrides) -> dict:
    paths = ProjectPaths.from_root(root)
    cfg = load_step1_config(config_path)
    journal_dir, split, resolved = pipeline.ensure_run(cfg, paths)
    return aie_mod.run_aie(
        cfg,
        paths,
        journal_dir=journal_dir,
        split=split,
        resolved=resolved,
        **overrides,
    )


def _aie_dir(root: Path) -> Path:
    journals = list((root / "log" / "journal").glob("*/aie"))
    assert len(journals) == 1
    return journals[0]


# -- config validation ---------------------------------------------------------


def test_config_refuses_bad_method_and_empty_k(fake_repo):
    with pytest.raises(ConfigError, match="unknown method"):
        load_step1_config(write_aie_config(fake_repo, methods=["nope"]))
    with pytest.raises(ConfigError, match="k_values"):
        load_step1_config(write_aie_config(fake_repo, k_values=[]))
    with pytest.raises(ConfigError, match="duplicates"):
        load_step1_config(write_aie_config(fake_repo, k_values=[2, 2]))
    with pytest.raises(ConfigError, match=">= 1"):
        load_step1_config(write_aie_config(fake_repo, corruption_tries=0))


def test_config_refuses_spec_contradicting_aie_block(fake_repo):
    config_path = write_aie_config(fake_repo)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["samples"]["aie_corrupted"]["seed"] = 99  # aie block says 7
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="contradicts the aie block"):
        load_step1_config(config_path)


def test_run_identity_of_non_aie_configs_is_unchanged_by_the_block(fake_repo):
    plain = load_step1_config(fake_repo / "configs" / "step1_test.yaml")
    with_aie = load_step1_config(write_aie_config(fake_repo))
    assert requested_semantics(plain) == requested_semantics(with_aie)
    semantics = requested_semantics(plain)
    assert "aie" not in semantics
    assert "aie_corrupted" not in semantics["samples"]


def test_cli_refuses_config_without_aie_block(fake_repo, capsys):
    from subspaces.runners import step1 as step1_cli

    rc = step1_cli.main(
        [
            "--root",
            str(fake_repo),
            "aie",
            "--config",
            str(fake_repo / "configs" / "step1_test.yaml"),
        ]
    )
    assert rc == 2
    assert "no aie block" in capsys.readouterr().err
    assert not (fake_repo / "log" / "journal").exists()  # refused pre-journal


# -- chain + artifact identity -------------------------------------------------


def test_chain_produces_scores_and_evals_and_reuses_on_rerun(aie_repo):
    root, config_path, calls = aie_repo
    summary = _run(root, config_path)
    aie_dir = _aie_dir(root)

    score_files = sorted(p.name for p in aie_dir.glob("scores-*.json"))
    assert len(score_files) == 2  # one per method
    eval_files = sorted(p.name for p in aie_dir.glob("eval-*.json"))
    assert len(eval_files) == 4  # 2 methods x k in {2, 3}
    assert sorted(calls) == ["cie_replace", "zs_add_proxy"]
    assert set(summary["evals"]) == {
        "cie_replace-k2",
        "cie_replace-k3",
        "zs_add_proxy-k2",
        "zs_add_proxy-k3",
    }
    for entry in summary["evals"].values():
        assert set(entry["metrics"]) == {"clean", "aie_unit"}

    # scores artifacts carry the full identity + integrity block
    cie_scores = json.loads(
        next(aie_dir.glob("scores-cie_replace-*.json")).read_text(encoding="utf-8")
    )
    proxy_scores = json.loads(
        next(aie_dir.glob("scores-zs_add_proxy-*.json")).read_text(encoding="utf-8")
    )
    assert cie_scores["impl"]["method"] == {"name": "cie_replace", "version": 2}
    assert proxy_scores["impl"]["method"] == {"name": "zs_add_proxy", "version": 1}
    assert cie_scores["config"]["readout"] == "first_token_prob"
    # sites joins the scores identity ONLY for the proxy (cie patches at
    # each head's own layer and never reads inject_layer)
    assert "sites" not in cie_scores["config"]
    assert "sites" in proxy_scores["config"]
    assert set(cie_scores["inputs"]) == {"samples", "z_cache"}
    npz = np.load(aie_dir / cie_scores["arrays"]["file"])
    assert "aie" in npz.files

    # evals rank by SIGNED aie: k=2 on the fake matrix = the two largest
    evaluation = json.loads(
        next(aie_dir.glob("eval-cie_replace-k2-*.json")).read_text(encoding="utf-8")
    )
    assert evaluation["config"]["ranking"] == "signed_desc"
    assert [head[:2] for head in evaluation["heads"]] == [[3, 3], [3, 2]]

    # identical rerun: everything reused, no recompute, no new files
    before = sorted(p.name for p in aie_dir.iterdir())
    _run(root, config_path)
    assert sorted(calls) == ["cie_replace", "zs_add_proxy"]  # unchanged
    assert sorted(p.name for p in aie_dir.iterdir()) == before


def test_changed_batch_size_forks_only_the_affected_method(aie_repo):
    root, config_path, calls = aie_repo
    _run(root, config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["aie"]["batch_size"] = 4  # cie_replace identity changes; proxy does not
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    _run(root, config_path)
    aie_dir = _aie_dir(root)
    assert len(list(aie_dir.glob("scores-cie_replace-*.json"))) == 2  # forked
    assert len(list(aie_dir.glob("scores-zs_add_proxy-*.json"))) == 1  # reused
    assert calls.count("cie_replace") == 2
    assert calls.count("zs_add_proxy") == 1


def test_tampered_scores_manifest_refuses(aie_repo):
    root, config_path, _calls = aie_repo
    _run(root, config_path)
    aie_dir = _aie_dir(root)
    target = next(aie_dir.glob("scores-cie_replace-*.json"))
    manifest = json.loads(target.read_text(encoding="utf-8"))
    manifest["config"]["batch_size"] = 999  # same path, different identity
    target.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError, match="refusing to reuse or"):
        _run(root, config_path)


def test_corrupted_scores_npz_refuses_reuse(aie_repo):
    root, config_path, _calls = aie_repo
    _run(root, config_path)
    aie_dir = _aie_dir(root)
    manifest_path = next(aie_dir.glob("scores-cie_replace-*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    npz_path = aie_dir / manifest["arrays"]["file"]
    loaded = {key: np.load(npz_path)[key] for key in np.load(npz_path).files}
    loaded["aie"] = loaded["aie"] + 1.0
    np.savez_compressed(npz_path, **loaded)
    with pytest.raises(ArtifactError, match="digest mismatch"):
        _run(root, config_path)


def test_corrupted_eval_outcomes_npz_refuses_reuse(aie_repo):
    root, config_path, _calls = aie_repo
    _run(root, config_path)
    aie_dir = _aie_dir(root)
    outcomes_path = next(aie_dir.glob("eval-cie_replace-k2-*-outcomes.npz"))
    loaded = {key: np.load(outcomes_path)[key] for key in np.load(outcomes_path).files}
    loaded["aie_unit"] = 1 - loaded["aie_unit"]
    np.savez_compressed(outcomes_path, **loaded)
    with pytest.raises(ArtifactError, match="digest mismatch"):
        _run(root, config_path)


def test_eval_forks_on_k_and_scores_identity(aie_repo):
    root, config_path, _calls = aie_repo
    _run(root, config_path)
    aie_dir = _aie_dir(root)
    k2 = next(aie_dir.glob("eval-cie_replace-k2-*.json"))
    k3 = next(aie_dir.glob("eval-cie_replace-k3-*.json"))
    manifest_k2 = json.loads(k2.read_text(encoding="utf-8"))
    manifest_k3 = json.loads(k3.read_text(encoding="utf-8"))
    assert manifest_k2["config"]["k"] == 2 and manifest_k3["config"]["k"] == 3
    assert len(manifest_k2["heads"]) == 2 and len(manifest_k3["heads"]) == 3
    # per-method scores refs differ -> the two methods' evals never collide
    proxy_k2 = json.loads(
        next(aie_dir.glob("eval-zs_add_proxy-k2-*.json")).read_text(encoding="utf-8")
    )
    assert (
        proxy_k2["inputs"]["aie_scores"]["semantic_fingerprint"]
        != manifest_k2["inputs"]["aie_scores"]["semantic_fingerprint"]
    )


def test_head_limit_restricts_scoring_and_selection(tmp_path, gpu_stubs, monkeypatch):
    root = build_fake_repo(tmp_path / "repo")
    config_path = write_aie_config(root, k_values=[2], head_limit=3)
    monkeypatch.setattr(aie_mod, "compute_method_scores", _fake_method_scores)
    _run(root, config_path, methods=["cie_replace"])
    aie_dir = _aie_dir(root)
    scores = json.loads(
        next(aie_dir.glob("scores-cie_replace-*.json")).read_text(encoding="utf-8")
    )
    assert scores["n_scored_heads"] == 3
    assert scores["head_limit_applied"] == 3
    evaluation = json.loads(
        next(aie_dir.glob("eval-cie_replace-k2-*.json")).read_text(encoding="utf-8")
    )
    # best heads within the flat cap [(0,0),(0,1),(0,2)] by the fake scores
    assert [head[:2] for head in evaluation["heads"]] == [[0, 2], [0, 1]]


# -- CLI overrides -------------------------------------------------------------


def test_overrides_narrow_but_never_extend(aie_repo):
    root, config_path, calls = aie_repo
    _run(root, config_path, methods=["zs_add_proxy"], k_values=[2])
    assert calls == ["zs_add_proxy"]
    aie_dir = _aie_dir(root)
    assert len(list(aie_dir.glob("eval-*.json"))) == 1
    with pytest.raises(ConfigError, match="contradicts the config's aie.methods"):
        _run(root, config_path, methods=["bogus"])
    with pytest.raises(ConfigError, match="contradicts the config's aie.k_values"):
        _run(root, config_path, k_values=[7])
    with pytest.raises(ConfigError, match="at least one"):
        _run(root, config_path, methods=[])


def test_run_aie_refuses_without_block(fake_repo):
    paths = ProjectPaths.from_root(fake_repo)
    cfg = load_step1_config(fake_repo / "configs" / "step1_test.yaml")
    journal_dir, split, resolved = pipeline.ensure_run(cfg, paths)
    with pytest.raises(ConfigError, match="no aie block"):
        aie_mod.run_aie(
            cfg, paths, journal_dir=journal_dir, split=split, resolved=resolved
        )


def test_make_samples_materializes_corrupted_kind_for_aie_configs(fake_repo):
    from subspaces.runners import step1 as step1_cli

    config_path = write_aie_config(fake_repo)
    rc = step1_cli.main(
        ["--root", str(fake_repo), "make-samples", "--config", str(config_path)]
    )
    assert rc == 0
    manifests = list(
        (fake_repo / "log" / "cache" / "samples").glob("*/aie_corrupted.json")
    )
    assert len(manifests) == 1
    # and no held-out manifest leaked out of the selection stage
    assert not list((fake_repo / "log" / "cache" / "samples").glob("*/final_eval.json"))
