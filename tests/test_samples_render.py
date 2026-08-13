"""Sample-manifest rendering: legacy prompt equivalence, determinism,
no-drop counts, never-silently-regenerate, and both sampling lanes."""

from __future__ import annotations

import json

import pytest

from subspaces.artifacts import ArtifactError, semantic_fingerprint
from subspaces.config import load_step1_config
from subspaces.step1.pipeline import ensure_run, stage_selection_samples
from subspaces.step1.samples import (
    _load_prompt_format_registry,
    generate_samples,
    render_prompt,
    samples_expected,
)

EXAMPLES = [{"input": str(x), "output": str(x + 7)} for x in range(1, 11)]


def _cfg(fake_paths):
    return load_step1_config(fake_paths.root / "configs" / "step1_test.yaml")


# the COMPLETE registry — a new/removed format must consciously update this
SUPPORTED_FORMATS = ("arrow", "qa", "io", "ab", "textlabel", "fx")


def test_format_registry_is_exactly_the_supported_set():
    registry = _load_prompt_format_registry()
    assert set(registry.FORMATS) == set(SUPPORTED_FORMATS)


@pytest.mark.parametrize("format_name", SUPPORTED_FORMATS)
def test_render_matches_legacy_format_input(format_name, monkeypatch):
    """Byte-identical to the legacy renderer for BOTH the n-shot prompt and
    the zero-shot prompt (legacy: format_input([], x_q))."""
    from subspaces.utils.data import format_input  # legacy (heavy) — Lane-A anchor

    registry = _load_prompt_format_registry()
    fmt = registry.get_format(format_name)
    demo_ids, query_id = [2, 5, 0], 7
    prompt, zero_shot, expected = render_prompt(EXAMPLES, demo_ids, query_id, fmt)

    monkeypatch.setenv("FV_PROMPT_FORMAT", format_name)
    x_icl = [EXAMPLES[i]["input"] for i in demo_ids]
    legacy = format_input(x_icl, EXAMPLES[query_id]["input"], EXAMPLES)
    legacy_zero_shot = format_input([], EXAMPLES[query_id]["input"], EXAMPLES)
    assert prompt == legacy
    assert zero_shot == legacy_zero_shot
    assert expected == EXAMPLES[query_id]["output"]


def test_arrow_prompt_shape():
    registry = _load_prompt_format_registry()
    prompt, zero_shot, expected = render_prompt(
        EXAMPLES, [0, 1], 2, registry.get_format("arrow")
    )
    assert prompt == "1->8#2->9#3->"
    assert zero_shot == "3->"
    assert expected == "10"


def test_corrected_manifest_contents_and_determinism(fake_paths, tmp_path):
    cfg = _cfg(fake_paths)
    _, split, _ = ensure_run(cfg, fake_paths)
    manifests = stage_selection_samples(cfg, fake_paths, split)

    assert set(manifests) == {"activation", "scan"}
    scan_manifest, _scan_path = manifests["scan"]
    assert scan_manifest["task_order"] == list(split.train_tasks)
    for task_id, count in scan_manifest["counts"].items():
        assert count == cfg.samples.scan.examples_per_task  # no dropped examples
        samples = scan_manifest["tasks"][task_id]["samples"]
        assert len(samples) == count
        for sample in samples:
            assert sample["query_id"] not in sample["demo_ids"]
            assert sample["prompt"].endswith(f"{sample['query']['input']}->")
            assert sample["expected"] == sample["query"]["output"]

    # determinism: regenerating into a fresh dir yields the same semantics
    fresh = generate_samples(
        cfg,
        cfg.samples.scan,
        "scan",
        split,
        fake_paths,
        tmp_path / "samples_alt",
    )
    assert semantic_fingerprint(fresh) == semantic_fingerprint(scan_manifest)


def test_existing_manifest_reused_not_regenerated(fake_paths, monkeypatch):
    cfg = _cfg(fake_paths)
    _, split, _ = ensure_run(cfg, fake_paths)
    first = stage_selection_samples(cfg, fake_paths, split)

    import subspaces.step1.samples as samples_mod

    monkeypatch.setattr(
        samples_mod,
        "corrected_sample_plan",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must reuse")),
    )
    second = stage_selection_samples(cfg, fake_paths, split)
    assert second == first


def test_changed_spec_refuses_stale_manifest(fake_paths):
    cfg = _cfg(fake_paths)
    _, split, _ = ensure_run(cfg, fake_paths)
    _, scan_path = stage_selection_samples(cfg, fake_paths, split)["scan"]

    cfg.samples.scan.seed = 43  # different sampling identity, same file path
    with pytest.raises(ArtifactError, match="refusing to reuse"):
        generate_samples(
            cfg,
            cfg.samples.scan,
            "scan",
            split,
            fake_paths,
            scan_path.parent,
        )


def test_legacy_lane_runs_legacy_draw_code(fake_paths, tmp_path):
    import yaml

    raw = yaml.safe_load(
        (fake_paths.root / "configs" / "step1_test.yaml").read_text(encoding="utf-8")
    )
    raw["protocol"] = "legacy_reproduction"
    raw["samples"] = {
        "activation": {"examples_per_task": 5, "task_set": "all"},
        "scan": {"examples_per_task": 3, "task_set": "all"},
        "final_eval": {"examples_per_task": 3, "task_set": "all"},
    }
    legacy_path = tmp_path / "legacy.yaml"
    legacy_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_step1_config(legacy_path)

    run_dir, split, _ = ensure_run(cfg, fake_paths)
    manifest = generate_samples(
        cfg,
        cfg.samples.scan,
        "scan",
        split,
        fake_paths,
        run_dir / "step1" / "samples",
    )
    assert manifest["config"]["mode"] == "legacy"
    assert set(manifest["task_order"]) == {
        "number-add1",
        "number-add2",
        "number-add3",
    }
    for data in manifest["tasks"].values():
        assert len(data["samples"]) == 3
        for sample in data["samples"]:
            assert sample["query_id"] not in sample["demo_ids"]

    again = generate_samples(
        cfg,
        cfg.samples.scan,
        "scan",
        split,
        fake_paths,
        run_dir / "step1" / "samples_alt",
    )
    assert semantic_fingerprint(again) == semantic_fingerprint(manifest)


def test_legacy_lane_deterministic_in_fresh_process(fake_paths):
    """Audit regression closed: the first subspaces.utils import consumes global RNG
    state, so it must happen BEFORE seeding — a fresh interpreter generating
    the first eval-kind manifest must match the in-process result."""
    import subprocess
    import sys

    import yaml

    raw = yaml.safe_load(
        (fake_paths.root / "configs" / "step1_test.yaml").read_text(encoding="utf-8")
    )
    raw["protocol"] = "legacy_reproduction"
    raw["samples"] = {
        "activation": {"examples_per_task": 5, "task_set": "all"},
        "scan": {"examples_per_task": 3, "task_set": "all"},
        "final_eval": {"examples_per_task": 3, "task_set": "all"},
    }
    legacy_path = fake_paths.root / "configs" / "legacy_fresh.yaml"
    legacy_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    code = (
        "import json, sys\n"
        "from subspaces.artifacts import semantic_fingerprint\n"
        "from subspaces.config import load_step1_config\n"
        "from subspaces.paths import ProjectPaths\n"
        "from subspaces.step1.pipeline import ensure_run\n"
        "from subspaces.step1.samples import generate_samples\n"
        f"paths = ProjectPaths.from_root({str(fake_paths.root)!r})\n"
        f"cfg = load_step1_config({str(legacy_path)!r})\n"
        "run_dir, split, _ = ensure_run(cfg, paths)\n"
        "manifest = generate_samples(cfg, cfg.samples.scan, 'scan', split, paths, "
        "run_dir / 'step1' / 'fresh_samples')\n"
        "print(semantic_fingerprint(manifest))\n"
    )
    import os

    env = {**os.environ, "MKL_THREADING_LAYER": "GNU"}  # torch+numpy-mkl coexistence
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    fresh_fingerprint = result.stdout.strip().splitlines()[-1]

    # in-process (subspaces.utils long imported) generation into a different dir
    from subspaces.artifacts import semantic_fingerprint
    from subspaces.step1.pipeline import ensure_run
    from subspaces.step1.samples import generate_samples

    cfg = load_step1_config(legacy_path)
    run_dir, split, _ = ensure_run(cfg, fake_paths)
    manifest = generate_samples(
        cfg,
        cfg.samples.scan,
        "scan",
        split,
        fake_paths,
        run_dir / "step1" / "inproc_samples",
    )
    assert semantic_fingerprint(manifest) == fresh_fingerprint


def test_all_queries_used_once_at_full_count(fake_paths):
    cfg = _cfg(fake_paths)
    cfg.samples.scan.examples_per_task = 10  # == n_examples in the fixture tasks
    run_dir, split, _ = ensure_run(cfg, fake_paths)
    manifest = generate_samples(
        cfg,
        cfg.samples.scan,
        "scan",
        split,
        fake_paths,
        run_dir / "step1" / "samples",
    )
    for data in manifest["tasks"].values():
        query_ids = [sample["query_id"] for sample in data["samples"]]
        assert query_ids == list(range(10))  # every unique query exactly once


def test_json_serializable_and_prompt_format_recorded(fake_paths):
    cfg = _cfg(fake_paths)
    _, split, _ = ensure_run(cfg, fake_paths)
    manifest, on_disk_path = stage_selection_samples(cfg, fake_paths, split)[
        "activation"
    ]
    # content-addressed cache slot: log/cache/samples/<fp16>/<kind>.json where
    # fp16 derives from the identity envelope computable BEFORE generation
    expected = samples_expected(
        cfg, cfg.samples.activation, "activation", split, fake_paths
    )
    assert (
        on_disk_path
        == fake_paths.samples_cache_dir
        / semantic_fingerprint(expected)[:16]
        / "activation.json"
    )
    on_disk = json.loads(on_disk_path.read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert on_disk["prompt_format"]["name"] == "arrow"
    assert on_disk["prompt_format"]["out_sep"] == "#"
