"""aie_corrupted sample kind: demo_label_permutation v1 planning/rendering,
determinism, identity forks, RNG decoupling, and held-out isolation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import yaml
from conftest import build_fake_repo

from subspaces.artifacts import dataset_fingerprint, semantic_fingerprint
from subspaces.config import SampleSpec, load_step1_config
from subspaces.paths import ProjectPaths
from subspaces.step1.samples import (
    HoldoutViolation,
    generate_samples,
    permute_demo_labels,
    resolve_sample_tasks,
    samples_expected,
)
from subspaces.step1.split import load_task_split


def write_aie_config(root: Path, name: str = "step1_aie_test.yaml", **aie) -> Path:
    """Clone the fake-repo config and add an aie block (+ matching
    samples.aie_corrupted spec, as the consistency guard requires)."""
    raw = yaml.safe_load(
        (root / "configs" / "step1_test.yaml").read_text(encoding="utf-8")
    )
    block = {
        "methods": ["cie_replace", "zs_add_proxy"],
        "corrupted_examples_per_task": 4,
        "corrupted_seed": 7,
        "corruption_tries": 20,
        "k_values": [2],
        "batch_size": 2,
        "proxy_batch_size": 3,
    }
    block.update(aie)
    raw["aie"] = block
    raw["samples"]["aie_corrupted"] = {
        "examples_per_task": block["corrupted_examples_per_task"],
        "seed": block["corrupted_seed"],
        "task_set": "train",
    }
    out = root / "configs" / name
    out.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return out


def _load(root: Path, name: str = "step1_aie_test.yaml", **aie):
    path = write_aie_config(root, name, **aie)
    cfg = load_step1_config(path)
    paths = ProjectPaths.from_root(root)
    split = load_task_split(cfg.task.task_split, paths)
    return cfg, paths, split


# -- permutation rule (pure) ---------------------------------------------------


def test_permute_distinct_labels_has_zero_fixed_points():
    labels = ["1", "2", "3", "4", "5"]
    permuted, n_fixed = permute_demo_labels(labels, random.Random(0), tries=20)
    assert n_fixed == 0
    assert sorted(permuted) == sorted(labels)  # multiset preserved
    assert all(new != old for new, old in zip(permuted, labels, strict=True))


def test_permute_is_deterministic_for_a_seeded_rng():
    labels = ["a", "b", "c", "d"]
    first = permute_demo_labels(labels, random.Random(123), tries=20)
    second = permute_demo_labels(labels, random.Random(123), tries=20)
    assert first == second


def test_permute_few_distinct_labels_keeps_first_minimal_draw():
    # ["a","a","b"]: zero fixed points is impossible (whichever slot gets
    # "b", one of the two "a"s stays in an "a" slot) — minimum is 1.
    permuted, n_fixed = permute_demo_labels(["a", "a", "b"], random.Random(0), 20)
    assert n_fixed == 1
    assert sorted(permuted) == ["a", "a", "b"]
    # constant labels: every permutation is the identity by value
    permuted, n_fixed = permute_demo_labels(["7", "7", "7"], random.Random(0), 20)
    assert permuted == ["7", "7", "7"]
    assert n_fixed == 3


def test_permute_bounds():
    with pytest.raises(ValueError, match="tries"):
        permute_demo_labels(["a", "b"], random.Random(0), 0)
    with pytest.raises(ValueError, match="empty"):
        permute_demo_labels([], random.Random(0), 5)


# -- manifest generation -------------------------------------------------------


def test_corrupted_manifest_deterministic_across_calls(fake_repo, tmp_path):
    cfg, paths, split = _load(fake_repo)
    spec = cfg.samples.aie_corrupted
    first = generate_samples(cfg, spec, "aie_corrupted", split, paths, tmp_path / "a")
    second = generate_samples(cfg, spec, "aie_corrupted", split, paths, tmp_path / "b")
    assert semantic_fingerprint(first) == semantic_fingerprint(second)


def test_corrupted_samples_zero_fixed_points_and_intact_query(fake_repo, tmp_path):
    cfg, paths, split = _load(fake_repo)
    manifest = generate_samples(
        cfg, cfg.samples.aie_corrupted, "aie_corrupted", split, paths, tmp_path
    )
    assert manifest["config"]["corruption"] == {
        "name": "demo_label_permutation",
        "version": 1,
        "tries": 20,
    }
    task_dir = paths.task_dir(cfg.task.dataset_dir)
    for task_id, data in manifest["tasks"].items():
        examples = json.loads(
            (task_dir / f"{task_id}.json").read_text(encoding="utf-8")
        )
        assert len(data["samples"]) == 4
        for sample in data["samples"]:
            original = sample["original_demo_outputs"]
            corrupted = [demo["output"] for demo in sample["demos"]]
            # fake-repo labels are distinct within a task -> always achievable
            assert sample["n_fixed_points"] == 0
            assert sorted(corrupted) == sorted(original)
            assert all(new != old for new, old in zip(corrupted, original, strict=True))
            # demo INPUTS and the query/expected stay intact
            for demo, demo_id in zip(sample["demos"], sample["demo_ids"], strict=True):
                assert demo["input"] == examples[demo_id]["input"]
            query = examples[sample["query_id"]]
            assert sample["query"] == query
            assert sample["expected"] == str(query["output"])
            # the rendered prompt carries the CORRUPTED demos + intact query
            rendered = "".join(
                f"{demo['input']}->{demo['output']}#" for demo in sample["demos"]
            )
            assert sample["prompt"] == rendered + f"{query['input']}->"
            assert sample["zero_shot_prompt"] == f"{query['input']}->"


def test_corrupted_fallback_records_fixed_points(tmp_path):
    """A constant-label task cannot reach zero fixed points; the manifest
    still generates, recording the minimal count per sample."""
    root = build_fake_repo(tmp_path / "repo")
    task_dir = root / "dataset_files" / "number_add"
    constant = [{"input": str(x), "output": "7"} for x in range(1, 11)]
    (task_dir / "number-add1.json").write_text(json.dumps(constant), encoding="utf-8")
    split_path = root / "configs" / "task_splits" / "number_add_test.yaml"
    split_raw = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    split_raw["dataset_fingerprint"] = dataset_fingerprint(task_dir)
    split_path.write_text(yaml.safe_dump(split_raw), encoding="utf-8")

    cfg, paths, split = _load(root)
    manifest = generate_samples(
        cfg, cfg.samples.aie_corrupted, "aie_corrupted", split, paths, tmp_path / "m"
    )
    for sample in manifest["tasks"]["number-add1"]["samples"]:
        assert sample["n_fixed_points"] == cfg.task.n_shot  # all-identical labels
        assert [demo["output"] for demo in sample["demos"]] == ["7"] * cfg.task.n_shot
    for sample in manifest["tasks"]["number-add3"]["samples"]:
        assert sample["n_fixed_points"] == 0  # distinct labels unaffected


def test_identity_forks_on_seed_examples_and_rule_params(fake_repo):
    cfg, paths, split = _load(fake_repo)
    base = samples_expected(
        cfg, cfg.samples.aie_corrupted, "aie_corrupted", split, paths
    )

    cfg_seed, _, _ = _load(fake_repo, "aie_seed.yaml", corrupted_seed=8)
    cfg_count, _, _ = _load(fake_repo, "aie_count.yaml", corrupted_examples_per_task=3)
    cfg_tries, _, _ = _load(fake_repo, "aie_tries.yaml", corruption_tries=5)
    for other in (cfg_seed, cfg_count, cfg_tries):
        forked = samples_expected(
            other, other.samples.aie_corrupted, "aie_corrupted", split, paths
        )
        assert semantic_fingerprint(forked) != semantic_fingerprint(base)


def test_corrupted_draws_decoupled_from_activation_at_same_seed(fake_repo, tmp_path):
    """Same seed + count as the activation kind must give a DIFFERENT draw
    (the kind participates in the per-task seed derivation)."""
    cfg, paths, split = _load(
        fake_repo,
        "aie_same_seed.yaml",
        corrupted_examples_per_task=5,
        corrupted_seed=42,
    )
    assert cfg.samples.activation.examples_per_task == 5
    assert cfg.samples.activation.seed == 42
    activation = generate_samples(
        cfg, cfg.samples.activation, "activation", split, paths, tmp_path / "act"
    )
    corrupted = generate_samples(
        cfg, cfg.samples.aie_corrupted, "aie_corrupted", split, paths, tmp_path / "cor"
    )
    demo_draws = {
        kind: [
            sample["demo_ids"]
            for task in manifest["task_order"]
            for sample in manifest["tasks"][task]["samples"]
        ]
        for kind, manifest in (("activation", activation), ("aie", corrupted))
    }
    assert demo_draws["activation"] != demo_draws["aie"]


def test_corrupted_kind_is_selection_side_train_only(fake_repo):
    cfg, paths, split = _load(fake_repo)
    spec = SampleSpec(examples_per_task=3, task_set="train")
    tasks = resolve_sample_tasks(spec, "aie_corrupted", split, "corrected_holdout")
    assert tasks == ("number-add1", "number-add3")
    for bad_set in ("eval", "all"):
        bad = SampleSpec(examples_per_task=3, task_set=bad_set)
        with pytest.raises(HoldoutViolation, match="task_set=train"):
            resolve_sample_tasks(bad, "aie_corrupted", split, "corrected_holdout")


def test_corrupted_kind_requires_the_aie_block(fake_repo):
    paths = ProjectPaths.from_root(fake_repo)
    cfg = load_step1_config(fake_repo / "configs" / "step1_test.yaml")
    split = load_task_split(cfg.task.task_split, paths)
    assert cfg.aie is None
    with pytest.raises(ValueError, match="aie config block"):
        samples_expected(cfg, cfg.samples.aie_corrupted, "aie_corrupted", split, paths)
