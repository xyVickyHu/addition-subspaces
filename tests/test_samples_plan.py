import pytest

from subspaces.config import SampleSpec, load_step1_config
from subspaces.step1 import pipeline
from subspaces.step1 import samples as samples_mod
from subspaces.step1.pipeline import ensure_run
from subspaces.step1.samples import (
    HoldoutViolation,
    corrected_sample_plan,
    resolve_sample_tasks,
)
from subspaces.step1.split import load_task_split


def test_plan_all_queries_used_exactly_once():
    plan = corrected_sample_plan(100, 100, n_shot=5, seed=42)
    assert [entry["query_id"] for entry in plan] == list(range(100))
    for entry in plan:
        assert len(entry["demo_ids"]) == 5
        assert entry["query_id"] not in entry["demo_ids"]
        assert len(set(entry["demo_ids"])) == 5  # no replacement


def test_plan_deterministic_and_seed_sensitive():
    assert corrected_sample_plan(100, 100, 5, 42) == corrected_sample_plan(
        100, 100, 5, 42
    )
    assert corrected_sample_plan(100, 100, 5, 42) != corrected_sample_plan(
        100, 100, 5, 43
    )


def test_plan_subset_deterministic_without_replacement():
    plan = corrected_sample_plan(100, 20, n_shot=5, seed=42)
    query_ids = [entry["query_id"] for entry in plan]
    assert len(query_ids) == 20 and len(set(query_ids)) == 20
    assert query_ids == sorted(query_ids)


def test_plan_bounds():
    with pytest.raises(ValueError, match="exceeds"):
        corrected_sample_plan(100, 101, 5, 42)
    with pytest.raises(ValueError, match="n_shot"):
        corrected_sample_plan(5, 5, 5, 42)


def test_guard_selection_side_train_only(fake_paths):
    split = load_task_split("configs/task_splits/number_add_test.yaml", fake_paths)
    spec = SampleSpec(examples_per_task=3, task_set="train")
    tasks = resolve_sample_tasks(spec, "scan", split, "corrected_holdout")
    assert tasks == ("number-add1", "number-add3")  # train tasks only

    bad = SampleSpec(examples_per_task=3, task_set="all")
    with pytest.raises(HoldoutViolation, match="task_set=train"):
        resolve_sample_tasks(bad, "activation", split, "corrected_holdout")
    with pytest.raises(HoldoutViolation, match="task split"):
        resolve_sample_tasks(spec, "scan", None, "corrected_holdout")


def test_guard_final_eval_is_eval_only(fake_paths):
    split = load_task_split("configs/task_splits/number_add_test.yaml", fake_paths)
    spec = SampleSpec(examples_per_task=3, task_set="eval")
    assert resolve_sample_tasks(spec, "final_eval", split, "corrected_holdout") == (
        "number-add2",
    )
    bad = SampleSpec(examples_per_task=3, task_set="train")
    with pytest.raises(HoldoutViolation, match="task_set=eval"):
        resolve_sample_tasks(bad, "final_eval", split, "corrected_holdout")


def test_guard_heldout_activation_is_eval_side(fake_paths):
    """The 4th sample role: held-out activation estimation is eval-side and
    never a selection-side manifest."""
    split = load_task_split("configs/task_splits/number_add_test.yaml", fake_paths)
    spec = SampleSpec(examples_per_task=3, seed=1043, task_set="eval")
    tasks = resolve_sample_tasks(spec, "heldout_activation", split, "corrected_holdout")
    assert tasks == ("number-add2",)
    bad = SampleSpec(examples_per_task=3, task_set="train")
    with pytest.raises(HoldoutViolation, match="task_set=eval"):
        resolve_sample_tasks(bad, "heldout_activation", split, "corrected_holdout")


def test_heldout_activation_and_final_eval_use_distinct_seeds():
    """Prompts for held-out z-estimation vs final evaluation must differ: the
    committed default seeds are distinct, so the demo draws differ even where
    queries overlap."""
    from subspaces.config import Step1Config

    cfg = Step1Config()
    assert cfg.samples.heldout_activation.seed != cfg.samples.final_eval.seed
    plan_a = corrected_sample_plan(100, 100, 5, cfg.samples.heldout_activation.seed)
    plan_b = corrected_sample_plan(100, 100, 5, cfg.samples.final_eval.seed)
    assert plan_a != plan_b


def test_legacy_protocol_passthrough(fake_paths):
    split = load_task_split("configs/task_splits/number_add_test.yaml", fake_paths)
    spec = SampleSpec(examples_per_task=3, task_set="all")
    tasks = resolve_sample_tasks(spec, "scan", split, "legacy_reproduction")
    assert set(tasks) == {"number-add1", "number-add2", "number-add3"}


def test_pipeline_never_materializes_final_eval_before_selection(
    fake_paths, monkeypatch
):
    """Data-access guard: the composed selection path requests only
    activation+scan manifests, over training tasks only."""
    cfg = load_step1_config(fake_paths.root / "configs" / "step1_test.yaml")
    _journal_dir, split, _resolved = ensure_run(cfg, fake_paths)

    requested: list[tuple[str, tuple[str, ...]]] = []

    def recorder(cfg_, spec, kind, split_, paths_, out_dir):
        tasks = resolve_sample_tasks(spec, kind, split_, cfg_.protocol)
        requested.append((kind, tasks))
        return {"kind": kind}

    monkeypatch.setattr(samples_mod, "generate_samples", recorder)
    pipeline.stage_selection_samples(cfg, fake_paths, split)

    kinds = [kind for kind, _ in requested]
    assert kinds == ["activation", "scan"]
    eval_tasks = set(split.eval_tasks)
    for _kind, tasks in requested:
        assert not (set(tasks) & eval_tasks), "held-out task leaked into selection"
