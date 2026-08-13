from pathlib import Path

import pytest
import yaml

from subspaces.paths import ProjectPaths, find_repo_root
from subspaces.step1.split import TaskSplitError, load_task_split

REPO = find_repo_root(Path(__file__))


def test_committed_paper_split_is_exact_25_5():
    paths = ProjectPaths.from_root(REPO)
    split = load_task_split("configs/task_splits/number_add_paper.yaml", paths)
    assert len(split.train_tasks) == 25
    assert len(split.eval_tasks) == 5
    assert set(split.eval_tasks) == {
        "number-add20",
        "number-add2",
        "number-add27",
        "number-add16",
        "number-add21",
    }
    # disjoint + full coverage of the 30 add-k tasks
    assert not set(split.train_tasks) & set(split.eval_tasks)
    assert len(set(split.task_set("all"))) == 30
    # historical order preserved (first entries of args.json indist_keys)
    assert split.train_tasks[:3] == ("number-add28", "number-add30", "number-add7")


def _mutate_split(fake_repo: Path, **updates) -> Path:
    split_path = fake_repo / "configs" / "task_splits" / "number_add_test.yaml"
    raw = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    raw.update(updates)
    split_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return split_path


def test_fixture_split_loads(fake_repo, fake_paths):
    split = load_task_split("configs/task_splits/number_add_test.yaml", fake_paths)
    assert split.task_set("train") == ("number-add1", "number-add3")
    assert split.task_set("eval") == ("number-add2",)
    with pytest.raises(TaskSplitError, match="unknown task set"):
        split.task_set("holdout")


def test_overlap_refused(fake_repo, fake_paths):
    _mutate_split(fake_repo, eval_tasks=["number-add2", "number-add1"])
    with pytest.raises(TaskSplitError, match="overlap"):
        load_task_split("configs/task_splits/number_add_test.yaml", fake_paths)


def test_incomplete_coverage_refused(fake_repo, fake_paths):
    _mutate_split(fake_repo, train_tasks=["number-add1"])
    with pytest.raises(TaskSplitError, match="not listed"):
        load_task_split("configs/task_splits/number_add_test.yaml", fake_paths)


def test_missing_file_refused(fake_repo, fake_paths):
    (fake_repo / "dataset_files" / "number_add" / "number-add3.json").unlink()
    with pytest.raises(TaskSplitError, match="missing on disk"):
        load_task_split("configs/task_splits/number_add_test.yaml", fake_paths)


def test_dataset_change_refused(fake_repo, fake_paths):
    victim = fake_repo / "dataset_files" / "number_add" / "number-add1.json"
    victim.write_text(victim.read_text() + " ", encoding="utf-8")
    with pytest.raises(TaskSplitError, match="fingerprint mismatch"):
        load_task_split("configs/task_splits/number_add_test.yaml", fake_paths)


FAMILY_SPLITS = {
    # split file -> (n_train, n_eval, first held-out task)
    "number_add_subtract_paper.yaml": (26, 5, "number-add-19"),
    "number_mul_paper.yaml": (25, 5, "number-mul16"),
    "extractive_paper.yaml": (22, 5, "fruit_v_animal_3"),
    "abstractive_paper.yaml": (21, 5, "person-occupation"),
}


@pytest.mark.parametrize("split_file", sorted(FAMILY_SPLITS))
def test_committed_family_splits_load_and_validate(split_file):
    """Every committed sweep-family split loads through the full validator
    (disjoint, covers the dataset dir, files exist, fingerprint matches)."""
    paths = ProjectPaths.from_root(REPO)
    split = load_task_split(f"configs/task_splits/{split_file}", paths)
    n_train, n_eval, first_eval = FAMILY_SPLITS[split_file]
    assert len(split.train_tasks) == n_train
    assert len(split.eval_tasks) == n_eval
    assert split.eval_tasks[0] == first_eval  # historical order preserved
    assert not set(split.train_tasks) & set(split.eval_tasks)
