"""Fixed task splits, loaded from committed files — never generated from a seed.

The canonical addition split lives at ``configs/task_splits/number_add_paper.yaml``
(recovered from the historical training ``args.json``). Loading validates that the
train/eval lists are disjoint, cover every task file in the dataset directory, all
task files exist, and the dataset directory still matches the recorded fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from subspaces.artifacts import dataset_fingerprint
from subspaces.paths import ProjectPaths

SPLIT_SCHEMA_VERSION = 1


class TaskSplitError(ValueError):
    pass


@dataclass(frozen=True)
class TaskSplit:
    family: str
    train_tasks: tuple[str, ...]
    eval_tasks: tuple[str, ...]
    dataset_fingerprint: str
    source: str

    def task_set(self, name: str) -> tuple[str, ...]:
        if name == "train":
            return self.train_tasks
        if name == "eval":
            return self.eval_tasks
        if name == "all":
            return self.train_tasks + self.eval_tasks
        raise TaskSplitError(f"unknown task set {name!r} (train|eval|all)")


def load_task_split(path: str | Path, paths: ProjectPaths) -> TaskSplit:
    split_path = paths.resolve(path)
    with open(split_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise TaskSplitError(f"{split_path}: expected a mapping")
    version = raw.get("schema_version")
    if version != SPLIT_SCHEMA_VERSION:
        raise TaskSplitError(
            f"{split_path}: schema_version {version!r} not supported "
            f"(expected {SPLIT_SCHEMA_VERSION})"
        )
    required = {"family", "train_tasks", "eval_tasks", "dataset_fingerprint", "source"}
    missing = required - set(raw)
    if missing:
        raise TaskSplitError(f"{split_path}: missing key(s) {sorted(missing)}")

    split = TaskSplit(
        family=str(raw["family"]),
        train_tasks=tuple(raw["train_tasks"]),
        eval_tasks=tuple(raw["eval_tasks"]),
        dataset_fingerprint=str(raw["dataset_fingerprint"]),
        source=str(raw["source"]),
    )
    validate_task_split(split, paths)
    return split


def validate_task_split(split: TaskSplit, paths: ProjectPaths) -> None:
    train, eval_ = set(split.train_tasks), set(split.eval_tasks)
    if len(train) != len(split.train_tasks) or len(eval_) != len(split.eval_tasks):
        raise TaskSplitError("task split contains duplicate task names")
    overlap = train & eval_
    if overlap:
        raise TaskSplitError(f"train/eval overlap: {sorted(overlap)}")

    task_dir = paths.task_dir(split.family)
    if not task_dir.is_dir():
        raise TaskSplitError(f"dataset dir not found: {task_dir}")
    on_disk = {p.stem for p in task_dir.glob("*.json")}
    listed = train | eval_
    if listed != on_disk:
        raise TaskSplitError(
            f"split does not cover the dataset dir exactly.\n"
            f"  listed but missing on disk: {sorted(listed - on_disk)}\n"
            f"  on disk but not listed:     {sorted(on_disk - listed)}"
        )

    actual = dataset_fingerprint(task_dir)
    if actual != split.dataset_fingerprint:
        raise TaskSplitError(
            f"dataset fingerprint mismatch for {task_dir}\n"
            f"  recorded {split.dataset_fingerprint}\n"
            f"  actual   {actual}\n"
            "The dataset changed since the split was committed; refusing."
        )
