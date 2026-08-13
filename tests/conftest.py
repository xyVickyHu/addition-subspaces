"""Shared fixtures: a minimal fake repository so tests never write into the
real repo tree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from subspaces.artifacts import dataset_fingerprint
from subspaces.paths import ProjectPaths

TASKS = ("number-add1", "number-add2", "number-add3")


@pytest.fixture(autouse=True)
def _isolate_prompt_format_env(monkeypatch):
    """cli.main adopts the config's format into $FV_PROMPT_FORMAT (production
    behavior); without isolation the first CLI test would leak it into every
    later test — an order-dependence hazard (review finding F2)."""
    monkeypatch.delenv("FV_PROMPT_FORMAT", raising=False)


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    return build_fake_repo(tmp_path / "repo")


def build_fake_repo(root: Path) -> Path:
    """Repo skeleton: dataset (3 tasks), committed 2/1 split, toy matrix, config."""
    (root / "dataset_files" / "number_add").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    task_dir = root / "dataset_files" / "number_add"
    for offset, name in enumerate(TASKS, start=1):
        examples = [{"input": str(x), "output": str(x + offset)} for x in range(1, 11)]
        (task_dir / f"{name}.json").write_text(json.dumps(examples), encoding="utf-8")

    split = {
        "schema_version": 1,
        "family": "number_add",
        "dataset_fingerprint": dataset_fingerprint(task_dir),
        "source": "test fixture",
        "train_tasks": ["number-add1", "number-add3"],
        "eval_tasks": ["number-add2"],
    }
    split_path = root / "configs" / "task_splits" / "number_add_test.yaml"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(yaml.safe_dump(split), encoding="utf-8")

    matrix_dir = root / "matrices" / "toy"
    checkpoints = matrix_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    import torch

    for epoch in (0, 2, 10):
        # real loadable coefficient matrices (content varies by epoch); two
        # clear spike heads so elbow selection is deterministic
        matrix = torch.zeros(4, 4)
        matrix[1, 1] = 0.9
        matrix[2, 3] = 0.8
        matrix[0, 0] = 0.001 * (epoch + 1)
        torch.save(matrix, checkpoints / f"matrix_epoch{epoch}.pth")
    (matrix_dir / "args.json").write_text(
        json.dumps(
            {
                "model_name": "test-model",
                "task_dir": "number_add",
                "n_shot": 2,
                "n_example": 5,
                "seed": 42,
                "lambdaL1": 0.05,
                "zero_one_interval": True,
                "layer": "mid",
            }
        ),
        encoding="utf-8",
    )

    config = {
        "schema_version": 1,
        "run_name": "testrun",
        "protocol": "corrected_holdout",
        "model": {"name": "test-model", "device": "cpu"},
        "task": {
            "dataset_dir": "number_add",
            "n_shot": 2,
            "task_split": "configs/task_splits/number_add_test.yaml",
        },
        "matrix": {"mode": "reuse", "reuse_path": "matrices/toy"},
        "samples": {
            "activation": {"examples_per_task": 5, "task_set": "train"},
            "scan": {"examples_per_task": 3, "task_set": "train"},
            "heldout_activation": {
                "examples_per_task": 5,
                "seed": 1043,
                "task_set": "eval",
            },
            "final_eval": {"examples_per_task": 5, "task_set": "eval"},
        },
        "scan": {"c_min": 0, "c_max": 20},
    }
    (root / "configs" / "step1_test.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    return root


@pytest.fixture()
def fake_paths(fake_repo: Path) -> ProjectPaths:
    return ProjectPaths.from_root(fake_repo)


def tree_snapshot(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


def _fake_eval_prompts(
    model, prompts, targets, *, layer_name=None, vector=None, batch_size
):
    """Deterministic per-prompt 0/1 outcomes; sensitive to the intervention
    vector so curves vary with (head, c)."""
    signature = "clean" if vector is None else f"{float(vector.sum()):.4f}"
    correct = []
    for prompt in prompts:
        digest = hashlib.sha256(f"{prompt}|{signature}".encode()).digest()
        correct.append(1 if digest[0] < 192 else 0)  # ~75% correct
    sizes = [
        min(batch_size, len(prompts) - start)
        for start in range(0, len(prompts), batch_size)
    ]
    return correct, sizes


def _fake_compute_z(model, samples_manifest, task_ids):
    import torch

    z_results = {}
    for task_id in task_ids:
        seed = int.from_bytes(hashlib.sha256(task_id.encode()).digest()[:4], "big")
        generator = torch.Generator().manual_seed(seed)
        z_results[task_id] = torch.randn(4, 4, 8, generator=generator)
    return z_results


@pytest.fixture()
def gpu_stubs(monkeypatch):
    import subspaces.step1.eval_gpu as eval_gpu

    monkeypatch.setattr(eval_gpu, "eval_prompts", _fake_eval_prompts)
    monkeypatch.setattr(eval_gpu, "compute_z_for_tasks", _fake_compute_z)
    monkeypatch.setattr(eval_gpu, "require_resolved_model", lambda resolved: None)
    monkeypatch.setattr(eval_gpu, "load_model", lambda cfg: object())
