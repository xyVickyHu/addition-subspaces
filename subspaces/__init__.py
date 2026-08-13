"""subspaces: function-vector library + paper-phase entry points.

Refactored home of the original research code, replacing the monolithic
legacy utility module with a structured package.

Subpackages:
    subspaces.utils      legacy library modules (data, model, intervention, heads, …)
    subspaces.runners    entry-point CLIs (legacy per-phase runners + step1/2/3)
    subspaces.step1/2/3  the three-step analysis framework
    subspaces.paths, subspaces.config, subspaces.artifacts, subspaces.head_sets   framework plumbing

The legacy ``from subspaces import X`` re-export contract is preserved, but resolved
LAZILY (PEP 562): the heavy dependencies (torch, transformers,
transformer_lens) load only when a legacy name or ``subspaces.utils``/``subspaces.runners``
is actually accessed. Framework modules import without them (regression-tested
in ``tests/test_lazy_imports.py``).
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "LimitedTaskDataset",
    "auto_main_heads",
    "auto_threshold",
    "compute_FVs_dict",
    "compute_mean_z_result",
    "compute_nll",
    "compute_pca_FVs_dict",
    "compute_pca_subspace",
    "compute_project_vectors",
    "compute_subspace_complement",
    "compute_task_accuracy",
    "compute_z_result_per_task",
    "compute_z_results_dict",
    "create_log_file",
    "create_train_val_test_splits",
    "eval_fv_per_example_correctness",
    "extract_heads_above_threshold",
    "find_largest_elements",
    "find_output_for_input",
    "fit_mod_vectors",
    "format_input",
    "get_No_checkpoint",
    "get_largest_element_matrix",
    "get_last_checkpoint",
    "intervened_generation_with_accuracy",
    "intervened_generation_with_nll",
    "load_latest_matrix",
    "load_model_no_grad",
    "load_task_data",
    "load_tasks_split_indist_ood",
    "load_z_results_dict",
    "print_cuda_memory",
    "process_batch_data_individual",
    "recovery_scan",
    "select_heads",
    "select_main_heads_by_recovery_weak",
    "sort_files_by_epoch",
]

_LAZY_SUBMODULES = ("utils", "runners")


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in __all__:
        utils = importlib.import_module(".utils", __name__)
        return getattr(utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY_SUBMODULES))
