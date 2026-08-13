"""``subspaces.runners`` — entry-point CLIs, one per paper analysis phase.

Each module exposes a ``main()`` that parses argv and runs the phase.
The parent-level ``.py`` files (``FV_matrix.py``, ``head_eval.py``, etc.)
are thin shims that delegate here so existing CLI invocations and sbatch
wrappers keep working unchanged.

Modules:
    train_matrix             paper §3 — head-selection optimization training
    run_head_select          paper §3 (post-training) — extract significant heads from a matrix
    run_head_eval            paper §3 — per-head intervention accuracy sweep
    run_head_sum_eval        paper §3 — subset-FV intervention accuracy sweep
    run_pca_decomposition    paper §4 — PCA + 6D periodic+magnitude basis
    run_pipeline             end-to-end orchestrator across all phases
"""
