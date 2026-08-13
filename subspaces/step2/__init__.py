"""Step 2: activation-subspace (PCA) analysis of specified heads.

Task-generic core implemented in :mod:`subspaces.step2.pca` (per-head PCA over
per-task prompt-mean z vectors; PCs-to-threshold statistic). The
addition-only periodic analysis is implemented in :mod:`subspaces.step2.periodic`
(opt-in ``--plugin periodic``; paper-§4 mod-vector fit). The causal
projection is served by the Step-1 headset-eval projected arm, so the
``causal_projection`` plugin name remains a refusing placeholder.
"""
