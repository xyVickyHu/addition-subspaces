"""Step 1: identify selected and main attention heads.

Substeps (each consumes a persisted artifact from the preceding one):

1. ``matrix``      — train or reuse the coefficient matrix (GPU when training)
2. ``selected`` — select selected heads from the matrix (CPU)
3. ``samples``     — materialize deterministic sample manifests (CPU)
4. ``recovery``    — per-head recovery scan + selector-independent baselines (GPU)
5. ``selectors``   — apply a main-head selector to the saved scan (pure CPU)
6. ``headset_eval``— evaluate a main selection's head sets on final_eval_samples (GPU)
7. ``pipeline``    — compose the above; fingerprint-based reuse; heads artifact
"""
