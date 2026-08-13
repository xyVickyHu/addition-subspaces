"""Substep 2 — significant-head selection from the coefficient matrix.

Artifact (kind ``significant_heads``,
``<matrix-node>/sig-<method>-v<V>/significant_heads.json``):

- input reference to the matrix artifact (selected-checkpoint hash, verified
  against the actual file before loading);
- selection method and parameters, and the realized threshold;
- ordered significant heads with their (signed) coefficients;
- model dimensions (n_layers, n_heads from the matrix shape);
- implementation identity (``IMPL``) for lineage/reuse checks.

Methods are a REGISTRY keyed by ``(name, version)`` (analogous to
``subspaces/step1/selectors.py``): ``("elbow",1)``/``("fraction",1)``/``("fixed",1)``
deliberately CALL the legacy implementation (``subspaces.utils.heads.auto_threshold``
+ ``select_heads``) instead of duplicating it — zero drift risk by
construction; Lane-A equivalence tests additionally pin the canonical result
(exactly 33 heads on the paper matrix, ordered set equal to the shipped repro
scan). ``("largest_gap",1)`` is implemented here: it cuts the descending
|coef| curve at its largest consecutive drop and keeps everything at or above
the drop's upper edge.

Version split: the module-level ``algorithm_version`` is the SHARED-ENGINE
version (matrix loading, strictly-above ``select_heads`` application,
manifest assembly) and is what joins ``algorithm_versions()``; each method's
own version rides in the artifact identity via ``impl_for`` — adding or
bumping one method never changes another method's identity. CPU-only;
changing the threshold never retrains the matrix.
"""

from __future__ import annotations

from subspaces.artifacts import ArtifactError, make_manifest, sha256_file
from subspaces.config import ConfigError, Step1Config
from subspaces.paths import ProjectPaths

# v2: impl block carries {method: {name, version}} (registry refactor).
SIGNIFICANT_SCHEMA_VERSION = 2
# Shared-engine version. v2 = the registry refactor itself (numerics of the
# three legacy methods unchanged — pinned by fixtures).
ENGINE_VERSION = 2
IMPL = {"module": "subspaces.step1.significant", "algorithm_version": ENGINE_VERSION}

# Exact parameter schemas per (name, version) — config validation matches
# these exactly (missing, unknown, and irrelevant parameters are rejected).
METHOD_PARAM_SCHEMAS: dict[tuple[str, int], frozenset[str]] = {
    ("elbow", 1): frozenset(),
    ("fraction", 1): frozenset({"fraction"}),
    ("fixed", 1): frozenset({"fixed_threshold"}),
    ("largest_gap", 1): frozenset(),
}


def impl_for(sig_cfg) -> dict:
    """Implementation identity for one significant config: shared-engine
    version plus the method's own (name, version)."""
    return {
        **IMPL,
        "method": {"name": sig_cfg.method, "version": sig_cfg.method_version},
    }


def significant_identity_config(sig_cfg) -> dict:
    """The significant config slice that enters artifact identity: method,
    method_version, and ONLY the parameters registered for that (name,
    version). A future parameter FIELD added to SignificantConfig therefore
    cannot fork existing methods' identities — it only enters the identity
    of methods whose schema names it."""
    schema = METHOD_PARAM_SCHEMAS[(sig_cfg.method, sig_cfg.method_version)]
    identity = {"method": sig_cfg.method, "method_version": sig_cfg.method_version}
    for param in sorted(schema):
        identity[param] = getattr(sig_cfg, param)
    return identity


def _load_matrix(matrix_ref: dict, paths: ProjectPaths):
    import torch  # heavy import confined to the compute substep

    selected = matrix_ref["selected_checkpoint"]
    checkpoint = (
        paths.resolve(matrix_ref["matrix_dir"]) / "checkpoints" / selected["file"]
    )
    # derived checkpoints record the byte sha as "file_sha256" (volatile,
    # load-verification only); pinned/latest checkpoints record "sha256".
    expected_sha = selected.get("sha256") or selected.get("file_sha256")
    actual_sha = sha256_file(checkpoint) if checkpoint.is_file() else None
    if actual_sha != expected_sha:
        raise ArtifactError(
            f"checkpoint content mismatch for {checkpoint}\n"
            f"  matrix_ref expects sha256 {expected_sha}\n"
            f"  actual                  {actual_sha}\n"
            "Refusing to select heads from a changed checkpoint."
        )
    matrix = torch.load(checkpoint, map_location="cpu")
    if not isinstance(matrix, torch.Tensor):
        raise ArtifactError(f"{checkpoint} did not contain a torch.Tensor")
    return matrix.detach().cpu()


def largest_gap_threshold(matrix) -> tuple[float, dict]:
    """Threshold at the largest consecutive drop of the sorted |coef| curve.

    All |coefficients| of the head grid are sorted descending (no floor —
    exact zeros participate, so the drop from the smallest nonzero
    coefficient to zero is a legitimate candidate cut). The cut is the
    largest drop between consecutive values; on an exact tie between drops
    the highest cut (fewest heads) wins deterministically. The realized
    threshold is the value at the drop's LOWER edge, so the legacy
    strictly-above selection (``select_heads``) keeps exactly the heads whose
    |coefficient| is >= the drop's upper edge — including ties at the upper
    edge, excluding ties at the lower edge.

    Returns ``(threshold, info)``; ``info`` records the winning drop and the
    runner-up drop, because a near-tie between two candidate cuts is
    scientifically material (the canonical matrix has one). ``runner_up`` is
    None when no second candidate cut exists (a zero non-drop is not a
    candidate cut).
    """
    import torch  # heavy import confined to the compute substep

    values = torch.sort(matrix.detach().abs().flatten(), descending=True).values
    if values.numel() < 2:
        raise ArtifactError(
            "largest_gap is undefined on a matrix with fewer than two "
            f"coefficients (got shape {tuple(matrix.shape)})"
        )
    if not bool(torch.isfinite(values).all()):
        raise ArtifactError(
            "largest_gap refuses a matrix containing non-finite (NaN/Inf) "
            "coefficients — corrupted checkpoint? (trained matrices are "
            "[0,1]-clipped)"
        )
    drops = values[:-1] - values[1:]

    def _first_max_index(tensor) -> int:
        # torch.argmax tie-breaking is not contractual across versions;
        # first occurrence (= highest cut) is part of this method's spec.
        return int((tensor == tensor.max()).nonzero(as_tuple=True)[0][0].item())

    best = _first_max_index(drops)
    if float(drops[best].item()) <= 0.0:
        raise ArtifactError(
            "largest_gap is undefined: all matrix coefficients are equal "
            f"({float(values[0].item())!r}); the sorted curve has no drop"
        )
    runner_up = None
    if drops.numel() > 1:
        remaining = drops.clone()
        remaining[best] = float("-inf")
        second = _first_max_index(remaining)
        # a zero non-drop is not a candidate cut: runner_up stays None when
        # every other consecutive pair ties (no second candidate cut exists).
        if float(drops[second].item()) > 0.0:
            runner_up = {
                "upper": float(values[second].item()),
                "lower": float(values[second + 1].item()),
                "drop": float(drops[second].item()),
                "rank_of_upper": second + 1,
            }
    info = {
        "upper": float(values[best].item()),
        "lower": float(values[best + 1].item()),
        "drop": float(drops[best].item()),
        "rank_of_upper": best + 1,
        "n_selected": best + 1,
        "runner_up": runner_up,
    }
    return info["lower"], info


def select_significant(
    matrix_ref: dict,
    cfg: Step1Config,
    paths: ProjectPaths,
    expected: dict | None = None,
) -> dict:
    """Select significant heads; returns the ``significant_heads`` manifest.

    ``expected`` (from the pipeline) supplies the exact identity envelope
    (inputs/config/impl) so the persisted artifact matches the reuse rule.
    """
    # Legacy numerical implementation, reused on purpose (no drift).
    from subspaces.utils.heads import auto_threshold, select_heads

    matrix = _load_matrix(matrix_ref, paths)
    method = cfg.significant.method
    key = (method, cfg.significant.method_version)
    if key not in METHOD_PARAM_SCHEMAS:  # config validation makes this unreachable
        raise ConfigError(f"unknown significant method {key!r}")
    gap_info = None
    if key == ("elbow", 1):
        threshold = auto_threshold(matrix, method="elbow")
    elif key == ("fraction", 1):
        threshold = auto_threshold(
            matrix, method="fraction", fraction=cfg.significant.fraction
        )
    elif key == ("fixed", 1):
        threshold = auto_threshold(
            matrix, method="fixed", fixed=cfg.significant.fixed_threshold
        )
    elif key == ("largest_gap", 1):
        threshold, gap_info = largest_gap_threshold(matrix)
    else:
        raise ArtifactError(
            f"significant method {key!r} is registered but not dispatched — "
            "update select_significant"
        )

    heads = select_heads(matrix, threshold)
    heads_with_coefs = [
        [layer_idx, head_idx, float(matrix[layer_idx, head_idx].item())]
        for layer_idx, head_idx in heads
    ]
    n_layers, n_heads = (int(dim) for dim in matrix.shape)

    payload = {
        "impl": impl_for(cfg.significant),
        "threshold": float(threshold),
        "threshold_method": method,
        "heads": heads_with_coefs,
        "count": len(heads_with_coefs),
        "model_dims": {"n_layers": n_layers, "n_heads": n_heads},
    }
    if gap_info is not None:
        payload["gap"] = gap_info
    envelope = expected or {}
    return make_manifest(
        kind="significant_heads",
        schema_version=SIGNIFICANT_SCHEMA_VERSION,
        paths=paths,
        config=envelope.get(
            "config",
            {"significant": significant_identity_config(cfg.significant)},
        ),
        inputs=envelope.get("inputs"),
        payload=payload,
    )
