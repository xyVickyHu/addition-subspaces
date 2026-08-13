"""Step-2 ``periodic`` plugin — addition-only §4 mod-vector decomposition.

Faithful port of the frozen legacy math (``pca.py::opt_shift`` /
``fit_mod_vectors`` / ``subspace_principal_angles``, themselves verbatim ports
of the original ``coords_heads.ipynb::fit_head_period``): for each requested
head, PCA over the head's per-task prompt-mean output vectors → top-``n_pcs``
subspace, then a least-squares fit of six named periodic feature patterns of
the task parameter k (add-k):

    mod50  = cos(2πk/50 + θ*)   [magnitude]     mod2   = cos(2πk/2)   [unit]
    mod5   = cos(2πk/5  + θ*)   [unit]          mod10c = cos(2πk/10)  [unit]
    mod10s = sin(2πk/10)        [unit]          mod25  = cos(2πk/25 + θ*) [magnitude]

``mod_vectors = pc_subspace @ weights`` (d_model, 6) are the six feature
directions in activation space; columns 1-4 span the 4-D UNIT subspace and
columns 0,5 the 2-D MAGNITUDE subspace (paper §4.2/4.3).

This analysis is meaningful only on the add-k family: task ids must all match
``number-add<k>`` with unique integer k, and the fit runs over rows sorted by
ASCENDING k (the reference protocol), not by lexical task id. Anything else
refuses before any write. float64 throughout the fit.

NOTE (known trap): the top-``n_pcs`` basis is ``pca.components_.T[:, :n_pcs]``.
Never route this through ``subspaces.utils.pca.compute_pca_subspace`` — its list
argument selects 1-indexed COMPONENTS (``[6]`` = only PC6, a 1-D subspace).

Artifact: kind ``step2_periodic`` — a hash-named single-slot sibling
(``periodic-<identity10>.json``) with a ``periodic-<identity10>-modvectors.npz``
sidecar (keys ``"L:H"`` → float64 (d_model, 6)), following the
``subspaces.step2.pca.run_subspace_analysis`` contract (reuse-or-refuse; plots carry
no identity). Ground-truth comparison against a saved ``mod_vectors_dict.pth``
(``{(l,h): (d_model, 6)}``) is optional; heads absent from the GT dict report
null GT fields (not a refusal).

CPU-only: given existing z caches, no model is loaded and no GPU is needed.
"""

from __future__ import annotations

import re
from pathlib import Path

from subspaces.artifacts import (
    ArtifactError,
    file_ref,
    identity_of,
    make_manifest,
    manifest_ref,
    reuse_or_refuse,
    write_json_atomic,
)
from subspaces.head_sets import read_heads_manifest, validate_heads
from subspaces.paths import ProjectPaths
from subspaces.step2.pca import default_out_dir, load_and_merge_zcaches

STEP2_PERIODIC_KIND = "step2_periodic"
STEP2_PERIODIC_SCHEMA_VERSION = 1
ENGINE_VERSION = 1
IMPL = {"module": "subspaces.step2.periodic", "algorithm_version": ENGINE_VERSION}

# Exact parameter schemas per (name, version) — the subspaces/step2/pca.py pattern.
METHOD_PARAM_SCHEMAS: dict[tuple[str, int], frozenset[str]] = {
    ("mod_fit", 1): frozenset({"n_pcs", "theta_step"}),
}
METHOD = "mod_fit"
METHOD_VERSION = 1

# Column order of the six feature directions (legacy pca.py).
# Matches the legacy mod_vectors_dict.pth artifact column-for-column — do not
# reorder, or every GT comparison silently compares mismatched directions.
MOD_VECTOR_COLS = ("mod50", "mod2", "mod5", "mod10c", "mod10s", "mod25")
UNIT_COLS = (1, 2, 3, 4)  # periods 2/5/10 (cos2, cos5, cos10, sin10) → 4-D unit
MAGNITUDE_COLS = (0, 5)  # periods 50/25 → 2-D magnitude (tens digit)

_ADD_TASK_RE = re.compile(r"^number-add(\d+)$")


def impl() -> dict:
    return {**IMPL, "method": {"name": METHOD, "version": METHOD_VERSION}}


def identity_config(cfg, heads: list[tuple[int, int]], head_set: str) -> dict:
    """Identity slice: method, method_version, EXACTLY the registered
    parameters, and the resolved head set (sorted — per-head fits are
    independent, so selector output order must not fork identity)."""
    schema = METHOD_PARAM_SCHEMAS[(METHOD, METHOD_VERSION)]
    identity = {"method": METHOD, "method_version": METHOD_VERSION}
    for param in sorted(schema):
        identity[param] = getattr(cfg, param)
    identity["heads"] = [list(head) for head in sorted(heads)]
    identity["head_set"] = head_set
    return identity


# -- add-k task parsing ---------------------------------------------------------


def parse_add_tasks(task_ids: list[str]) -> tuple[list[str], list[int]]:
    """Map task ids to their integer k, refusing anything outside the add-k
    family, and return ``(fit_task_order, ks)`` sorted by ASCENDING k.

    The §4 fit is defined over the task parameter k; the reference protocol
    (the legacy notebook / run_pca_decomposition) orders rows
    numerically by k, not by lexical task id.
    """
    by_k: dict[int, str] = {}
    for task_id in task_ids:
        match = _ADD_TASK_RE.fullmatch(task_id)
        if match is None:
            raise ArtifactError(
                f"task {task_id!r} is not an add-k task (expected "
                "'number-add<k>'): the periodic plugin fits periodic functions "
                "of k and is meaningful only on the number-add family; refusing."
            )
        k = int(match.group(1))
        if k in by_k:
            raise ArtifactError(
                f"duplicate task parameter k={k} ({by_k[k]!r} and {task_id!r}): "
                "the periodic fit needs one row per unique k; refusing."
            )
        by_k[k] = task_id
    ks = sorted(by_k)
    return [by_k[k] for k in ks], ks


# -- the §4 math (verbatim port from the legacy pca.py) --------------------------


def opt_shift(coords, omega, xs, theta_step: float = 0.01):
    """Phase-search helper (paper §4.2): return ``cos(2π·omega·xs + θ*)``
    where θ* minimises the least-squares residual of reconstructing the
    MEAN-REMOVED cosine from the PC ``coords``. The returned target is the
    NON-mean-removed cosine at the best phase."""
    import numpy as np

    best, best_theta = np.inf, 0.0
    for theta in np.arange(0, 2 * np.pi, theta_step):
        fnc = np.cos(2 * np.pi * omega * xs + theta)
        fnc = fnc - fnc.mean()
        _, resid, _, _ = np.linalg.lstsq(coords, fnc, rcond=None)
        r = (resid[0] / (np.linalg.norm(fnc) ** 2)) if resid.size else np.inf
        if r < best:
            best, best_theta = r, theta
    return np.cos(2 * np.pi * omega * xs + best_theta)


def fit_mod_vectors(x, ks, n_pcs: int = 6, theta_step: float = 0.01) -> dict:
    """§4.2/4.3 mod-vector fit for one head (faithful legacy port).

    ``x`` is the head's per-task matrix (n_tasks, d_model) as a NUMPY array
    (torch input is unsupported — callers upcast bf16 via
    ``.to(torch.float32).numpy()`` first, matching the reference numerics);
    row ``i`` corresponds to task value ``ks[i]`` (ascending in the standard
    call path; the fit itself is row-permutation invariant given matched ks).

    1. float64; sklearn ``PCA()`` full fit → ``pc_subspace =
       components_.T[:, :n_pcs]`` (top-n_pcs — NOT ``compute_pca_subspace``,
       whose list argument means 1-indexed component picks).
    2. ``coords = x @ pc_subspace``, mean-centred over tasks.
    3. Six periodic targets over ``xs = ks`` (phases via :func:`opt_shift`),
       least squares ``coords @ weights ≈ targets``; per-direction R²
       (column-mean-removed ss_tot, 0-guarded).
    4. ``mod_vectors = pc_subspace @ weights`` (d_model, 6).
    """
    import numpy as np
    from sklearn.decomposition import PCA

    x = np.asarray(x, dtype="float64")
    xs = np.asarray(ks, dtype="float64")
    n = x.shape[0]

    pca = PCA()
    pca.fit(x)
    pc_subspace = pca.components_.T[:, :n_pcs]  # (d_model, n_pcs)
    cumvar = float(np.cumsum(pca.explained_variance_ratio_)[n_pcs - 1])

    coords = x @ pc_subspace  # (n, n_pcs)
    coords = coords - coords.mean(axis=0)

    targets = np.zeros((n, 6))
    targets[:, 0] = opt_shift(coords, 0.02, xs, theta_step)  # ~period 50 (magnitude)
    targets[:, 1] = np.cos(2 * np.pi * xs / 2)  # period 2      (unit)
    targets[:, 2] = opt_shift(coords, 0.2, xs, theta_step)  # period 5      (unit)
    targets[:, 3] = np.cos(2 * np.pi * xs / 10)  # period 10 cos (unit)
    targets[:, 4] = np.sin(2 * np.pi * xs / 10)  # period 10 sin (unit)
    targets[:, 5] = opt_shift(coords, 0.04, xs, theta_step)  # ~period 25 (magnitude)

    weights, _, _, _ = np.linalg.lstsq(coords, targets, rcond=None)  # (n_pcs, 6)
    targets_hat = coords @ weights
    ss_res = ((targets - targets_hat) ** 2).sum(axis=0)
    ss_tot = ((targets - targets.mean(axis=0)) ** 2).sum(axis=0)
    target_r2 = 1.0 - ss_res / np.where(ss_tot == 0, 1.0, ss_tot)

    mod_vectors = pc_subspace @ weights  # (d_model, 6)
    return {
        "mod_vectors": mod_vectors,
        "pc_subspace": pc_subspace,
        "weights": weights,
        "coords": coords,
        "targets": targets,
        "targets_hat": targets_hat,
        "target_r2": {
            name: float(r) for name, r in zip(MOD_VECTOR_COLS, target_r2, strict=True)
        },
        "pca_cumvar_at_npcs": cumvar,
    }


def subspace_principal_angles(a, b) -> tuple[float, float]:
    """Max and mean principal angles (degrees) between the column spans of
    ``a``/``b`` (D, r): QR-orthonormalize, SVD of the cross-Gram, arccos."""
    import numpy as np

    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    ang = np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))
    return float(ang.max()), float(ang.mean())


# -- ground truth ---------------------------------------------------------------


def _sha256_of(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gt_mod_vectors(path: Path) -> dict:
    """Load a ground-truth ``mod_vectors_dict.pth``: ``{(l, h): (d_model, 6)}``
    (numpy float64 or torch tensors). Unreadable or mis-shaped files refuse.

    Deliberate dtype upgrade vs the legacy reference runner: torch GT
    tensors are kept at float64 here (the reference cast to float32). On the
    shipped GT (float32-representable values) both paths are bit-identical;
    a genuinely-float64 GT would differ from the legacy runner at ~1e-7."""
    import numpy as np
    import torch

    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ArtifactError(
            f"cannot read ground-truth mod vectors {path}: {exc}"
        ) from exc
    if not isinstance(raw, dict) or not raw:
        raise ArtifactError(
            f"{path}: expected a non-empty dict {{(l, h): (d_model, 6)}}, "
            f"got {type(raw).__name__}"
        )
    gt: dict[tuple[int, int], object] = {}
    for key, value in raw.items():
        try:
            layer_idx, head_idx = (int(key[0]), int(key[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise ArtifactError(
                f"{path}: GT key {key!r} is not an (l, h) head tuple"
            ) from exc
        array = (
            value.detach().cpu().to(torch.float64).numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value, dtype="float64")
        )
        if array.ndim != 2 or array.shape[1] != len(MOD_VECTOR_COLS):
            raise ArtifactError(
                f"{path}: GT entry {key!r} has shape {array.shape}, expected "
                f"(d_model, {len(MOD_VECTOR_COLS)})"
            )
        gt[(layer_idx, head_idx)] = array
    return gt


def gt_metrics(mod_vectors, gt_array) -> dict:
    """Per-column |cos| and principal angles (6-D / 4-D unit / 2-D magnitude)
    between the fitted mod vectors and a GT entry — the legacy
    ``run_pca_decomposition`` comparison, key for key."""
    import numpy as np

    cos = {
        MOD_VECTOR_COLS[j]: float(
            abs(
                np.dot(mod_vectors[:, j], gt_array[:, j])
                / (np.linalg.norm(mod_vectors[:, j]) * np.linalg.norm(gt_array[:, j]))
            )
        )
        for j in range(len(MOD_VECTOR_COLS))
    }
    return {
        "gt_cos": cos,
        "gt_angle_6d_max_mean": list(subspace_principal_angles(mod_vectors, gt_array)),
        "gt_angle_unit_max_mean": list(
            subspace_principal_angles(
                mod_vectors[:, list(UNIT_COLS)], gt_array[:, list(UNIT_COLS)]
            )
        ),
        "gt_angle_magnitude_max_mean": list(
            subspace_principal_angles(
                mod_vectors[:, list(MAGNITUDE_COLS)], gt_array[:, list(MAGNITUDE_COLS)]
            )
        ),
    }


# -- orchestration (mirrors subspaces.step2.pca.run_subspace_analysis) ------------------


def run_periodic_analysis(
    cfg,
    heads: list[tuple[int, int]],
    paths: ProjectPaths,
    *,
    head_set: str,
    heads_artifact: str | Path | None,
    z_cache_dirs: list[Path],
    out_dir: Path | None = None,
    plots: bool = False,
    expected_model: dict | None = None,
) -> tuple[dict, Path, bool]:
    """Compute (or reuse) one step2_periodic artifact.

    Returns ``(manifest, artifact_path, reused)``. The artifact is a
    hash-named single-slot sibling of the heads artifact
    (``periodic-<identity10>.json`` + ``periodic-<identity10>-modvectors.npz``
    sidecar), immutable under the shared reuse-or-refuse rule.
    """
    import numpy as np

    z_results, info = load_and_merge_zcaches(z_cache_dirs, paths)
    cache_model = info["model"] or {}
    if expected_model:
        for field in ("name", "revision"):
            declared = expected_model.get(field)
            actual = cache_model.get(field)
            if declared and actual and declared != actual:
                raise ArtifactError(
                    f"context.model.{field} ({declared!r}) does not match the "
                    f"z caches' extraction identity ({actual!r}); refusing a "
                    "mislabeled analysis."
                )
    dims = info["dims"]
    validate_heads(heads, n_layers=dims["n_layers"], n_heads=dims["n_heads"])

    # add-k family gate + fit row order (ascending k, NOT lexical task order)
    fit_task_order, ks = parse_add_tasks(info["task_order"])
    if len(ks) < cfg.n_pcs + 2:
        raise ArtifactError(
            f"periodic fit needs at least n_pcs+2 = {cfg.n_pcs + 2} add-k "
            f"tasks (n_pcs={cfg.n_pcs}), got {len(ks)}; refusing an "
            "underdetermined fit."
        )

    inputs: dict = {
        "z_caches": [
            {"content_fingerprint": cache["content_fingerprint"]}
            for cache in sorted(
                info["z_caches"], key=lambda c: c["content_fingerprint"]
            )
        ]
    }
    resolved_artifact: Path | None = None
    if heads_artifact is not None:
        resolved_artifact = paths.resolve(heads_artifact)
        heads_manifest, _kind = read_heads_manifest(resolved_artifact)
        inputs["heads_artifact"] = manifest_ref(
            resolved_artifact, paths, heads_manifest
        )
    gt = None
    if cfg.gt_mod_vectors is not None:
        # file_ref refuses a missing file; loading refuses an unreadable one.
        # The ref's sha256 joins identity (its path key is volatile), so a
        # different GT artifact lands in a sibling file.
        inputs["gt_mod_vectors"] = file_ref(cfg.gt_mod_vectors, paths)
        gt = load_gt_mod_vectors(paths.resolve(cfg.gt_mod_vectors))

    expected = {
        "kind": STEP2_PERIODIC_KIND,
        "schema_version": STEP2_PERIODIC_SCHEMA_VERSION,
        "inputs": inputs,
        "config": identity_config(cfg, heads, head_set),
        "impl": impl(),
    }
    target_dir = (
        paths.resolve(out_dir)
        if out_dir is not None
        else default_out_dir(resolved_artifact, paths, kind_label=STEP2_PERIODIC_KIND)
    )
    stem = f"periodic-{identity_of(expected)[:10]}"
    out_path = target_dir / f"{stem}.json"
    sidecar_path = target_dir / f"{stem}-modvectors.npz"
    existing = reuse_or_refuse(
        out_path,
        expected,
        expect_kind=STEP2_PERIODIC_KIND,
        max_schema_version=expected["schema_version"],
    )
    if existing is not None:
        if not sidecar_path.is_file():
            raise ArtifactError(
                f"{out_path} exists but its mod-vector sidecar "
                f"{sidecar_path.name} is missing; the artifact is incomplete. "
                "Remove the manifest explicitly to regenerate."
            )
        recorded_sha = existing.get("mod_vectors_sha256")
        if recorded_sha is not None and _sha256_of(sidecar_path) != recorded_sha:
            raise ArtifactError(
                f"{sidecar_path} content does not match the manifest's "
                "mod_vectors_sha256 — the sidecar was modified or truncated. "
                "Remove the manifest explicitly to regenerate."
            )
        if plots:
            # plots carry no identity and are not referenced by the manifest;
            # the fit is cheap and deterministic, so regenerate them from the
            # same inputs on a reused artifact.
            fits = _compute_fits(cfg, heads, z_results, fit_task_order, ks)
            _write_plots(fits, target_dir / f"{stem}-plots", ks)
        return existing, out_path, True

    fits = _compute_fits(cfg, heads, z_results, fit_task_order, ks)
    cumvar_key = f"pca_cumvar_at_{cfg.n_pcs}"
    per_head: dict[str, dict] = {}
    mod_vectors_out: dict[str, object] = {}
    for (layer_idx, head_idx), fit in fits.items():
        key = f"{layer_idx}:{head_idx}"
        r2 = fit["target_r2"]
        record = {
            cumvar_key: fit["pca_cumvar_at_npcs"],
            "target_r2": r2,
            "unit_r2_mean": float(np.mean([r2[MOD_VECTOR_COLS[i]] for i in UNIT_COLS])),
            "magnitude_r2_mean": float(
                np.mean([r2[MOD_VECTOR_COLS[i]] for i in MAGNITUDE_COLS])
            ),
            "mean_r2": float(np.mean(list(r2.values()))),
        }
        if gt is not None:
            if (layer_idx, head_idx) in gt:
                record.update(gt_metrics(fit["mod_vectors"], gt[(layer_idx, head_idx)]))
            else:
                # covered-by-GT is a property of the GT artifact, not an
                # input error: absent heads report null, they do not refuse.
                record.update(
                    {
                        "gt_cos": None,
                        "gt_angle_6d_max_mean": None,
                        "gt_angle_unit_max_mean": None,
                        "gt_angle_magnitude_max_mean": None,
                    }
                )
        per_head[key] = record
        mod_vectors_out[key] = fit["mod_vectors"]  # float64 (d_model, 6)

    manifest = make_manifest(
        kind=STEP2_PERIODIC_KIND,
        schema_version=expected["schema_version"],
        paths=paths,
        config=expected["config"],
        inputs=expected["inputs"],
        payload={
            "impl": expected["impl"],
            "model": info["model"],
            "hook_site": info["hook_site"],
            "dataset_fingerprint": info["dataset_fingerprint"],
            "dims": dims,
            "task_order": info["task_order"],
            "fit_task_order": fit_task_order,
            "ks": ks,
            "n_tasks": len(ks),
            "z_caches": info["z_caches"],
            "col_names": list(MOD_VECTOR_COLS),
            "unit_cols": list(UNIT_COLS),
            "magnitude_cols": list(MAGNITUDE_COLS),
            "gt_heads": (
                sorted(f"{gl}:{gh}" for gl, gh in gt) if gt is not None else None
            ),
            "mod_vectors_file": sidecar_path.name,
            "mod_vectors_sha256": None,  # filled after savez, before the manifest
            "per_head": per_head,
        },
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    # sidecar first, manifest last: the hash-named JSON is the commit point,
    # so a crash in between leaves no manifest claiming a missing sidecar.
    np.savez(sidecar_path, **{k: np.asarray(v) for k, v in mod_vectors_out.items()})
    manifest["mod_vectors_sha256"] = _sha256_of(sidecar_path)
    write_json_atomic(out_path, manifest)
    if plots:
        _write_plots(fits, target_dir / f"{stem}-plots", ks)
    return manifest, out_path, False


def _compute_fits(cfg, heads, z_results, fit_task_order, ks) -> dict:
    """Per-head §4 fit over rows ordered by ascending k. The bf16→float32→
    float64 upcast path matches the legacy ``fit_mod_vectors`` exactly."""
    import torch

    fits = {}
    for layer_idx, head_idx in sorted(heads):
        x = (
            torch.stack(
                [z_results[task][layer_idx, head_idx] for task in fit_task_order],
                dim=0,
            )
            .to(torch.float32)
            .numpy()
            .astype("float64")
        )
        fits[(layer_idx, head_idx)] = fit_mod_vectors(
            x, ks, n_pcs=cfg.n_pcs, theta_step=cfg.theta_step
        )
    return fits


def _write_plots(fits: dict, plots_dir: Path, ks: list[int]) -> None:
    """Per-head diagnostic plots (ported from coords_heads.ipynb cells 4/11):
    (a) PC coordinate vs k for components 1..6 (2x3 grid); (b) fitted
    ``targets_hat`` ('.-') vs target patterns ('--') per direction. Plots are
    best-effort, carry no identity, and are not referenced by the manifest."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir.mkdir(parents=True, exist_ok=True)
        colors = ("m", "r", "g", "b", "k", "y")
        for (layer_idx, head_idx), fit in fits.items():
            tag = f"L{layer_idx}H{head_idx}"
            coords = fit["coords"]
            n_show = min(6, coords.shape[1])
            fig, axs = plt.subplots(2, 3, figsize=(16, 8))
            for idx, ax in enumerate(axs.flatten()):
                if idx >= n_show:
                    ax.axis("off")
                    continue
                ax.plot(ks, coords[:, idx], linewidth=2)
                ax.set_title(f"Component {idx + 1}")
                if idx % 3 == 0:
                    ax.set_ylabel("Coordinates")
                if idx >= 3:
                    ax.set_xlabel("Task Add-k Index (k)")
                ax.grid(alpha=0.3)
            fig.suptitle(f"PC coordinates vs k — {tag}")
            fig.tight_layout()
            fig.savefig(str(plots_dir / f"coords_vs_k_{tag}.png"), dpi=120)
            plt.close(fig)

            targets, targets_hat = fit["targets"], fit["targets_hat"]
            plt.figure(figsize=(10, 6))
            for j, name in enumerate(MOD_VECTOR_COLS):
                color = colors[j % len(colors)]
                plt.plot(ks, targets_hat[:, j], ".-", color=color, label=name)
                plt.plot(ks, targets[:, j], "--", color=color, alpha=0.6)
            plt.xlabel("Task Add-k Index (k)")
            plt.ylabel("Coordinate function")
            plt.title(f"Fit ('.-') vs target ('--') — {tag}")
            plt.legend(ncol=3, fontsize=8)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(str(plots_dir / f"fit_vs_target_{tag}.png"), dpi=120)
            plt.close()
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"[step2:periodic] plot step skipped: {exc}")
