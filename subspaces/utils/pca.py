"""PCA subspace fitting + projection (paper §4)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA

from .activations import compute_FVs_dict
from .heads import get_largest_element_matrix


def compute_subspace_complement(M, chosen_indices):
    """Return the subspace inside ``span(M)`` orthogonal to ``M[:, chosen_indices]``.

    Inputs:
        ``M`` — (D, r) matrix, assumed full column rank.
        ``chosen_indices`` — subset of column indices to exclude.
    Output: (D, r - |chosen|) orthonormal columns spanning the complement
    inside ``span(M)``.
    """
    if isinstance(M, torch.Tensor):
        M = M.cpu().numpy()
    Q, R = np.linalg.qr(M[:, chosen_indices])
    rest_indices = [i for i in range(M.shape[1]) if i not in chosen_indices]
    rest_proj_on_Q = Q @ Q.T @ M[:, rest_indices]
    rest_proj_out_Q = M[:, rest_indices] - rest_proj_on_Q
    rest_proj_out_Q_orth, _ = np.linalg.qr(rest_proj_out_Q)
    return rest_proj_out_Q_orth


def compute_pca_subspace(FVs_dict, n_components, task_list=None):
    """PCA over the FV matrix; selects components per ``n_components``.

    ``n_components`` semantics (preserved from the legacy API):
        ``[0]``        — return the empty subspace.
        ``[-1]``       — return all components (PCA basis).
        ``[k]``        — return the top-k components.
        ``[i, j, k]``  — return the listed components (1-indexed).
        ``[-1-k]``     — return all components except the k-th.

    Returns ``(subspace_components, mean_FVs, explained_variance)``.
    """
    if task_list is not None:
        train_FVs_dict = {key: FVs_dict[key] for key in task_list if key in FVs_dict}
    else:
        train_FVs_dict = FVs_dict

    train_FVs_matrix = torch.cat(
        [FVs.cpu().detach().unsqueeze(0) for FVs in train_FVs_dict.values()], dim=0
    ).numpy()

    mean_FVs = np.mean(train_FVs_matrix, axis=0)

    pca = PCA()
    pca.fit(train_FVs_matrix)

    if n_components == [0]:
        subspace_components = pca.components_.T[:, : n_components[0]]
    elif n_components == [-1]:
        subspace_components = pca.components_.T
        print("get all components in compute_pca_subspace")
    elif n_components[0] < -1:
        all_components = list(range(pca.components_.shape[0]))
        for item in n_components:
            all_components.remove(-1 - item - 1)
        subspace_components = pca.components_.T[:, all_components]
    else:
        n_components = [n - 1 for n in n_components]
        subspace_components = pca.components_.T[:, n_components]

    explained_variance = pca.explained_variance_ratio_
    return subspace_components, mean_FVs, explained_variance


def compute_pca_FVs_dict(
    z_results_dict,
    M,
    train_keys,
    H1=[-1],
    H2=[-1],
    in_subspace1=None,
    out_subspace1=None,
    device="cuda",
):
    """Project per-task FVs derived from the two leading head positions onto PCA subspaces.

    ``M1`` and ``M2`` are sparse matrices selecting only the top-1 and top-2 head
    positions; ``H1`` / ``H2`` choose PCA components (see ``compute_pca_subspace``).
    """
    M1 = get_largest_element_matrix(M, [1], val_or_pos="val", device=device)
    FVs_dict_1 = compute_FVs_dict(z_results_dict, M1, device)
    M2 = get_largest_element_matrix(M, [2], val_or_pos="val", device=device)
    FVs_dict_2 = compute_FVs_dict(z_results_dict, M2, device)
    pca_FVs_dict_1 = FVs_dict_1.copy()
    pca_FVs_dict_2 = FVs_dict_2.copy()
    subspace_components, mean_train_FVs, _ = compute_pca_subspace(
        FVs_dict_1, H1, train_keys
    )
    if H1 != [-1] or in_subspace1 is not None or out_subspace1 is not None:
        for key, FV in FVs_dict_1.items():
            if out_subspace1 is not None:
                out_FV = compute_project_vectors(
                    FV.unsqueeze(0), out_subspace1, mean_train_FVs
                ).squeeze(0)
                mean_train_FVs_tensor = torch.tensor(
                    mean_train_FVs, dtype=FV.dtype, device=FV.device
                )
                # add the mean back (current version, since coords_allcomb_exc)
                pca_FVs_dict_1[key] = FV - out_FV + mean_train_FVs_tensor
            elif in_subspace1 is not None:
                pca_FVs_dict_1[key] = compute_project_vectors(
                    FV.unsqueeze(0), in_subspace1, mean_train_FVs
                ).squeeze(0)
            else:
                pca_FVs_dict_1[key] = compute_project_vectors(
                    FV.unsqueeze(0), subspace_components, mean_train_FVs
                ).squeeze(0)
    subspace_components, mean_train_FVs, _ = compute_pca_subspace(
        FVs_dict_2, H2, train_keys
    )
    if H2 != [-1]:
        for key, FV in FVs_dict_2.items():
            pca_FVs_dict_2[key] = compute_project_vectors(
                FV.unsqueeze(0), subspace_components, mean_train_FVs
            ).squeeze(0)
    return {
        key: pca_FVs_dict_1[key] + pca_FVs_dict_2[key] for key in pca_FVs_dict_1.keys()
    }


# Column order of the six feature directions returned by ``fit_mod_vectors``.
# Matches the legacy ``coords_heads.ipynb`` / ``mod_vectors_dict.pth`` artifact
# (targets[:,0]=mod50 ... targets[:,5]=mod25), so a fitted ``mod_vectors`` array
# is column-comparable to the saved ground-truth artifact.
MOD_VECTOR_COLS = ("mod50", "mod2", "mod5", "mod10c", "mod10s", "mod25")
UNIT_COLS = (
    1,
    2,
    3,
    4,
)  # periods 2/5/10 (cos2, cos5, cos10, sin10) → 4D unit-digit subspace
MAGNITUDE_COLS = (0, 5)  # periods 50/25 → 2D magnitude (tens-digit) subspace


def opt_shift(coords, omega, xs, theta_step=0.01):
    """Phase-search helper for ``fit_mod_vectors`` (paper §4.2).

    Returns ``cos(2π·omega·xs + θ*)`` where ``θ*`` minimises the least-squares
    residual of reconstructing the mean-removed cosine from the PC ``coords``.
    Ported verbatim from ``coords_heads.ipynb::fit_head_period``.
    """
    best, best_theta = np.inf, 0.0
    for theta in np.arange(0, 2 * np.pi, theta_step):
        fnc = np.cos(2 * np.pi * omega * xs + theta)
        fnc = fnc - fnc.mean()
        _, resid, _, _ = np.linalg.lstsq(coords, fnc, rcond=None)
        r = (resid[0] / (np.linalg.norm(fnc) ** 2)) if resid.size else np.inf
        if r < best:
            best, best_theta = r, theta
    return np.cos(2 * np.pi * omega * xs + best_theta)


def fit_mod_vectors(head_z_results, ks, n_pcs=6):
    """Faithful port of ``coords_heads.ipynb::fit_head_period`` (paper §4.2/4.3).

    For one head's per-task mean activations ``head_z_results`` (n_tasks, d_model),
    with row ``i`` corresponding to task value ``ks[i]``:

      1. PCA → top-``n_pcs`` subspace ``pc_subspace`` (d_model, n_pcs)  [§4.1].
      2. Project z onto the PCs → ``coords`` (n_tasks, n_pcs), mean-centred.
      3. Fit six periodic target patterns over k (phase via ``opt_shift``):
            mod2  = cos(2πk/2)            mod10c = cos(2πk/10)
            mod5  = cos(2πk/5 + θ*)       mod10s = sin(2πk/10)
            mod25 = cos(2πk/25 + θ*)      mod50  = cos(2πk/50 + θ*)
         by least squares ``coords @ weights ≈ targets``  [§4.2].
      4. ``mod_vectors = pc_subspace @ weights`` (d_model, 6): the six named
         feature directions in activation space; columns 1–4 span the **4-D unit
         subspace** (periods 2/5/10) and columns 0,5 the **2-D magnitude
         subspace** (periods 25/50)  [§4.3].

    Returns a dict: ``mod_vectors`` (d_model, 6), ``pc_subspace``, ``weights``,
    ``targets``, ``targets_hat``, ``target_r2`` (per-direction R² of the fitted
    pattern), ``pca_cumvar_at_npcs`` (variance in the n_pcs-D subspace), and the
    column-index split (``col_names`` / ``unit_cols`` / ``magnitude_cols``).
    """
    if isinstance(head_z_results, torch.Tensor):
        X = head_z_results.detach().cpu().to(torch.float32).numpy().astype("float64")
    else:
        X = np.asarray(head_z_results, dtype="float64")
    xs = np.asarray(ks, dtype="float64")
    n = X.shape[0]

    pca = PCA()
    pca.fit(X)
    pc_subspace = pca.components_.T[:, :n_pcs]  # (d_model, n_pcs)
    cumvar = float(np.cumsum(pca.explained_variance_ratio_)[n_pcs - 1])

    coords = X @ pc_subspace  # (n, n_pcs)
    coords = coords - coords.mean(axis=0)

    targets = np.zeros((n, 6))
    targets[:, 0] = opt_shift(coords, 0.02, xs)  # ~ period 50 (magnitude)
    targets[:, 1] = np.cos(2 * np.pi * xs / 2)  # period 2    (unit)
    targets[:, 2] = opt_shift(coords, 0.2, xs)  # period 5    (unit)
    targets[:, 3] = np.cos(2 * np.pi * xs / 10)  # period 10 cos (unit)
    targets[:, 4] = np.sin(2 * np.pi * xs / 10)  # period 10 sin (unit)
    targets[:, 5] = opt_shift(coords, 0.04, xs)  # ~ period 25 (magnitude)

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
        "targets": targets,
        "targets_hat": targets_hat,
        "target_r2": {name: float(r) for name, r in zip(MOD_VECTOR_COLS, target_r2)},
        "pca_cumvar_at_npcs": cumvar,
        "col_names": list(MOD_VECTOR_COLS),
        "unit_cols": list(UNIT_COLS),
        "magnitude_cols": list(MAGNITUDE_COLS),
    }


def subspace_principal_angles(A, B):
    """Max and mean principal angles (degrees) between the column spans of A, B.

    Each of ``A``/``B`` is (D, r); columns need not be orthonormal. 0° means the
    spans coincide along that direction. Used to compare a fitted subspace to a
    ground-truth one (paper §4 reproduction check).
    """
    A = (
        A.detach().cpu().numpy()
        if isinstance(A, torch.Tensor)
        else np.asarray(A, dtype="float64")
    )
    B = (
        B.detach().cpu().numpy()
        if isinstance(B, torch.Tensor)
        else np.asarray(B, dtype="float64")
    )
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    ang = np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))
    return float(ang.max()), float(ang.mean())


def compute_project_vectors(test_FVs, subspace_components, train_data_mean):
    """Subtract mean → project onto ``subspace_components`` → re-center."""
    train_data_mean_tensor = torch.tensor(
        train_data_mean, dtype=test_FVs.dtype, device=test_FVs.device
    )
    subspace_components_tensor = torch.tensor(
        subspace_components, dtype=test_FVs.dtype, device=test_FVs.device
    )
    centered_test_FVs = test_FVs - train_data_mean_tensor
    projected_test_FVs = torch.mm(centered_test_FVs, subspace_components_tensor)
    reprojected_test_FVs = torch.mm(projected_test_FVs, subspace_components_tensor.t())
    return reprojected_test_FVs + train_data_mean_tensor
