"""Coefficient-matrix helpers: head ranking, threshold selection, post-training extraction."""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from .io import sort_files_by_epoch

# -----------------------------------------------------------------------------
# Low-level: sort matrix entries by |coef| or load a checkpoint
# -----------------------------------------------------------------------------


def find_largest_elements(matrix, device="cuda"):
    """Return ``[(row, col), ...]`` indices of every entry, sorted by |coef| desc."""
    if matrix.device != device:
        matrix = matrix.to(device)
    abs_matrix = torch.abs(matrix)
    flat_indices = torch.argsort(abs_matrix.flatten(), descending=True)
    rows = (flat_indices // matrix.shape[1]).to(device)
    cols = (flat_indices % matrix.shape[1]).to(device)
    return list(zip(rows, cols))


def get_largest_element_matrix(matrix, Nos, val_or_pos, device):
    """Sparse matrix with only the ``Nos``-th-largest positions filled (by value or 1)."""
    if not Nos:
        return None
    pos = find_largest_elements(matrix, device)
    res_pos = [pos[No - 1] for No in Nos]
    res_matrix = torch.zeros_like(matrix).to(device)
    for i, j in res_pos:
        if val_or_pos == "val":
            res_matrix[i, j] = matrix[i, j]
        else:
            res_matrix[i, j] = 1
    return res_matrix


def _validate_checkpoint_dir(checkpoint_dir: str) -> List[str]:
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    checkpoint_files = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pth") and f.startswith("matrix_epoch")
    ]
    if not checkpoint_files:
        raise FileNotFoundError(f"No matrix_epoch*.pth files found in {checkpoint_dir}")
    return checkpoint_files


def load_latest_matrix(log_dir: str) -> Tuple[torch.Tensor, str]:
    """Load the highest-epoch ``matrix_epoch*.pth`` under ``<log_dir>/checkpoints/``."""
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    checkpoint_files = _validate_checkpoint_dir(checkpoint_dir)
    latest_checkpoint = sort_files_by_epoch(checkpoint_files)[-1]
    latest_path = os.path.join(checkpoint_dir, latest_checkpoint)
    matrix = torch.load(latest_path, map_location=torch.device("cpu"))
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"Checkpoint {latest_path} did not contain a torch.Tensor.")
    return matrix.detach().cpu(), latest_path


def extract_heads_above_threshold(
    matrix: torch.Tensor, threshold: float
) -> List[Tuple[Tuple[int, int], float]]:
    """Return ``[((layer, head), coef), ...]`` for entries with ``coef > threshold``."""
    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D coefficient matrix, got shape {tuple(matrix.shape)}"
        )
    mask = matrix > threshold
    rows, cols = torch.nonzero(mask, as_tuple=True)
    heads_with_values: List[Tuple[Tuple[int, int], float]] = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        coeff = float(matrix[row, col].item())
        heads_with_values.append(((row, col), coeff))
    # Stable order: by coefficient descending, then layer/head ascending.
    heads_with_values.sort(key=lambda item: (-item[1], item[0][0], item[0][1]))
    return heads_with_values


# -----------------------------------------------------------------------------
# Automated selection (new — added 2026-05-27 per refactor)
# -----------------------------------------------------------------------------


def _sorted_abs_coefs(matrix: torch.Tensor) -> torch.Tensor:
    return torch.sort(matrix.abs().flatten(), descending=True).values


def auto_threshold(
    matrix: torch.Tensor,
    *,
    method: str = "elbow",
    fraction: float = 0.95,
    floor: float = 1e-6,
    fixed: Optional[float] = None,
) -> float:
    """Pick a head-coefficient threshold from a trained matrix.

    Methods:
        ``"elbow"``     — knee detection on the sorted |coef| curve. The break
                          point is the index that maximizes the perpendicular
                          distance to the chord between the largest and smallest
                          (above-floor) coefficient. Threshold = |coef| at that
                          index. Calibrated so the canonical paper matrix
                          returns ~33 heads at threshold ≈ 0.2.
        ``"fraction"``  — retain the prefix of sorted |coef| capturing
                          ``fraction`` of the total mass. Threshold = the
                          smallest |coef| in that prefix.
        ``"fixed"``     — return ``fixed`` (passthrough, equivalent to manual
                          ``--threshold`` flag).
    """
    if method == "fixed":
        if fixed is None:
            raise ValueError("method='fixed' requires `fixed=<threshold>`")
        return float(fixed)

    coefs = _sorted_abs_coefs(matrix)
    coefs = coefs[coefs > floor]
    if coefs.numel() == 0:
        return float(floor)

    if method == "fraction":
        total = coefs.sum().item()
        cum = torch.cumsum(coefs, dim=0)
        idx = int((cum >= fraction * total).nonzero(as_tuple=True)[0][0].item())
        return float(coefs[idx].item())

    if method == "elbow":
        # Treat the sorted coefficient curve as a 2D point cloud (rank, |coef|),
        # normalize both axes to [0, 1], find the point farthest from the chord
        # joining the first and last point. That index is the knee.
        n = coefs.numel()
        if n < 3:
            return float(coefs[-1].item())
        x = torch.linspace(0.0, 1.0, n, device=coefs.device)
        y = (coefs - coefs.min()) / (coefs.max() - coefs.min() + 1e-12)
        # Distance from each (x_i, y_i) to chord from (0,1) to (1,0):
        # chord is y = 1 - x; perpendicular distance = |y_i - (1 - x_i)| / sqrt(2)
        dist = (y - (1.0 - x)).abs() / (2**0.5)
        idx = int(dist.argmax().item())
        return float(coefs[idx].item())

    raise ValueError(
        f"Unknown method {method!r}; expected one of 'elbow', 'fraction', 'fixed'."
    )


def select_heads(matrix: torch.Tensor, threshold: float) -> List[Tuple[int, int]]:
    """Return ``[(layer, head), ...]`` for ``|coef| > threshold``, sorted by |coef| desc."""
    mask = matrix.abs() > threshold
    rows, cols = torch.nonzero(mask, as_tuple=True)
    coefs = matrix[rows, cols].abs()
    order = torch.argsort(coefs, descending=True)
    return [(int(rows[i].item()), int(cols[i].item())) for i in order.tolist()]


def _load_per_head_accuracy_dict(
    matrix_dir: str, n_shot: int = 5, mode: str = "owTomean"
) -> dict:
    """Load ``head_acc_dict.pth`` from ``matrix_dir`` and extract per-head accuracies.

    The file is keyed by strings like ``"[(L, H)]_[N_SHOT]_owTomean"`` with
    values ``[acc_indist, acc_ood]``. We return ``{(L, H): acc_indist}`` for
    all single-head entries matching ``n_shot`` and ``mode``. Missing file
    or zero matches → empty dict (caller falls back).
    """
    import os
    import re

    path = os.path.join(matrix_dir, "head_acc_dict.pth")
    if not os.path.exists(path):
        return {}
    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        return {}
    pattern = re.compile(r"^\[\((\d+),\s*(\d+)\)\]_\[(\d+)\]_(\w+)$")
    out = {}
    for key, val in raw.items():
        m = pattern.match(str(key))
        if not m:
            continue
        L, H, n, k_mode = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        if n != n_shot or k_mode != mode:
            continue
        if isinstance(val, (list, tuple)) and len(val) > 0:
            out[(L, H)] = float(val[0])
        elif isinstance(val, (int, float)):
            out[(L, H)] = float(val)
    return out


def auto_main_heads(
    matrix: torch.Tensor,
    *,
    k: Optional[int] = None,
    rank_by: str = "recovery",
    recovery_decisions: Optional[dict] = None,
    rho_floor: float = 0.3,
    matrix_dir: Optional[str] = None,
    n_shot: int = 5,
    accuracy_floor: Optional[float] = None,
    accuracy_fn: Optional[Callable[[List[Tuple[int, int]]], float]] = None,
    fallback: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """Pick the main-head subset from a trained matrix.

    Modes:
        ``rank_by="recovery"`` *(new default)* — data-driven; cardinality not
            fixed. Requires ``recovery_decisions``, the dict produced by
            :func:`select_main_heads_by_recovery_weak`. Selects heads whose
            dose-response rises steadily from the c=0 leave-one-out baseline by
            at least the adaptive gain bar.
            See ``select_main_heads_by_recovery_weak`` for the test details.
        ``rank_by="accuracy"`` *(legacy, paper repro)* — top-``k`` by per-head
            individual accuracy from ``<matrix_dir>/head_acc_dict.pth``.
            Requires ``k`` and ``matrix_dir``. Calibrated: paper matrix
            returns ``[(13,6),(15,2),(15,1)]`` at ``k=3``.
        ``rank_by="coef"`` *(legacy, cheap)* — top-``k`` by ``|coef|``.
        ``accuracy_floor`` *(alternate legacy)* — smallest |coef|-sorted
            prefix whose joint accuracy under ``accuracy_fn`` clears the
            floor; requires ``accuracy_fn``.

    A ``fallback`` sequence, if non-empty, overrides every mode — used to
    pin the paper-exact set verbatim during reproduction.
    """
    if fallback is not None and len(fallback) > 0:
        return [(int(l), int(h)) for l, h in fallback]

    if rank_by == "recovery":
        if recovery_decisions is None:
            raise ValueError(
                "rank_by='recovery' requires `recovery_decisions`. Build it via "
                "subspaces.utils.heads.recovery_scan + select_main_heads_by_recovery_weak, "
                "or pass rank_by='accuracy' (k>0, matrix_dir set) for the "
                "legacy paper-repro path."
            )
        main = [
            (int(pos[0]), int(pos[1]))
            for pos, d in recovery_decisions.items()
            if d.get("is_main", False)
        ]
        # Stable order: by top accuracy across scales (acc*) desc, then layer/head asc.
        main.sort(
            key=lambda lh: (
                -recovery_decisions[lh].get("acc_at_c_star", 0.0),
                lh[0],
                lh[1],
            )
        )
        if k is not None and k > 0:
            main = main[:k]
        return main

    if accuracy_floor is not None:
        if accuracy_fn is None:
            raise ValueError("accuracy_floor requires `accuracy_fn`")
        sorted_heads = select_heads(matrix, threshold=0.0)
        for n in range(1, len(sorted_heads) + 1):
            prefix = sorted_heads[:n]
            if accuracy_fn(prefix) >= accuracy_floor:
                return prefix
        return sorted_heads

    if k is None or k <= 0:
        raise ValueError(
            "Legacy rank_by={!r} requires k>0. Use rank_by='recovery' "
            "(data-driven, no fixed k) for the new pipeline.".format(rank_by)
        )

    if rank_by == "accuracy":
        acc_dict = (
            _load_per_head_accuracy_dict(matrix_dir, n_shot=n_shot)
            if matrix_dir
            else {}
        )
        if not acc_dict:
            raise ValueError(
                "rank_by='accuracy' needs <matrix_dir>/head_acc_dict.pth (run "
                "subspaces.runners.run_head_eval first to populate it). "
                "Falling forward to rank_by='coef' is unsafe — request it explicitly."
            )
        ranked = sorted(acc_dict.items(), key=lambda kv: -kv[1])
        return [pos for pos, _acc in ranked[:k]]

    if rank_by == "coef":
        return select_heads(matrix, threshold=0.0)[:k]

    raise ValueError(f"Unknown rank_by={rank_by!r}")


# -----------------------------------------------------------------------------
# Recovery scan — dose-response per selected head, used by the new
# data-driven main-head selector.
# -----------------------------------------------------------------------------


def recovery_scan(
    *,
    selected_heads: Sequence[Tuple[int, int]],
    mean_z: torch.Tensor,
    z_results: dict,
    c_grid: Sequence[int],
    model,
    tasks: dict,
    task_names: Sequence[str],
    n_shot: int,
    test_limit: int,
    bs: int,
    layer_name: str,
    device: torch.device,
    verbose: bool = True,
    eval_seed: int = 42,
) -> Tuple[dict, dict, float, dict]:
    """Per-head dose-response scan with the *scaled head kept task-conditioned*.

    For each selected head ``(l, h)`` and integer scalar ``c`` in ``c_grid``, and for
    each task ``t``, build

        ``FV(c, h, t) = c · z_results[t][l, h] + Σ_{h' ≠ h ∈ selected} mean_z[h']``

    i.e. the *scaled* head keeps its **original per-task value** (scaled by
    ``c``) while every other selected head is held at its **cross-task mean**
    (mean-ablation). Because the scaled head is task-conditioned, the FV varies
    per task and ``c = 1`` is **no longer a global invariant** — each head has
    its own natural-scale (``c = 1``) accuracy. ``c = 1`` is that head's natural
    magnitude (reference for "does scaling up help"); ``c = 0`` is leave-one-out
    (head removed, all others at mean).

    Precompute ``mean_fv = Σ_{h ∈ selected} mean_z[h]`` once; the others-term for head
    ``(l, h)`` is then ``mean_fv − mean_z[l, h]``.

    We seed ``random`` deterministically *per task* before each call so the same
    per-example prompts are drawn for every ``(h, c)`` pair — this preserves the
    paired-bootstrap structure of the per-example Δ test. Per-example 0/1
    correctness vectors are kept so the downstream paired bootstrap is ~free.

    Returns:
        per_example_correctness: ``{(L, H): {c: [0/1, ...]}}`` (per-example 0/1).
        per_head_curves: ``{(L, H): {c: mean_acc}}`` (aggregated dose-response).
        mean_fv_acc: scalar accuracy of the global mean-FV (all selected heads at
            their cross-task mean) — a diagnostic, *not* the per-head c=1.
        meta: bookkeeping (selected head count, c grid, etc.).
    """
    import random as _random

    from .intervene import eval_fv_per_example_correctness

    selected_heads_t: List[Tuple[int, int]] = [
        (int(l), int(h)) for (l, h) in selected_heads
    ]
    mean_z = mean_z.to(device)
    z_dev = {tn: z_results[tn].to(device) for tn in task_names}
    c_grid = list(c_grid)
    if 1 not in c_grid:
        raise ValueError(
            f"c=1 must be in c_grid (natural-scale reference). Got c_grid={c_grid}."
        )

    mean_fv = torch.zeros(mean_z.shape[-1], device=device, dtype=mean_z.dtype)
    for l, h in selected_heads_t:
        mean_fv = mean_fv + mean_z[l, h]

    # Global mean-FV diagnostic — all selected heads held at their cross-task mean.
    fv_meanall = {tn: mean_fv for tn in task_names}
    meanall_corr: List[int] = []
    for ti, tn in enumerate(task_names):
        _random.seed(eval_seed + ti)
        meanall_corr.extend(
            eval_fv_per_example_correctness(
                tn,
                tasks,
                n_shot,
                test_limit,
                bs,
                fv_meanall,
                model,
                layer_name,
            )
        )
    mean_fv_acc = sum(meanall_corr) / max(len(meanall_corr), 1)
    if verbose:
        print(
            f"[recovery_scan] global mean-FV acc (all selected at cross-task mean): "
            f"{mean_fv_acc:.4f} (n={len(meanall_corr)})",
            flush=True,
        )

    per_example: dict = {}
    per_head_curves: dict = {}
    for hi, (l, h) in enumerate(selected_heads_t):
        others_term = mean_fv - mean_z[l, h]
        per_example[(l, h)] = {}
        per_head_curves[(l, h)] = {}
        for c in c_grid:
            # Task-conditioned scaled head + mean-ablated others.
            fv_dict = {tn: c * z_dev[tn][l, h] + others_term for tn in task_names}
            corr_flat: List[int] = []
            for ti, tn in enumerate(task_names):
                _random.seed(eval_seed + ti)
                corr_flat.extend(
                    eval_fv_per_example_correctness(
                        tn,
                        tasks,
                        n_shot,
                        test_limit,
                        bs,
                        fv_dict,
                        model,
                        layer_name,
                    )
                )
            per_example[(l, h)][c] = corr_flat
            per_head_curves[(l, h)][c] = float(sum(corr_flat) / max(len(corr_flat), 1))
        if verbose:
            pretty = ", ".join(
                f"c={c}:{per_head_curves[(l, h)][c]:.3f}" for c in c_grid
            )
            print(
                f"[recovery_scan] head ({l},{h}) [{hi + 1}/{len(selected_heads_t)}]: {pretty}",
                flush=True,
            )

    meta = {
        "mean_fv_acc": float(mean_fv_acc),
        "n_selected_heads": len(selected_heads_t),
        "c_grid": list(c_grid),
        "n_eval_examples_per_head_per_c": len(meanall_corr),
        "scaled_head_vector": "task_conditioned",
    }
    return per_example, per_head_curves, float(mean_fv_acc), meta


def select_main_heads_by_recovery_weak(
    per_head_curves: dict,
    *,
    full_selected_fv_acc: float,
    rel_floor: float = 0.2,
    abs_floor: float = 0.01,
    eps: float = 0.01,
) -> Tuple[List[Tuple[int, int]], dict]:
    """Data-driven main-head selection from a task-conditioned recovery scan.

    This is the project's single recovery-mode selector (it replaced the earlier
    strict BH-FDR / ρ-vs-clean / band-width selector). The intent: a head is
    "main" *as long as scaling it can steadily improve accuracy above its
    leave-one-out (c=0) baseline*. The gain bar is set **adaptively** from two
    references (per the task design):

        * ``acc0 = acc_h(c=0)`` — the head's leave-one-out accuracy (head removed,
          every other selected head held at its cross-task mean);
        * ``A_selected = full_selected_fv_acc`` — accuracy of the *sum of all
          selected heads at unit coefficient* (the full-selected-set FV). This is
          how far the whole localized set gets; ``headroom = A_selected − acc0`` is
          the room a single head could plausibly recover.

    For head ``h`` with dose-response curve ``acc_h(c)`` over the scan grid:
        ``c* = argmax_c acc_h(c)``, ``acc* = acc_h(c*)``, ``gain = acc* − acc0``.

    A head is main iff **all** of:
        (1) ``c* > 0`` — the peak comes from scaling the head *up*, not from
            deleting it (a head best-left-out is not contributing signal);
        (2) **steady rise** — no drop greater than ``eps`` anywhere along the
            segment ``[0 .. c*]`` (scaling consistently helps on the way to the
            peak; post-peak decline under over-scaling is ignored);
        (3) ``gain ≥ max(abs_floor, rel_floor · headroom)`` — the head recovers
            at least ``rel_floor`` of the selected-set's headroom above
            leave-one-out, **and** at least ``abs_floor`` in absolute terms. The
            absolute guard prevents near-chance drift from qualifying on tasks
            where ``A_selected`` is tiny (so ``rel_floor · headroom`` ≈ 0).

    Heads are ordered by ``acc*`` (top accuracy across scales) descending, then
    by ``gain``, then layer/head. Returns ``(main_heads, decisions)`` with
    ``decisions`` keyed by ``(L, H)``.
    """
    main_heads: List[Tuple[int, int]] = []
    decisions: dict = {}
    A_selected = float(full_selected_fv_acc)

    for (l, h), curve in per_head_curves.items():
        cs = sorted(curve.keys(), key=lambda c: int(c))  # robust to int or str keys
        accs = [float(curve[c]) for c in cs]
        i_star = int(max(range(len(accs)), key=lambda i: accs[i]))
        c_star = int(cs[i_star])
        acc_star = accs[i_star]
        acc0 = accs[0]
        gain = acc_star - acc0

        seg = accs[: i_star + 1]
        max_drop = max((seg[i] - seg[i + 1] for i in range(len(seg) - 1)), default=0.0)
        steady = max_drop <= eps

        headroom = A_selected - acc0
        bar = max(abs_floor, rel_floor * headroom) if headroom > 0 else abs_floor
        rel = (gain / headroom) if abs(headroom) > 1e-9 else float("nan")

        is_main = bool(c_star != 0 and steady and gain >= bar)

        flags: List[str] = []
        if c_star == 0:
            flags.append("peak_at_c=0_leave_one_out")
        if not steady:
            flags.append("not_steady_to_peak")
        if gain < bar:
            flags.append("gain_below_bar")

        decisions[(int(l), int(h))] = {
            "c_star": int(c_star),
            "acc_at_c_star": acc_star,
            "acc_at_c0": acc0,
            "gain": float(gain),
            "rel_recovery": float(rel) if rel == rel else None,  # NaN→None
            "headroom_vs_sumSelected": float(headroom),
            "bar": float(bar),
            "max_drop_to_peak": float(max_drop),
            "steady": bool(steady),
            "is_main": is_main,
            "main_reason": "steady_scaling_recovery" if is_main else None,
            "flags": flags,
        }
        if is_main:
            main_heads.append((int(l), int(h)))

    # Order by top accuracy across scales (acc*), then gain, then layer/head.
    main_heads.sort(
        key=lambda lh: (
            -decisions[lh]["acc_at_c_star"],
            -decisions[lh]["gain"],
            lh[0],
            lh[1],
        )
    )
    return main_heads, decisions


def select_main_heads_by_recovery_weak(
    per_head_curves: dict,
    *,
    full_selected_fv_acc: float,
    rel_floor: float = 0.2,
    abs_floor: float = 0.05,
    eps: float = 0.03,
) -> Tuple[List[Tuple[int, int]], dict]:
    """Weaker, data-driven main-head selection from the same recovery scan.

    Looser sibling of :func:`select_main_heads_by_recovery`. The intent: a head
    is "main" *as long as scaling it can steadily improve accuracy above its
    leave-one-out (c=0) baseline* — dropping the strict rule's BH-FDR
    significance, ρ-vs-clean, and band-width plateau gates. The gain bar is set
    **adaptively** from two references (per the task design):

        * ``acc0 = acc_h(c=0)`` — the head's leave-one-out accuracy (head removed,
          every other selected head held at its cross-task mean);
        * ``A_selected = full_selected_fv_acc`` — accuracy of the *sum of all
          selected heads at unit coefficient* (the full-selected-set FV). This is
          how far the whole localized set gets; ``headroom = A_selected − acc0`` is
          the room a single head could plausibly recover.

    For head ``h`` with dose-response curve ``acc_h(c)`` over the scan grid:
        ``c* = argmax_c acc_h(c)``, ``acc* = acc_h(c*)``, ``gain = acc* − acc0``.

    A head is main iff **all** of:
        (1) ``c* > 0`` — the peak comes from scaling the head *up*, not from
            deleting it (a head best-left-out is not contributing signal);
        (2) **steady rise** — no drop greater than ``eps`` anywhere along the
            segment ``[0 .. c*]`` (scaling consistently helps on the way to the
            peak; post-peak decline under over-scaling is ignored);
        (3) ``gain ≥ max(abs_floor, rel_floor · headroom)`` — the head recovers
            at least ``rel_floor`` of the selected-set's headroom above
            leave-one-out, **and** at least ``abs_floor`` in absolute terms. The
            absolute guard prevents near-chance drift from qualifying on tasks
            where ``A_selected`` is tiny (so ``rel_floor · headroom`` ≈ 0).

    Heads are ordered by ``acc*`` (top accuracy across scales) descending, then
    by ``gain``, then layer/head. Returns ``(main_heads, decisions)`` with
    ``decisions`` keyed by ``(L, H)`` (same shape contract as the strict
    selector, plus ``rel_recovery``/``headroom_vs_sumSelected``/``bar`` fields).
    """
    main_heads: List[Tuple[int, int]] = []
    decisions: dict = {}
    A_selected = float(full_selected_fv_acc)

    for (l, h), curve in per_head_curves.items():
        cs = sorted(curve.keys(), key=lambda c: int(c))  # robust to int or str keys
        accs = [float(curve[c]) for c in cs]
        i_star = int(max(range(len(accs)), key=lambda i: accs[i]))
        c_star = int(cs[i_star])
        acc_star = accs[i_star]
        acc0 = accs[0]
        gain = acc_star - acc0

        seg = accs[: i_star + 1]
        max_drop = max((seg[i] - seg[i + 1] for i in range(len(seg) - 1)), default=0.0)
        steady = max_drop <= eps

        headroom = A_selected - acc0
        bar = max(abs_floor, rel_floor * headroom) if headroom > 0 else abs_floor
        rel = (gain / headroom) if abs(headroom) > 1e-9 else float("nan")

        is_main = bool(c_star != 0 and steady and gain >= bar)

        flags: List[str] = []
        if c_star == 0:
            flags.append("peak_at_c=0_leave_one_out")
        if not steady:
            flags.append("not_steady_to_peak")
        if gain < bar:
            flags.append("gain_below_bar")

        decisions[(int(l), int(h))] = {
            "c_star": int(c_star),
            "acc_at_c_star": acc_star,
            "acc_at_c0": acc0,
            "gain": float(gain),
            "rel_recovery": float(rel) if rel == rel else None,  # NaN→None
            "headroom_vs_sumSelected": float(headroom),
            "bar": float(bar),
            "max_drop_to_peak": float(max_drop),
            "steady": bool(steady),
            "is_main": is_main,
            "main_reason": "steady_scaling_recovery" if is_main else None,
            "flags": flags,
        }
        if is_main:
            main_heads.append((int(l), int(h)))

    # Order by top accuracy across scales (acc*), then gain, then layer/head.
    main_heads.sort(
        key=lambda lh: (
            -decisions[lh]["acc_at_c_star"],
            -decisions[lh]["gain"],
            lh[0],
            lh[1],
        )
    )
    return main_heads, decisions
