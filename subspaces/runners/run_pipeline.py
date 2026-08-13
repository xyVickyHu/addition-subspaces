"""End-to-end pipeline that runs every paper analysis phase for a (model, task).

Stages:
    1. Matrix training (skipped if a matching matrix dir is on disk).
    2. Auto threshold + main-head selection, three FV-variant accuracy
       evaluations (full significant FV, per-head scaled-with-mean-ablation,
       main+mean-ablation).
    3. PCA decomposition per significant head + paper §4 period/magnitude
       basis fits on the main heads; cumulative-variance plots are saved.
    4. Paper §5 — per-demo extracted-signal analysis on the main heads:
       §5.1/5.2 label-token peaking + signal alignment and the label-token
       aggregation analysis. Uses :mod:`subspaces.utils.signals`; needs GPU
       forward passes; add-k family only.
    5. Appends a structured section to ``subspaces/experiments.md`` summarising the
       run with relative-path links to plots and JSON artifacts.

Reproduce paper baseline:
    python -m subspaces.run_pipeline \\
      --model_name meta-llama/Meta-Llama-3-8B-Instruct \\
      --task_dir number_add \\
      --reuse-matrix log/[0,1]_matrix_add_0204_clip_lambda0.05

Run on a new model / task: same command with different flags; defaults to
the paper baseline.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from subspaces.utils.activations import (
    compute_mean_z_result,
    load_z_results_dict,
)
from subspaces.utils.heads import (
    auto_main_heads,
    auto_threshold,
    load_latest_matrix,
    recovery_scan,
    select_heads,
    select_main_heads_by_recovery_weak,
)
from subspaces.utils.io import load_task_data
from subspaces.utils.model import load_model_no_grad


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--task_dir", default="number_add")
    p.add_argument("--layer_name", default="blocks.10.hook_resid_mid")
    p.add_argument("--n_shot", type=int, default=5)
    p.add_argument("--n_example", type=int, default=100)
    p.add_argument(
        "--reuse-matrix",
        default=None,
        help="Path to a matrix dir; skips Stage 1 training.",
    )
    p.add_argument(
        "--auto-threshold", default="elbow", choices=("elbow", "fraction", "fixed")
    )
    p.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.2,
        help="Used when --auto-threshold=fixed.",
    )
    p.add_argument(
        "--main-rank-by",
        default="recovery",
        choices=("recovery", "accuracy", "coef"),
        help=(
            "How to pick main heads. 'recovery' (default, data-driven, no fixed k) "
            "runs a dose-response scan over c={0..C_max} and selects heads whose "
            "curve rises steadily from the c=0 leave-one-out baseline by at least "
            "the adaptive gain bar max(--weak-abs-floor, --weak-rel-floor·(A_sig−acc0)) "
            "(see select_main_heads_by_recovery_weak). 'accuracy' is the legacy "
            "paper-repro path (top-k by head_acc_dict.pth, needs run_head_eval first). "
            "'coef' is top-k by |coef|."
        ),
    )
    p.add_argument(
        "--main-k",
        type=int,
        default=None,
        help="Cap on main-head count. Default None = no cap (data-driven). "
        "Required if --main-rank-by in {accuracy,coef}.",
    )
    p.add_argument(
        "--weak-rel-floor",
        type=float,
        default=0.2,
        help="Recovery (weak) selector: a head's gain acc(c*)−acc(c=0) must "
        "recover ≥ this fraction of the significant-set headroom "
        "(A_sig − acc(c=0)). Default 0.2.",
    )
    p.add_argument(
        "--weak-abs-floor",
        type=float,
        default=0.01,
        help="Recovery (weak) selector: absolute floor on gain acc(c*)−acc(c=0); "
        "guards low-A_sig tasks where rel_floor·headroom ≈ 0. Default 0.01.",
    )
    p.add_argument(
        "--weak-eps",
        type=float,
        default=0.01,
        help="Recovery (weak) selector: max tolerated single-step dip along the "
        "rising segment [0..c*] for the curve to count as steady. Default 0.01.",
    )
    p.add_argument(
        "--rho-floor",
        type=float,
        default=0.3,
        help="DEPRECATED / ignored — accepted for backward compatibility with "
        "older launchers. The recovery selector is now the weak one "
        "(no ρ-vs-clean gate). Use --weak-rel-floor / --weak-abs-floor.",
    )
    p.add_argument(
        "--clean-floor",
        type=float,
        default=0.2,
        help="DEPRECATED / ignored by the weak recovery selector (no clean-floor "
        "guard). Accepted for backward compatibility with older launchers.",
    )
    p.add_argument(
        "--recovery-c-max",
        type=int,
        default=12,
        help="Recovery scan sweeps c in {0, 1, ..., recovery-c-max}.",
    )
    p.add_argument(
        "--recovery-test-limit",
        type=int,
        default=10,
        help="Per-task examples used in the recovery scan only "
        "(final FV evals use --test-limit). Default 10.",
    )
    p.add_argument(
        "--recovery-n-boot",
        type=int,
        default=2000,
        help="DEPRECATED / ignored — the weak recovery selector uses no bootstrap. "
        "Accepted for backward compatibility with older launchers.",
    )
    p.add_argument(
        "--main-heads-fallback",
        default=None,
        help="Comma-separated 'L:H,L:H,...' to pin specific main heads (e.g. '15:2,15:1,13:6'); "
        "overrides --main-rank-by.",
    )
    p.add_argument(
        "--scalar-sweep",
        default="0,1,2,4,6",
        help="(Legacy) comma-separated scalar values for the post-selection "
        "per-main-head curve. Not used by --main-rank-by=recovery, which "
        "uses --recovery-c-max instead.",
    )
    p.add_argument(
        "--signal-n-prompts-per-k",
        type=int,
        default=40,
        help="Stage 4 (§5): add-k prompts sampled per k for the extracted-signal analysis.",
    )
    p.add_argument(
        "--signal-k-max",
        type=int,
        default=30,
        help="Stage 4 (§5): sweep k=1..signal-k-max (paper: 30).",
    )
    p.add_argument(
        "--signal-n-comp",
        type=int,
        default=6,
        help="Stage 4 (§5): PCA subspace dim for the signal projection (paper: 6).",
    )
    p.add_argument("--test-limit", type=int, default=10, help="Eval samples per task.")
    p.add_argument(
        "--bs", type=int, default=None, help="Eval batch size (default test-limit)."
    )
    p.add_argument(
        "--lambdaL1",
        type=float,
        default=0.05,
        help="Used only for Stage 1 when training.",
    )
    p.add_argument(
        "--epoch-num", type=int, default=50, help="Used only for Stage 1 when training."
    )
    p.add_argument(
        "--stages",
        default="1,2,3,4,5",
        help="Comma-separated stage indices to run. Default runs all five.",
    )
    p.add_argument(
        "--rerun",
        action="store_true",
        help="Re-execute even if a matching manifest is present.",
    )
    p.add_argument(
        "--run-tag",
        default=None,
        help="Optional suffix for the run-dir slug (e.g. 'refactor' to separate "
        "a from-scratch retrain run from a reuse-matrix run for the same model+task).",
    )
    return p.parse_args()


def slugify(model_name: str, task_dir: str, run_tag: Optional[str] = None) -> str:
    model_slug = model_name.replace("/", "_").lower()
    base = f"{model_slug}__{task_dir}"
    if run_tag:
        return f"{base}__{run_tag}"
    return base


def append_experiments_md(experiments_md: Path, section: str) -> None:
    """Append ``section`` to ``experiments.md`` (creates with header if absent)."""
    if not experiments_md.exists():
        experiments_md.write_text(
            "# subspaces pipeline — experiments log\n\n"
            "Each run produced by `subspaces.run_pipeline` appends a section below.\n\n"
            "---\n\n"
        )
    with open(experiments_md, "a") as f:
        f.write(section)


def stage1_train_or_reuse(args, run_dir: Path) -> Tuple[Path, bool]:
    """Returns (matrix_dir, reused_flag). If ``--reuse-matrix`` is set or a
    canonical matrix already exists for this (model, task, lambda) combo, no
    training is done. Otherwise calls ``subspaces.train_matrix.main`` via subprocess
    (to inherit its argparse + wandb wiring).
    """
    if args.reuse_matrix:
        matrix_dir = PROJECT_ROOT / args.reuse_matrix
        return matrix_dir, True

    # Convention-canonical matrix path for this (model, task, lambda). The task
    # tag is 'add' for every number_add* variant (backward-compatible with the
    # existing `[0,1]_matrix_add_<model>_lambda<L>` dirs), and the task_dir name
    # otherwise so non-add tasks (number_mul, abstractive, ...) get distinct dirs
    # instead of all colliding on the same 'add'-named path.
    task_tag = "add" if args.task_dir.startswith("number_add") else args.task_dir
    canonical = (
        PROJECT_ROOT
        / "log"
        / (
            f"[0,1]_matrix_{task_tag}_{args.model_name.split('/')[-1].lower()}_lambda{args.lambdaL1}"
        )
    )
    if canonical.exists() and (canonical / "checkpoints").exists():
        return canonical, True

    # Train. Defer to subspaces.train_matrix.main via os.system so wandb side-effects
    # match the legacy training flow exactly.
    version = canonical.name
    cmd = (
        f"python -m subspaces.runners.train_matrix --version '{version}' --task_dir '{args.task_dir}' "
        f"--model_name '{args.model_name}' --layer_name '{args.layer_name}' "
        f"--lambdaL1 {args.lambdaL1} --epoch_num {args.epoch_num} "
        f"--n_shot {args.n_shot} --n_example {args.n_example} "
        f"--train_ratio 0.8 --bs 128 --zero_one_interval"
    )
    print(f"[stage 1] training: {cmd}")
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"matrix training exited with {rc}")
    return canonical, False


def stage2_head_selection_and_eval(
    args, matrix_dir: Path, run_dir: Path, device: torch.device
) -> Dict:
    """Stage 2 — head selection + FV-variant accuracy evaluations.

    Phases:
        (clean) clean greedy-decode baseline (ceiling reference; never subtracted).
        (a) FV = Σ z[l,h] over all significant heads (unit coefs; `sum_33` config).
        (a') Raw trained-coefficient FV = Σ_{h ∈ sig} M[l,h]·z[l,h] (the trained
            matrix entries used as coefficients; cf. (a) which uses unit coefs on
            the same head set). Recorded as `raw_coef_fv_acc`.
        (b) Recovery scan: per significant head ``(l, h)`` sweep ``c ∈ {0..C_max}``
            with ``FV(c,h,t) = c·z[l,h]_task + Σ_{h'≠h ∈ sig} mean_z[h']`` — the
            *scaled head keeps its task-conditioned per-task value*, others are
            mean-ablated. ``c=0`` is leave-one-out. Decision rule for main heads
            (weak selector): ``c* > 0`` AND the curve rises steadily to its peak
            (no single-step dip > --weak-eps on [0..c*]) AND
            ``gain = acc(c*) − acc(c=0) ≥ max(--weak-abs-floor, --weak-rel-floor·(A_sig − acc(c=0)))``,
            where ``A_sig`` is the full-significant-FV accuracy.
            See ``subspaces.utils.heads.select_main_heads_by_recovery_weak``.
        (c) Main + mean-ablation: ``FV = Σ_{h ∈ main} z[l,h] + Σ_{h ∈ sig ∖ main} mean_z[h]``.
    """
    from subspaces.utils.intervene import compute_task_accuracy

    matrix, ckpt = load_latest_matrix(str(matrix_dir))
    threshold = auto_threshold(
        matrix, method=args.auto_threshold, fixed=args.fixed_threshold
    )
    significant = select_heads(matrix, threshold)

    print(
        f"[stage 2] threshold={threshold:.4f} (method={args.auto_threshold}); |significant|={len(significant)}"
    )

    tasks = load_task_data(str(PROJECT_ROOT / "dataset_files" / args.task_dir))
    savevar_dir = _resolve_savevar_dir(matrix_dir, args.task_dir, args.n_shot)
    savevar_dir.mkdir(parents=True, exist_ok=True)

    model, _, _ = load_model_no_grad(args.model_name, device)
    z_results = load_z_results_dict(
        model,
        tasks,
        args.n_shot,
        args.n_example,
        str(savevar_dir),
        args.task_dir,
        device=device,
    )

    test_limit = args.test_limit
    bs = args.bs or test_limit
    task_names = sorted(tasks.keys())
    mean_z = compute_mean_z_result(z_results).to(device)

    # ---- (clean) ceiling baseline ----
    accs_clean: List[float] = []
    for tn in task_names:
        clean, _, _, _, _, *_ = compute_task_accuracy(
            tn,
            tasks,
            args.n_shot,
            test_limit,
            bs,
            None,
            None,
            None,
            None,
            model,
            args.layer_name,
            intervene=False,
            clean=True,
            corrupted=False,
        )
        accs_clean.append(clean)
    clean_acc = float(sum(accs_clean) / len(accs_clean))
    print(f"[stage 2-clean] mean clean accuracy = {clean_acc:.4f}")

    # ---- (a) Full-significant FV (sum_33 config, unit coefs over task-conditioned z) ----
    fvs_full = {}
    for tn in task_names:
        z = z_results[tn].to(device)
        fv = torch.zeros(z.shape[-1], dtype=z.dtype, device=device)
        for l, h in significant:
            fv = fv + z[l, h]
        fvs_full[tn] = fv
    accs_full: List[float] = []
    for tn in task_names:
        _, _, intv, _, _, *_ = compute_task_accuracy(
            tn,
            tasks,
            args.n_shot,
            test_limit,
            bs,
            fvs_full,
            None,
            None,
            None,
            model,
            args.layer_name,
            intervene=True,
            clean=False,
            corrupted=False,
        )
        accs_full.append(intv)
    full_acc = float(sum(accs_full) / len(accs_full))
    print(f"[stage 2a] full-significant FV mean intervention acc = {full_acc:.4f}")

    # ---- (a') Raw trained-coefficient FV (Σ_{h ∈ sig} M[l,h]·z[l,h]_task) ----
    # Same significant head set as (a), but coefficients are the *trained matrix
    # entries* M[l,h] instead of unit. M[l,h] are training-time SNR weights; the
    # paper's FV recipe uses unit coefs (variant a). This row records what the
    # raw trained coefficients alone recover.
    fvs_raw = {}
    for tn in task_names:
        z = z_results[tn].to(device)
        fv = torch.zeros(z.shape[-1], dtype=z.dtype, device=device)
        for l, h in significant:
            fv = fv + float(matrix[l, h]) * z[l, h]
        fvs_raw[tn] = fv
    accs_raw: List[float] = []
    for tn in task_names:
        _, _, intv, _, _, *_ = compute_task_accuracy(
            tn,
            tasks,
            args.n_shot,
            test_limit,
            bs,
            fvs_raw,
            None,
            None,
            None,
            model,
            args.layer_name,
            intervene=True,
            clean=False,
            corrupted=False,
        )
        accs_raw.append(intv)
    raw_coef_acc = float(sum(accs_raw) / len(accs_raw))
    print(f"[stage 2a'] raw trained-coef FV mean intervention acc = {raw_coef_acc:.4f}")

    # ---- (b) Recovery scan + data-driven main heads ----
    # Optional manual override (paper-exact reproduction).
    fallback_heads = None
    if args.main_heads_fallback:
        fallback_heads = [
            tuple(int(x) for x in tok.split(":"))
            for tok in args.main_heads_fallback.split(",")
        ]

    recovery_per_example: Dict = {}
    recovery_curves: Dict = {}
    recovery_meta: Dict = {}
    recovery_decisions: Dict = {}
    mean_fv_acc: float = float("nan")
    if fallback_heads is not None:
        main_heads = auto_main_heads(matrix, fallback=fallback_heads)
        print(
            f"[stage 2b] FALLBACK pin: main heads = {main_heads} (paper-exact override)"
        )
    elif args.main_rank_by == "recovery":
        c_grid = list(range(0, args.recovery_c_max + 1))
        rec_test_limit = args.recovery_test_limit or test_limit
        rec_bs = min(rec_test_limit, bs) if bs else rec_test_limit
        print(
            f"[stage 2b] recovery scan over c={c_grid}, test_limit={rec_test_limit}, "
            f"bs={rec_bs}, {len(significant)} sig heads -> "
            f"~{(1 + len(significant) * (len(c_grid) - 1)) * len(task_names)} task evals"
        )
        recovery_per_example, recovery_curves, mean_fv_acc, recovery_meta = (
            recovery_scan(
                sig_heads=significant,
                mean_z=mean_z,
                z_results=z_results,
                c_grid=c_grid,
                model=model,
                tasks=tasks,
                task_names=task_names,
                n_shot=args.n_shot,
                test_limit=rec_test_limit,
                bs=rec_bs,
                layer_name=args.layer_name,
                device=device,
            )
        )
        main_heads, recovery_decisions = select_main_heads_by_recovery_weak(
            recovery_curves,
            full_significant_fv_acc=full_acc,
            rel_floor=args.weak_rel_floor,
            abs_floor=args.weak_abs_floor,
            eps=args.weak_eps,
        )
        print(
            f"[stage 2b] recovery(weak)-selected {len(main_heads)} main heads "
            f"(rel_floor={args.weak_rel_floor}, abs_floor={args.weak_abs_floor}, "
            f"eps={args.weak_eps}, A_sig={full_acc:.4f}): {main_heads}"
        )
    else:
        # Legacy ranking modes (paper-repro path).
        main_heads = auto_main_heads(
            matrix,
            k=args.main_k,
            rank_by=args.main_rank_by,
            matrix_dir=str(matrix_dir),
        )
        print(
            f"[stage 2b] legacy main heads (rank_by={args.main_rank_by}, k={args.main_k}): {main_heads}"
        )

    # ---- (c) Main + mean-ablation FV ----
    fv_main = {}
    main_set = {(int(l), int(h)) for (l, h) in main_heads}
    for tn in task_names:
        base = torch.zeros_like(z_results[tn][0, 0]).to(device)
        for l, h in significant:
            if (l, h) in main_set:
                base = base + z_results[tn][l, h].to(device)
            else:
                base = base + mean_z[l, h]
        fv_main[tn] = base
    accs_main: List[float] = []
    for tn in task_names:
        _, _, intv, _, _, *_ = compute_task_accuracy(
            tn,
            tasks,
            args.n_shot,
            test_limit,
            bs,
            fv_main,
            None,
            None,
            None,
            model,
            args.layer_name,
            intervene=True,
            clean=False,
            corrupted=False,
        )
        accs_main.append(intv)
    main_meanab_acc = float(sum(accs_main) / len(accs_main))
    print(f"[stage 2c] main+meanab FV mean intervention acc = {main_meanab_acc:.4f}")

    # ---- Dose-response plot (one figure per run) ----
    if recovery_curves:
        try:
            import matplotlib.pyplot as plt

            plots_dir = run_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            for (l, h), curve in recovery_curves.items():
                cs = sorted(curve.keys())
                accs = [curve[c] for c in cs]
                is_main = (int(l), int(h)) in main_set
                ax.plot(
                    cs,
                    accs,
                    color="C3" if is_main else "lightgray",
                    alpha=1.0 if is_main else 0.5,
                    linewidth=2.0 if is_main else 0.8,
                    label=f"({l},{h})" if is_main else None,
                )
            ax.axhline(
                mean_fv_acc,
                color="C0",
                linestyle="--",
                linewidth=1,
                label=f"mean-FV (all ablated)={mean_fv_acc:.3f}",
            )
            ax.axhline(
                clean_acc,
                color="C2",
                linestyle=":",
                linewidth=1,
                label=f"clean ceiling={clean_acc:.3f}",
            )
            ax.set_xlabel("Scaling coefficient c")
            ax.set_ylabel("Intervened accuracy")
            ax.set_title(
                f"Recovery scan: per-head FV(c) = c·z[l,h]_task + Σ_{{h'≠h}} mean_z\n"
                f"main (weak) = c*>0 ∧ steady rise (dip≤{args.weak_eps}) ∧ "
                f"gain ≥ max({args.weak_abs_floor}, {args.weak_rel_floor}·(A_sig−acc0))"
            )
            ax.set_ylim(0, max(1.0, clean_acc + 0.05))
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(str(plots_dir / "stage2_recovery_dose_response.png"), dpi=120)
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            print(f"[stage 2b] plot skipped: {exc}")

    # Serialize per-head dose-response and decisions as plain dicts.
    def _str_pos(d: Dict) -> Dict:
        return {f"({k[0]},{k[1]})": v for k, v in d.items()}

    summary = {
        "threshold": threshold,
        "threshold_method": args.auto_threshold,
        "significant_heads": [list(h) for h in significant],
        "main_heads": [list(h) for h in main_heads],
        "main_rank_by": args.main_rank_by if not fallback_heads else "fallback",
        "weak_params": (
            {
                "rel_floor": args.weak_rel_floor,
                "abs_floor": args.weak_abs_floor,
                "eps": args.weak_eps,
                "A_sig_ref": full_acc,
            }
            if (args.main_rank_by == "recovery" and not fallback_heads)
            else None
        ),
        "full_significant_fv_acc": full_acc,
        "raw_coef_fv_acc": raw_coef_acc,
        "main_plus_meanab_fv_acc": main_meanab_acc,
        "clean_acc": clean_acc,
        "mean_fv_acc": mean_fv_acc,
        "recovery_curves": _str_pos(recovery_curves),
        "recovery_decisions": _str_pos(recovery_decisions),
        "recovery_meta": recovery_meta,
        "checkpoint": ckpt,
    }
    (run_dir / "stage2_head_eval.json").write_text(json.dumps(summary, indent=2))

    # Per-example correctness in a separate file (avoids bloating the main JSON
    # since the analysis-ready stats are already in `recovery_decisions`).
    if recovery_per_example:
        per_example_out = {
            f"({l},{h})": {str(c): list(map(int, v)) for c, v in cdict.items()}
            for (l, h), cdict in recovery_per_example.items()
        }
        (run_dir / "stage2_recovery_per_example.json").write_text(
            json.dumps(per_example_out, separators=(",", ":"))
        )
    return summary


def _resolve_savevar_dir(matrix_dir: Path, task_dir: str, n_shot: int) -> Path:
    """Prefer the per-matrix ``savevars/``; fall back to the task-default location
    used by the paper baseline (``log/<task_dir>/savevars/``).

    Format-aware: the per-format z-cache filename carries ``format_tag()`` so the
    presence check (used by the CPU-only Stage 3, which cannot recompute) matches
    the file the active ``$FV_PROMPT_FORMAT`` actually wrote.
    """
    from subspaces.utils.prompt_formats import format_tag

    candidate = (
        matrix_dir
        / "savevars"
        / f"z_results_dict_{task_dir}_shot{n_shot}{format_tag()}.pth"
    )
    if candidate.exists():
        return candidate.parent
    fallback = PROJECT_ROOT / "log" / task_dir / "savevars"
    return fallback


def stage3_pca(args, matrix_dir: Path, run_dir: Path, summary2: Dict) -> Dict:
    """PCA per significant head + period/magnitude axis fit on main heads."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA

    from subspaces.utils.pca import fit_mod_vectors

    savevar_dir = _resolve_savevar_dir(matrix_dir, args.task_dir, args.n_shot)
    z = load_z_results_dict(
        None,
        None,
        args.n_shot,
        args.n_example,
        str(savevar_dir),
        args.task_dir,
        device="cpu",
    )
    task_names = sorted(
        z.keys(),
        key=lambda n: (
            int(n.replace("number-add", "")) if n.startswith("number-add") else 0
        ),
    )
    pca_dict = {}
    pcs_at_95 = {}
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for h in summary2["significant_heads"]:
        l, head = int(h[0]), int(h[1])
        X = (
            torch.stack([z[t][l][head] for t in task_names], dim=0)
            .to(torch.float32)
            .numpy()
        )
        pca = PCA().fit(X)
        cumv = np.cumsum(pca.explained_variance_ratio_)
        pca_dict[f"({l},{head})"] = cumv.tolist()
        pcs_at_95[f"({l},{head})"] = int(np.searchsorted(cumv, 0.95) + 1)
        plt.figure()
        plt.plot(np.arange(1, len(cumv) + 1), cumv, marker="o")
        plt.title(f"PCA cumvar L{l}H{head}")
        plt.xlabel("# PCs")
        plt.ylabel("Cumulative explained variance")
        plt.ylim(0, 1.01)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(plots_dir / f"pca_cumvar_L{l}H{head}.png"), dpi=120)
        plt.close()

    # Paper §4: fit the six named feature directions (mod2/mod5/mod10c/mod10s
    # = 4D unit subspace; mod25/mod50 = 2D magnitude subspace) on the main heads.
    mod_vector_fits = {}
    if args.task_dir.startswith("number_add"):
        ks = [
            int(n.replace("number-add", ""))
            for n in task_names
            if n.startswith("number-add")
        ]
        if len(ks) == len(task_names):
            for h in summary2["main_heads"]:
                l, head = int(h[0]), int(h[1])
                X = torch.stack([z[t][l][head] for t in task_names], dim=0).to(
                    torch.float32
                )
                fit = fit_mod_vectors(X, ks=ks)
                mod_vector_fits[f"({l},{head})"] = {
                    "target_r2": fit["target_r2"],
                    "pca_cumvar_at_6": fit["pca_cumvar_at_npcs"],
                    "col_names": fit["col_names"],
                    "unit_cols": fit["unit_cols"],
                    "magnitude_cols": fit["magnitude_cols"],
                }

    out = {
        "pca_cumvar": pca_dict,
        "pcs_at_95": pcs_at_95,
        "mod_vector_fits": mod_vector_fits,
    }
    (run_dir / "stage3_pca.json").write_text(json.dumps(out, indent=2))
    print(f"[stage 3] PCA done; #PCs@95% per head: {pcs_at_95}")
    return out


def stage4_extracted_signals(
    args, matrix_dir: Path, run_dir: Path, summary2: Dict, device: torch.device
) -> Dict:
    """Paper §5 — per-demo extracted-signal analysis on the main heads.

    §5.1/5.2 label-token peaking + signal alignment and the label-token
    aggregation analysis. add-k family only — the prompt construction and
    ``h_k`` directions are defined for ``number-add{k}`` tasks.
    """
    from subspaces.utils.signals import collect_signals, summarize_signals

    main_heads = [tuple(map(int, h)) for h in summary2.get("main_heads", [])]
    if not main_heads:
        print("[stage 4] SKIPPED: no main heads selected.")
        return {"status": "skipped", "reason": "no main heads"}
    if not args.task_dir.startswith("number_add"):
        print(
            f"[stage 4] SKIPPED: §5 signal extraction is add-k only (task={args.task_dir})."
        )
        return {"status": "skipped", "reason": f"non-addk task {args.task_dir}"}

    savevar_dir = _resolve_savevar_dir(matrix_dir, args.task_dir, args.n_shot)
    z_path = savevar_dir / f"z_results_dict_{args.task_dir}_shot{args.n_shot}.pth"
    if not z_path.exists():
        print(f"[stage 4] SKIPPED: z_results cache not found at {z_path}.")
        return {"status": "skipped", "reason": f"missing z_results {z_path}"}
    z_results_dict = torch.load(str(z_path), map_location="cpu", weights_only=False)

    model, _, _ = load_model_no_grad(args.model_name, device)
    print(
        f"[stage 4] §5 signals: heads={main_heads} | "
        f"{args.signal_n_prompts_per_k} prompts/k × k=1..{args.signal_k_max}"
    )
    acc = collect_signals(
        model,
        z_results_dict,
        main_heads,
        device,
        n_prompts_per_k=args.signal_n_prompts_per_k,
        n_shot=args.n_shot,
        k_range=range(1, args.signal_k_max + 1),
        n_comp=args.signal_n_comp,
        seed=0,
    )
    summary = summarize_signals(acc, n_boot=args.recovery_n_boot, seed=0)

    out = {
        "status": "ok",
        "main_heads": [list(h) for h in main_heads],
        "per_head": summary,
    }
    (run_dir / "stage4_signals.json").write_text(json.dumps(out, indent=2))
    try:
        from subspaces.runners.run_signal_extraction import _plots

        _plots(summary, run_dir)
    except Exception as e:  # plotting must never block the numeric artifact
        print(f"[stage 4] plot step skipped ({e}).")
    print(f"[stage 4] §5 done; wrote {run_dir / 'stage4_signals.json'}")
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stages = {int(s.strip()) for s in args.stages.split(",")}
    slug = slugify(args.model_name, args.task_dir, args.run_tag)
    run_dir = PROJECT_ROOT / "subspaces" / "runs" / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    invocation_cmd = "python -m subspaces.runners.run_pipeline " + " ".join(
        sys.argv[1:]
    )

    manifest = run_dir / "manifest.json"
    if manifest.exists() and not args.rerun:
        print(f"[skip] manifest already at {manifest}; pass --rerun to force.")
        return

    matrix_dir, reused = (
        stage1_train_or_reuse(args, run_dir)
        if 1 in stages
        else (PROJECT_ROOT / args.reuse_matrix, True)
    )
    summary2 = (
        stage2_head_selection_and_eval(args, matrix_dir, run_dir, device)
        if 2 in stages
        else {}
    )
    summary3 = (
        stage3_pca(args, matrix_dir, run_dir, summary2)
        if (3 in stages and summary2)
        else {}
    )
    summary4 = (
        stage4_extracted_signals(args, matrix_dir, run_dir, summary2, device)
        if (4 in stages and summary2)
        else {}
    )

    manifest_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_name": args.model_name,
        "task_dir": args.task_dir,
        "layer_name": args.layer_name,
        "matrix_dir": str(matrix_dir),
        "reused_matrix": reused,
        "stages_run": sorted(stages),
        "invocation": invocation_cmd,
        "stage2": summary2,
        "stage3_summary": (
            {"pcs_at_95": summary3.get("pcs_at_95", {})} if summary3 else {}
        ),
        "stage4_summary": summary4,
    }
    manifest.write_text(json.dumps(manifest_data, indent=2))

    if 5 in stages:
        experiments_md = PROJECT_ROOT / "subspaces" / "experiments.md"
        section = _render_experiments_md_section(
            slug=slug,
            args=args,
            matrix_dir=matrix_dir,
            reused=reused,
            summary2=summary2,
            summary3=summary3,
            summary4=summary4,
            invocation_cmd=invocation_cmd,
        )
        append_experiments_md(experiments_md, section)
        print(f"[stage 5] appended to {experiments_md}")

    print(f"[done] manifest: {manifest}")


def _render_experiments_md_section(
    *,
    slug: str,
    args,
    matrix_dir: Path,
    reused: bool,
    summary2: Dict,
    summary3: Dict,
    invocation_cmd: str,
    summary4: Dict = None,
) -> str:
    """Per-run section for subspaces/experiments.md. Includes:
    threshold + main-head selection rule + recovery decisions table +
    headline accuracy table + PCA #PCs@95% + §5 signal extraction +
    the *exact command* used.
    """
    sig = summary2.get("significant_heads", [])
    main = summary2.get("main_heads", [])
    threshold = summary2.get("threshold", float("nan"))
    rank_by = summary2.get("main_rank_by", "?")
    weak_params = summary2.get("weak_params") or {}
    mean_fv = summary2.get("mean_fv_acc", float("nan"))
    clean = summary2.get("clean_acc", float("nan"))
    full_acc = summary2.get("full_significant_fv_acc", float("nan"))
    raw_coef_acc = summary2.get("raw_coef_fv_acc", float("nan"))
    main_acc = summary2.get("main_plus_meanab_fv_acc", float("nan"))

    decisions = summary2.get("recovery_decisions", {}) or {}
    decisions_md = ""
    if decisions:
        rows = [
            "| Head | c* | acc(c0) | acc(c*) | gain | rel | bar | steady | main | flags |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---|",
        ]
        # Sort heads by gain (acc(c*) − acc(c0)) desc.
        sorted_heads = sorted(decisions.items(), key=lambda kv: -kv[1].get("gain", 0.0))
        for head_str, d in sorted_heads:
            rel_v = d.get("rel_recovery")
            rel_str = f"{rel_v:.3f}" if isinstance(rel_v, (int, float)) else "—"
            flags = ",".join(d.get("flags", [])) or "—"
            rows.append(
                f"| {head_str} | {d.get('c_star','?')} | {d.get('acc_at_c0',0):.3f} "
                f"| {d.get('acc_at_c_star',0):.3f} "
                f"| {d.get('gain',0):+.3f} | {rel_str} | {d.get('bar',0):.3f} "
                f"| {'✅' if d.get('steady') else '·'} "
                f"| {'⭐' if d.get('is_main') else '·'} | {flags} |"
            )
        rf, af, ep = (
            weak_params.get("rel_floor"),
            weak_params.get("abs_floor"),
            weak_params.get("eps"),
        )
        a_sig = weak_params.get("A_sig_ref")
        a_sig_str = f"{a_sig:.4f}" if isinstance(a_sig, (int, float)) else "—"
        decisions_md = (
            f"\n**Stage 2b — Recovery (weak) decisions** "
            f"(global mean-FV acc [all sig ablated] = {mean_fv:.4f}; clean ceiling = {clean:.4f}; "
            f"acc(c0) = leave-one-out; A_sig (full-sig FV) = {a_sig_str}, headroom = A_sig−acc(c0); "
            f"gain = acc(c*)−acc(c0); rel = gain/headroom; "
            f"scaled head is task-conditioned; "
            f"main = c*>0 ∧ steady (max dip≤{ep}) ∧ gain ≥ bar = max({af}, {rf}·headroom))\n\n"
            + "\n".join(rows)
            + "\n"
        )

    pcs_at_95 = summary3.get("pcs_at_95", {}) if summary3 else {}

    # Stage 4 — §5 signal extraction (label peaking + alignment + aggregation).
    stage4_md = ""
    s4 = summary4 or {}
    if s4.get("status") == "ok" and s4.get("per_head"):
        ph = s4["per_head"]
        rows1 = [
            "| head | α/label | α/off-label | peaking | signal cos [95% CI] |",
            "|---|---:|---:|---:|---|",
        ]
        rows3 = [
            "| head | ondemo cos [95% CI] | routing cos [95% CI] | routing null [95% CI] | cos_w | cos_u | wgain (null) |",
            "|---|---|---|---|---:|---:|---|",
        ]
        for hk, r in ph.items():
            ci = r["mean_signal_cosine_ci"]
            rows1.append(
                f"| {hk} | {r['mean_attn_per_label']:.3f} | {r['mean_attn_per_offlabel']:.4f} "
                f"| {r['label_peaking_ratio']:.1f}× | {r['mean_signal_cosine']:.3f} "
                f"[{ci[0]:.3f}, {ci[1]:.3f}] |"
            )
            oc, rc, rn = (
                r["ondemo_cos_full"],
                r["routing_corr_cos"],
                r["routing_corr_cos_null"],
            )
            wg, wn = r["weighting_gain"], r["weighting_gain_shuffled_null"]
            rows3.append(
                f"| {hk} | {oc['mean']:.3f} [{oc['ci'][0]:.3f}, {oc['ci'][1]:.3f}] "
                f"| {rc['mean']:.3f} [{rc['ci'][0]:.3f}, {rc['ci'][1]:.3f}] "
                f"| {rn['mean']:.3f} [{rn['ci'][0]:.3f}, {rn['ci'][1]:.3f}] "
                f"| {r['cos_weighted_label']['mean']:.3f} | {r['cos_uniform_label']['mean']:.3f} "
                f"| {wg['mean']:+.3f} ({wn['mean']:+.3f}) |"
            )
        stage4_md = (
            "\n**Stage 4 — §5 signal extraction** (§5.1/5.2 label-token peaking + alignment)\n\n"
            + "\n".join(rows1)
            + "\n\n**Attention × activation relationship (label tokens only)** — `ondemo cos` = "
            "full-space cos(`s_i`, `h_k`) over labels; `routing cos` = within-prompt Pearson(α_i, "
            "alignment_i); `cos_w`/`cos_u` = α-weighted vs uniform aggregate alignment; `wgain` = "
            "their difference (shuffled-α null in parens).\n\n"
            + "\n".join(rows3)
            + "\n"
        )

    main_str = main
    return (
        f"## Run: {slug} ({datetime.date.today().isoformat()})\n\n"
        f"- Model: `{args.model_name}`\n"
        f"- Task: `{args.task_dir}`  ·  Layer: `{args.layer_name}`\n"
        f"- Matrix: `{matrix_dir}` (reused={reused})\n"
        f"- Threshold: `{args.auto_threshold}` → "
        f"{threshold:.4f} ({len(sig)} significant heads)\n"
        f"- Main-head rule: `{rank_by}`"
        + (
            f" (weak: eps={weak_params.get('eps')}, abs_floor={weak_params.get('abs_floor')}, "
            f"rel_floor={weak_params.get('rel_floor')})"
            if rank_by == "recovery"
            else ""
        )
        + f" → {len(main)} heads: {main_str}\n\n"
        f"**Command**\n\n```bash\n{invocation_cmd}\n```\n\n"
        f"**Accuracies**\n\n"
        f"| Variant | Mean intervention acc |\n|---|---:|\n"
        f"| Clean baseline (ceiling) | {clean:.4f} |\n"
        f"| Global mean-FV (all {len(sig)} sig at cross-task mean) | {mean_fv:.4f} |\n"
        f"| Full significant FV (sum_{len(sig)}, unit coefs) | {full_acc:.4f} |\n"
        f"| Raw trained-coef FV (Σ M[l,h]·z, {len(sig)} sig) | {raw_coef_acc:.4f} |\n"
        f"| Main + mean-ablation FV ({len(main)} main) | {main_acc:.4f} |\n"
        f"{decisions_md}"
        f"\n**Stage 3 — PCA #PCs@95% per head**\n\n```json\n"
        f"{json.dumps(pcs_at_95, indent=2)}\n```\n"
        f"{stage4_md}"
        f"\nArtifacts: `subspaces/runs/{slug}/stage2_head_eval.json`, `stage3_pca.json`, "
        + ("`stage4_signals.json`, " if stage4_md else "")
        + "`plots/stage2_recovery_dose_response.png`, `plots/pca_cumvar_*.png`"
        + (", `plots/attn_profile_*.png`" if stage4_md else "")
        + "\n\n---\n\n"
    )


if __name__ == "__main__":
    main()
