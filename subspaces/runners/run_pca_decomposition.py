"""Paper §4 — subspace decomposition into the six periodic feature directions (CPU).

For each requested head: PCA → 6-D subspace, then fit the six named feature
directions via :func:`subspaces.utils.pca.fit_mod_vectors` (a faithful port of the
original ``coords_heads.ipynb::fit_head_period``):

  * **4-D unit subspace**      — mod2, mod5, mod10cos, mod10sin (periods 2/5/10)
  * **2-D magnitude subspace** — mod25, mod50 (periods 25/50)

This runs purely on the cached per-task mean head activations (``z_results``),
so it needs no GPU. Outputs land in ``subspaces/runs/<out-slug>/``:

  * ``section4_mod_vectors.pth``  — ``{head: (d_model, 6) tensor}``
  * ``section4_metrics.json``     — per-head target R² + cumvar (+ GT comparison)

and, unless ``--no-log`` is given, a ``## Section 4 decomposition`` entry is
appended to ``subspaces/experiments.md``.

Examples
--------
Reproduce Llama and validate against the saved ground-truth artifact::

    python -m subspaces.runners.run_pca_decomposition \
        --model-name meta-llama/Meta-Llama-3-8B-Instruct \
        --z-results 'log/number_add/savevars/z_results_dict_number_add_shot5.pth' \
        --heads 15:2,15:1,13:6 \
        --out-run meta-llama_meta-llama-3-8b-instruct__number_add \
        --compare-mod-vectors 'artifacts/matrix_add_0204_clip_lambda0.05/mod_vectors_dict.pth'

Generalize to another model, reading its selected heads from its pipeline run::

    python -m subspaces.runners.run_pca_decomposition \
        --model-name Qwen/Qwen3-8B \
        --z-results 'log/number_add/savevars/z_results_dict_number_add_shot5.pth' \
        --heads-from-run qwen_qwen3-8b__number_add
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from subspaces.utils.pca import (  # noqa: E402
    MAGNITUDE_COLS,
    MOD_VECTOR_COLS,
    UNIT_COLS,
    fit_mod_vectors,
    subspace_principal_angles,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper §4 mod-vector decomposition for selected heads (CPU)."
    )
    p.add_argument(
        "--model-name",
        default=None,
        help="For logging; falls back to the run manifest's model_name.",
    )
    p.add_argument(
        "--z-results",
        required=True,
        help="Path to z_results_dict_*.pth (per-task mean head activations).",
    )
    p.add_argument(
        "--heads",
        default=None,
        help="'l:h,l:h,...' heads to decompose. Overrides --heads-from-run.",
    )
    p.add_argument(
        "--heads-from-run",
        default=None,
        help="Run slug under subspaces/runs/: read main_heads (fallback: significant heads from stage3_pca.json).",
    )
    p.add_argument(
        "--n-pcs", type=int, default=6, help="PCA subspace dimension (paper §4: 6)."
    )
    p.add_argument(
        "--compare-mod-vectors",
        default=None,
        help="Ground-truth mod_vectors_dict.pth to validate against.",
    )
    p.add_argument(
        "--out-run",
        default=None,
        help="Output run slug under subspaces/runs/. Defaults to --heads-from-run.",
    )
    p.add_argument(
        "--top-k-report",
        type=int,
        default=8,
        help="Max heads shown in the experiments.md table.",
    )
    p.add_argument(
        "--no-log",
        action="store_true",
        help="Do not append to subspaces/experiments.md.",
    )
    return p.parse_args()


def _parse_heads(spec: str):
    heads = []
    for tok in spec.split(","):
        tok = tok.strip().strip("()")
        if not tok:
            continue
        a, b = tok.split(":") if ":" in tok else tok.split(",")
        heads.append((int(a), int(b)))
    return heads


def _heads_from_run(run_dir: Path):
    """Return (heads, source). main_heads if present, else significant heads."""
    s2 = run_dir / "stage2_head_eval.json"
    if s2.exists():
        d = json.loads(s2.read_text())
        mh = d.get("main_heads") or []
        if mh:
            return [(int(a), int(b)) for a, b in mh], "main_heads"
    s3 = run_dir / "stage3_pca.json"
    if s3.exists():
        d = json.loads(s3.read_text())
        keys = list((d.get("pcs_at_95") or d.get("pca_cumvar") or {}).keys())
        heads = []
        for k in keys:
            a, b = k.strip("()").split(",")
            heads.append((int(a), int(b)))
        if heads:
            return heads, "significant_heads(fallback)"
    return [], "none"


def _to_f32(v):
    return (
        v.to(torch.float32)
        if hasattr(v, "to")
        else torch.tensor(np.asarray(v), dtype=torch.float32)
    )


def _head_matrix(z, task_names, l, h):
    return torch.stack(
        [_to_f32(z[t][l][h]) for t in task_names], dim=0
    )  # (n_tasks, d_model)


def main() -> None:
    args = parse_args()

    # Resolve output run dir + heads.
    out_slug = args.out_run or args.heads_from_run
    if out_slug is None:
        raise SystemExit(
            "Provide --out-run (or --heads-from-run) so outputs have a home under subspaces/runs/."
        )
    run_dir = PROJECT_ROOT / "subspaces" / "runs" / out_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name
    if args.heads:
        heads, head_source = _parse_heads(args.heads), "--heads"
    elif args.heads_from_run:
        src_dir = PROJECT_ROOT / "subspaces" / "runs" / args.heads_from_run
        heads, head_source = _heads_from_run(src_dir)
        if model_name is None:
            man = src_dir / "manifest.json"
            if man.exists():
                model_name = json.loads(man.read_text()).get("model_name")
    else:
        raise SystemExit("Provide --heads or --heads-from-run.")
    if not heads:
        raise SystemExit(
            f"No heads to decompose (source resolved empty: {head_source})."
        )
    model_name = model_name or out_slug

    # Load z_results (per-task mean head activations) and the add-k task order.
    z = torch.load(
        (
            str(PROJECT_ROOT / args.z_results)
            if not Path(args.z_results).is_absolute()
            else args.z_results
        ),
        map_location="cpu",
        weights_only=False,
    )
    task_names = sorted(
        z.keys(),
        key=lambda n: (
            int(n.replace("number-add", "")) if n.startswith("number-add") else 0
        ),
    )
    ks = [
        int(n.replace("number-add", ""))
        for n in task_names
        if n.startswith("number-add")
    ]
    if len(ks) != len(task_names):
        raise SystemExit(
            "z_results task names are not all 'number-add<k>'; §4 fit needs the add-k family."
        )

    gt = None
    if args.compare_mod_vectors:
        gt = torch.load(
            args.compare_mod_vectors, map_location="cpu", weights_only=False
        )

    metrics = {}
    mod_vectors_out = {}
    for l, h in heads:
        X = _head_matrix(z, task_names, l, h)
        fit = fit_mod_vectors(X, ks=ks, n_pcs=args.n_pcs)
        mv = fit["mod_vectors"]  # (d_model, 6)
        mod_vectors_out[f"({l},{h})"] = torch.tensor(mv, dtype=torch.float32)
        r2 = fit["target_r2"]
        rec = {
            "pca_cumvar_at_6": fit["pca_cumvar_at_npcs"],
            "target_r2": r2,
            "unit_r2_mean": float(np.mean([r2[MOD_VECTOR_COLS[i]] for i in UNIT_COLS])),
            "magnitude_r2_mean": float(
                np.mean([r2[MOD_VECTOR_COLS[i]] for i in MAGNITUDE_COLS])
            ),
            "mean_r2": float(np.mean(list(r2.values()))),
        }
        if gt is not None and (l, h) in gt:
            g = gt[(l, h)]
            g = (
                g.detach().cpu().to(torch.float32).numpy()
                if hasattr(g, "detach")
                else np.asarray(g, dtype="float64")
            )
            cos = {
                MOD_VECTOR_COLS[j]: float(
                    abs(
                        np.dot(mv[:, j], g[:, j])
                        / (np.linalg.norm(mv[:, j]) * np.linalg.norm(g[:, j]))
                    )
                )
                for j in range(6)
            }
            f6 = subspace_principal_angles(mv, g)
            fu = subspace_principal_angles(
                mv[:, list(UNIT_COLS)], g[:, list(UNIT_COLS)]
            )
            fm = subspace_principal_angles(
                mv[:, list(MAGNITUDE_COLS)], g[:, list(MAGNITUDE_COLS)]
            )
            rec["gt_cos"] = cos
            rec["gt_angle_6d_max_mean"] = list(f6)
            rec["gt_angle_unit_max_mean"] = list(fu)
            rec["gt_angle_magnitude_max_mean"] = list(fm)
        metrics[f"({l},{h})"] = rec

    # Save artifacts.
    torch.save(mod_vectors_out, str(run_dir / "section4_mod_vectors.pth"))
    meta = {
        "model_name": model_name,
        "z_results": args.z_results,
        "heads": [list(h) for h in heads],
        "head_source": head_source,
        "n_pcs": args.n_pcs,
        "compare_mod_vectors": args.compare_mod_vectors,
        "col_names": list(MOD_VECTOR_COLS),
        "unit_cols": list(UNIT_COLS),
        "magnitude_cols": list(MAGNITUDE_COLS),
        "per_head": metrics,
    }
    (run_dir / "section4_metrics.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[§4] wrote {run_dir/'section4_metrics.json'} and section4_mod_vectors.pth ({len(heads)} heads)."
    )

    if not args.no_log:
        _append_experiments_md(
            model_name, args, heads, head_source, metrics, gt is not None, out_slug
        )


def _append_experiments_md(
    model_name, args, heads, head_source, metrics, has_gt, out_slug
):
    date = datetime.date.today().isoformat()
    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["mean_r2"], reverse=True)
    shown = ranked[: args.top_k_report]
    lines = []
    lines.append(f"\n## Section 4 decomposition: {out_slug} ({date})\n")
    lines.append(f"- Model: `{model_name}`")
    lines.append(f"- z_results: `{args.z_results}`")
    lines.append(
        f"- Heads ({head_source}, {len(heads)} total): {', '.join(k for k,_ in metrics.items())}"
    )
    lines.append(
        "- Method: PCA 6-D subspace → `fit_mod_vectors` (4-D unit periods 2/5/10 + "
        "2-D magnitude periods 25/50), faithful port of `coords_heads.ipynb::fit_head_period`.\n"
    )
    lines.append(
        "**Per-head periodic fit** — `target R²` = how well each named direction is reconstructed "
        "from the head's 6 PCs; `cumvar@6` = variance captured by the 6-D PCA subspace"
        + (
            f". Showing top {len(shown)} of {len(metrics)} heads by mean R².\n"
            if len(metrics) > len(shown)
            else ".\n"
        )
    )
    lines.append(
        "| head | cumvar@6 | mod2 | mod5 | mod10c | mod10s | mod25 | mod50 | unit R̄² | mag R̄² |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, rec in shown:
        r2 = rec["target_r2"]
        lines.append(
            f"| {k} | {rec['pca_cumvar_at_6']:.3f} | {r2['mod2']:.3f} | {r2['mod5']:.3f} | "
            f"{r2['mod10c']:.3f} | {r2['mod10s']:.3f} | {r2['mod25']:.3f} | {r2['mod50']:.3f} | "
            f"{rec['unit_r2_mean']:.3f} | {rec['magnitude_r2_mean']:.3f} |"
        )
    if has_gt:
        lines.append(
            "\n**Reproduction vs ground-truth `mod_vectors_dict.pth`** — per-direction |cos| and "
            "principal angles (deg) between fitted and saved subspaces (0° = identical span).\n"
        )
        lines.append(
            "| head | \\|cos\\| mod2/5/10c/10s/25/50 | 6-D angle max/mean | unit angle max/mean | mag angle max/mean |"
        )
        lines.append("|---|---|---:|---:|---:|")
        for k, rec in shown:
            if "gt_cos" not in rec:
                continue
            c = rec["gt_cos"]
            cs = "/".join(
                f"{c[n]:.3f}"
                for n in ("mod2", "mod5", "mod10c", "mod10s", "mod25", "mod50")
            )
            a6, au, am = (
                rec["gt_angle_6d_max_mean"],
                rec["gt_angle_unit_max_mean"],
                rec["gt_angle_magnitude_max_mean"],
            )
            lines.append(
                f"| {k} | {cs} | {a6[0]:.1f}/{a6[1]:.1f} | {au[0]:.1f}/{au[1]:.1f} | {am[0]:.1f}/{am[1]:.1f} |"
            )
    lines.append(
        f"\nArtifacts: `subspaces/runs/{out_slug}/section4_mod_vectors.pth`, `subspaces/runs/{out_slug}/section4_metrics.json`"
    )
    lines.append("\n---")
    md_path = PROJECT_ROOT / "subspaces" / "experiments.md"
    with open(md_path, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[§4] appended Section 4 decomposition entry to {md_path}")


if __name__ == "__main__":
    main()
