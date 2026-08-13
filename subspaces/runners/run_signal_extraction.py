"""Paper §5 — "Signal Extractors of ICL Demonstrations" (GPU).

For each requested head, run forward passes over a seed-pinned add-k prompt
sample and measure:

  * **§5.1/5.2 — label-token peaking** — final-token attention concentrates on
    the demo label tokens (``label_peaking_ratio``), and the per-demo extracted
    signal ``s_i = O_h V_h z_{y_i}`` aligns (in the head's 6-D PCA subspace) with
    the add-k task direction ``h_k`` (``mean_signal_cosine``).
  * **label-token aggregation** — within the labels, does attention prefer the
    better-aligned labels, and does the attention-weighted average over labels
    align with ``h_k`` better than a uniform average? See
    :func:`subspaces.utils.signals.label_relationship`.

Unlike §4 (cache-only, CPU), this needs model forward passes → run on GPU
(e.g. via a Slurm launcher; see ``examples/slurm/``). Outputs land in
``subspaces/runs/<out-slug>/``:

  * ``section5_signals.json`` — per-head §5.1/5.2 + aggregation statistics (with CIs)
  * ``plots/attn_profile_L{l}H{h}.png`` — mean final-token attention by token role

and, unless ``--no-log`` is given, a ``## Section 5`` entry is appended to
``subspaces/experiments.md``.

Examples
--------
Reproduce Llama add-k §5 on the 3 main heads, into the existing run dir::

    python -m subspaces.runners.run_signal_extraction \
        --model-name meta-llama/Meta-Llama-3-8B-Instruct \
        --z-results 'log/number_add/savevars/z_results_dict_number_add_shot5.pth' \
        --heads 15:2,15:1,13:6 \
        --n-prompts-per-k 40 --k-max 30 \
        --out-run meta-llama_meta-llama-3-8b-instruct__number_add
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from subspaces.utils.model import load_model_no_grad  # noqa: E402
from subspaces.utils.signals import (  # noqa: E402
    MAIN_HEADS,
    collect_signals,
    summarize_signals,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper §5 extracted-signal analysis for selected heads (GPU)."
    )
    p.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument(
        "--z-results",
        required=True,
        help="z_results_dict_*.pth (per-task mean head activations / h_k).",
    )
    p.add_argument(
        "--heads",
        default=None,
        help="'l:h,l:h,...'. Default: the 3 paper main heads 15:2,15:1,13:6.",
    )
    p.add_argument(
        "--heads-from-run",
        default=None,
        help="Run slug under subspaces/runs/: read main_heads.",
    )
    p.add_argument(
        "--out-run",
        default=None,
        help="Output run slug under subspaces/runs/. Defaults to --heads-from-run.",
    )
    p.add_argument(
        "--n-prompts-per-k", type=int, default=40, help="Prompts sampled per task-k."
    )
    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument(
        "--task-prefix",
        default="number-add",
        help="z_results key prefix for the parametric family: 'number-add' (add/sub) "
        "or 'number-mul' (mul).",
    )
    p.add_argument(
        "--task-op",
        default="add",
        choices=("add", "mul"),
        help="Label function y=f(x,k): add (x+k; k<0 = subtraction) or mul (x*k).",
    )
    p.add_argument(
        "--k-min",
        type=int,
        default=1,
        help="Sweep k=k-min..k-max (subtraction: negative).",
    )
    p.add_argument(
        "--k-max",
        type=int,
        default=30,
        help="Sweep k=k-min..k-max (paper add-k: 1..30).",
    )
    p.add_argument(
        "--discrete",
        action="store_true",
        help="Reduced §5 for non-parametric (semantic) tasks: per-task FV as the task "
        "direction, full-space (no periodic subspace / no k-sweep). Needs --task-dir-name.",
    )
    p.add_argument(
        "--task-dir-name",
        default=None,
        help="dataset_files/<name> for --discrete (e.g. abstractive, extractive).",
    )
    p.add_argument(
        "--n-prompts-per-task",
        type=int,
        default=40,
        help="Prompts/task in --discrete mode.",
    )
    p.add_argument(
        "--n-comp", type=int, default=6, help="PCA subspace dim (paper §4: 6)."
    )
    p.add_argument(
        "--n-boot", type=int, default=2000, help="Bootstrap resamples for CIs."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-log",
        action="store_true",
        help="Do not append to subspaces/experiments.md.",
    )
    return p.parse_args()


def _parse_heads(spec: str):
    out = []
    for tok in spec.split(","):
        tok = tok.strip().strip("()")
        if not tok:
            continue
        a, b = tok.split(":") if ":" in tok else tok.split(",")
        out.append((int(a), int(b)))
    return out


def _heads_from_run(run_dir: Path):
    s2 = run_dir / "stage2_head_eval.json"
    if s2.exists():
        mh = json.loads(s2.read_text()).get("main_heads") or []
        if mh:
            return [(int(a), int(b)) for a, b in mh]
    return []


def _plots(summary, run_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    order = ["bos", "input", "arrow", "label", "hash", "final", "other"]
    for key, rec in summary.items():
        l, h = key.strip("()").split(",")
        # §5.2 attention-by-token-role bar (label peaking)
        pt = rec["attn_by_postype"]
        cats = [c for c in order if c in pt]
        plt.figure(figsize=(5, 3))
        plt.bar(
            cats,
            [pt[c] for c in cats],
            color=["#bbb" if c != "label" else "#d62728" for c in cats],
        )
        plt.ylabel("mean final-token attention")
        plt.title(f"L{l}H{h} label peaking (ratio {rec['label_peaking_ratio']:.1f}×)")
        plt.tight_layout()
        plt.savefig(str(plots_dir / f"attn_profile_L{l}H{h}.png"), dpi=120)
        plt.close()
        # relationship: aggregate alignment under attention-weighted vs uniform avg (label tokens)
        cw, cu = rec["cos_weighted_label"]["mean"], rec["cos_uniform_label"]["mean"]
        oc = rec["ondemo_cos_full"]["mean"]
        plt.figure(figsize=(4, 3))
        plt.bar(
            ["per-label\nmean", "uniform\navg", "α-weighted\navg"],
            [oc, cu, cw],
            color=["#7f7f7f", "#1f77b4", "#d62728"],
        )
        plt.ylabel("cos with h_k")
        plt.ylim(0, 1.01)
        plt.title(
            f"L{l}H{h} label-token alignment\nwgain={rec['weighting_gain']['mean']:+.3f}"
        )
        plt.tight_layout()
        plt.savefig(str(plots_dir / f"weighting_gain_L{l}H{h}.png"), dpi=120)
        plt.close()


def _append_experiments_md(model_name, args, heads, summary, out_slug):
    date = datetime.date.today().isoformat()
    lines = [f"\n## Section 5 signal extraction: {out_slug} ({date})\n"]
    lines.append(f"- Model: `{model_name}`")
    lines.append(f"- z_results: `{args.z_results}`")
    lines.append(f"- Heads ({len(heads)}): {', '.join(f'({l},{h})' for l, h in heads)}")
    if args.discrete:
        lines.append(
            "- Mode: **DISCRETE / reduced §5** — non-parametric tasks; per-task FV `h_task` "
            "is the task direction; all readings are **full-space** (no periodic 6-D subspace, "
            "no k-sweep). `signal cos`/`ondemo cos` are full-space here."
        )
        lines.append(
            f"- Tasks: `dataset_files/{args.task_dir_name}` (discrete ICL tasks)"
        )
        lines.append(
            f"- Sample: {args.n_prompts_per_task} prompts/task, {args.n_shot}-shot; "
            f"bootstrap n={args.n_boot}, seed={args.seed}.\n"
        )
    else:
        lines.append(
            f"- Task family: `{args.task_prefix}` / op `{args.task_op}` (y=f(x,k))"
        )
        lines.append(
            f"- Sample: {args.n_prompts_per_k} prompts/k × k={args.k_min}..{args.k_max}, {args.n_shot}-shot; "
            f"PCA subspace dim {args.n_comp}; bootstrap n={args.n_boot}, seed={args.seed}.\n"
        )
    lines.append(
        "**Setup & metric definitions.** For each clean add-k n-shot prompt "
        "`x1->y1#…#xn->yn#xq->`, one forward pass captures the head's final-token attention "
        "pattern and value vectors. For head (l,h) and demo label token y_i: `α_i` = final-token "
        "attention to y_i (Σ over all key positions = 1); `s_i = O_h V_h z_{y_i}` = the per-label "
        "extracted signal; `h_k = z_results[number-add k][l,h]` = the add-k task vector; "
        "`subspace` = top-6 PCA axes of {h_k}_k. Metrics:\n"
        "- `signal cos` (subspace) / `ondemo cos` (full-space) = mean over labels & prompts of "
        "cos(`s_i`,`h_k`) in the 6-D subspace / full d_model.\n"
        "- `peaking ratio` = mean α on a label ÷ mean α on a non-label position.\n"
        "- `routing cos` = **mean over prompts of the within-prompt Pearson r between `α_i` and "
        "cos(`s_i`,`h_k`) across the n_shot labels**; its **null** is the same statistic after "
        "permuting `α_i` among the labels (averaged over 30 perms/prompt). A positive routing "
        "relationship is indicated when the routing-cos 95% CI lies **above the null CI** (and "
        "above 0).\n"
        "- `wgain` = cos(α-weighted mean of `s_i` over labels, `h_k`) − cos(uniform mean, `h_k`); "
        "its `null` permutes `α` over labels. All CIs are 95% bootstrap over prompts; n_shot Pearson "
        "(5 points/prompt) is noisy per prompt but its cross-prompt mean is well-estimated.\n"
    )
    lines.append(
        "**§5.1/5.2 — label-token peaking + signal alignment** — final-token attention per "
        "label token vs per off-label token (ratio ≫1 = peaking); `signal cos` = mean α-free "
        "subspace cosine of the extracted signal `s_i` with the task direction `h_k`.\n"
    )
    lines.append(
        "| head | α/label | α/off-label | peaking ratio | signal cos [95% CI] |"
    )
    lines.append("|---|---:|---:|---:|---|")
    for l, h in heads:
        r = summary[f"({l},{h})"]
        ci = r["mean_signal_cosine_ci"]
        lines.append(
            f"| ({l},{h}) | {r['mean_attn_per_label']:.3f} | {r['mean_attn_per_offlabel']:.4f} "
            f"| {r['label_peaking_ratio']:.1f}× | {r['mean_signal_cosine']:.3f} "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] |"
        )
    lines.append(
        "\n**Attention × activation relationship (LABEL tokens only)** — refactor of the "
        "follow-up aggregation study's mode-3 analysis, restricted to the demo labels. "
        "`ondemo cos` = mean full-space cos(`s_i`, `h_k`) over labels (the §5.1/5.2 "
        "alignment claim); `routing cos` = mean within-prompt Pearson(α_i, cos(s_i,h_k)) "
        "— does attention prefer better-aligned labels; `cos_w`/`cos_u` = aggregate "
        "alignment under attention-weighted vs uniform averaging over labels; `wgain` = "
        "`cos_w − cos_u` (shuffled-α null in brackets); `task/sink` = attention mass on "
        "labels vs BOS/hash/arrow/final.\n"
    )
    lines.append(
        "| head | ondemo cos [95% CI] | routing cos [95% CI] | routing-cos null [95% CI] | cos_w | cos_u | wgain (null) | task/sink |"
    )
    lines.append("|---|---|---|---|---:|---:|---|---:|")
    for l, h in heads:
        r = summary[f"({l},{h})"]
        oc, rc, rn = (
            r["ondemo_cos_full"],
            r["routing_corr_cos"],
            r["routing_corr_cos_null"],
        )
        wg, wn = r["weighting_gain"], r["weighting_gain_shuffled_null"]
        lines.append(
            f"| ({l},{h}) | {oc['mean']:.3f} [{oc['ci'][0]:.3f}, {oc['ci'][1]:.3f}] "
            f"| {rc['mean']:.3f} [{rc['ci'][0]:.3f}, {rc['ci'][1]:.3f}] "
            f"| {rn['mean']:.3f} [{rn['ci'][0]:.3f}, {rn['ci'][1]:.3f}] "
            f"| {r['cos_weighted_label']['mean']:.3f} | {r['cos_uniform_label']['mean']:.3f} "
            f"| {wg['mean']:+.3f} ({wn['mean']:+.3f}) "
            f"| {r['task_mass']:.2f}/{r['sink_mass']:.2f} |"
        )
    lines.append(
        f"\nArtifacts: `subspaces/runs/{out_slug}/section5_signals.json`, "
        f"`subspaces/runs/{out_slug}/plots/attn_profile_*.png`, `weighting_gain_*.png`"
    )
    lines.append("\n---")
    md_path = PROJECT_ROOT / "subspaces" / "experiments.md"
    with open(md_path, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[§5] appended Section 5 entry to {md_path}")


def main() -> None:
    args = parse_args()
    out_slug = args.out_run or args.heads_from_run
    if out_slug is None:
        raise SystemExit(
            "Provide --out-run (or --heads-from-run) so outputs have a home under subspaces/runs/."
        )
    run_dir = PROJECT_ROOT / "subspaces" / "runs" / out_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.heads:
        heads = _parse_heads(args.heads)
    elif args.heads_from_run:
        heads = _heads_from_run(
            PROJECT_ROOT / "subspaces" / "runs" / args.heads_from_run
        )
    else:
        heads = list(MAIN_HEADS)
    if not heads:
        raise SystemExit("No heads resolved; pass --heads or a run with main_heads.")

    z_path = (
        args.z_results
        if Path(args.z_results).is_absolute()
        else str(PROJECT_ROOT / args.z_results)
    )
    z_results_dict = torch.load(z_path, map_location="cpu", weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[§5][warn] no CUDA device — forward passes will be very slow on CPU.")
    model, _, _ = load_model_no_grad(args.model_name, device)

    if args.discrete:
        from subspaces.utils.io import load_task_data
        from subspaces.utils.signals import collect_signals_discrete

        if not args.task_dir_name:
            raise SystemExit(
                "--discrete requires --task-dir-name (dataset_files/<name>)."
            )
        tasks_data = load_task_data(
            str(PROJECT_ROOT / "dataset_files" / args.task_dir_name)
        )
        print(
            f"[§5] DISCRETE (reduced, full-space): heads={heads} | {len(tasks_data)} tasks | "
            f"{args.n_prompts_per_task} prompts/task"
        )
        acc = collect_signals_discrete(
            model,
            z_results_dict,
            tasks_data,
            heads,
            device,
            n_prompts_per_task=args.n_prompts_per_task,
            n_shot=args.n_shot,
            seed=args.seed,
        )
    else:
        from subspaces.utils.signals import LABEL_FNS

        label_fn = LABEL_FNS[args.task_op]
        k_range = range(args.k_min, args.k_max + 1)
        print(
            f"[§5] collecting signals: heads={heads} | task={args.task_prefix}/{args.task_op} | "
            f"{args.n_prompts_per_k} prompts/k × k={args.k_min}..{args.k_max}"
        )
        acc = collect_signals(
            model,
            z_results_dict,
            heads,
            device,
            n_prompts_per_k=args.n_prompts_per_k,
            n_shot=args.n_shot,
            k_range=k_range,
            n_comp=args.n_comp,
            seed=args.seed,
            task_prefix=args.task_prefix,
            label_fn=label_fn,
        )
    summary = summarize_signals(acc, n_boot=args.n_boot, seed=args.seed)

    out = {
        "model_name": args.model_name,
        "z_results": args.z_results,
        "mode": "discrete" if args.discrete else "parametric",
        "task_prefix": args.task_prefix,
        "task_op": args.task_op,
        "task_dir_name": args.task_dir_name,
        "heads": [list(h) for h in heads],
        "n_prompts_per_k": args.n_prompts_per_k,
        "n_prompts_per_task": args.n_prompts_per_task,
        "n_shot": args.n_shot,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "n_comp": args.n_comp,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "per_head": summary,
    }
    (run_dir / "section5_signals.json").write_text(json.dumps(out, indent=2))
    print(f"[§5] wrote {run_dir / 'section5_signals.json'}")
    _plots(summary, run_dir)
    print(f"[§5] wrote plots to {run_dir / 'plots'}")
    if not args.no_log:
        _append_experiments_md(args.model_name, args, heads, summary, out_slug)


if __name__ == "__main__":
    main()
