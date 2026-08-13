"""Five-shot necessity check for the main heads (paper §3.2, camera-ready).

For an existing ``headset_eval`` manifest, re-evaluate the CLEAN n-shot
prompts of its recorded final-eval sample manifest while MEAN-ABLATING
selected heads during the forward pass: each ablated head's
``attn.hook_result`` output at the final prompt token is replaced by the
cross-task mean ``h_bar`` from the manifest's TRAIN z cache
(``subspaces.step1.icl_ablation``). Conditions:

    clean         no hooks; anchor against the manifest's recorded clean metric
    null_control  use_attn_result=True hook path active, nothing ablated
    main_trio     the manifest's main heads mean-ablated
    random_##     N random size-K subsets of the OTHER significant heads

Corrected-protocol counterpart of the paper paragraph "Validating necessity
of main heads in the five-shot setting" (paper-era numbers: trio 0.43 vs
clean 0.87; 95% of 20 random 3-subsets >= 0.86). Protocol differences to
label when comparing: prompts here are the corrected cell's 5 held-out tasks
x 100 deterministic examples (recorded clean 0.870) and h_bar averages the
25 training tasks, whereas the paper-era run sampled prompts across all 30
tasks and averaged h_bar over all 30.

Standalone report generator: writes JSON under ``log/fiveshot_necessity/``
and creates no lineage-tree artifacts.

Usage (from the repo root, GPU node):
    .venv/bin/python scripts/fiveshot_necessity.py \
        log/runs/<node>__<id>/sig-<m>-v1/scan-<tag>/main-<sel>-<h8>/eval-<h10>.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

PAPER_REFERENCE = {
    "paragraph": "Validating necessity of main heads in the five-shot setting",
    "trio_meanab_acc": 0.43,
    "clean_acc": 0.87,
    "n_random_sets": 20,
    "set_size": 3,
    "random_sets_claim": "95% have accuracy at least 0.86",
    "protocol": "paper-era (30-task prompts, 30-task h_bar); this run uses the "
    "corrected held-out protocol — label the difference when comparing",
}


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(
            cmd, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD") or None,
        "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no")),
    }


def load_manifest_cell(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "headset_eval":
        raise SystemExit(f"{manifest_path}: expected a headset_eval manifest")
    scoring = manifest["config"]["scoring"]
    if (scoring["name"], scoring["version"]) != ("exact_token_match", 1):
        raise SystemExit(
            f"{manifest_path}: scorer {scoring} is not exact_token_match v1; "
            "this report reimplements that scorer only"
        )
    samples = json.loads(
        Path(manifest["inputs"]["final_eval_samples"]["path"]).read_text()
    )
    sig_payload = json.loads(
        Path(manifest["inputs"]["significant_heads"]["path"]).read_text()
    )
    task_cfg = samples["config"]["task"]
    return {
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "samples": samples,
        "sig_heads": [(int(li), int(hi)) for li, hi, *_ in sig_payload["heads"]],
        "main_heads": [(int(li), int(hi)) for li, hi in manifest["main_heads"]],
        "batch_size": int(manifest["config"]["batch_size"]),
        "z_train_fp": manifest["inputs"]["z_cache_train"]["content_fingerprint"],
        "recorded_clean": manifest["metrics"]["clean"],
        "task_cfg": task_cfg,
        "label": "{}__{}__n{}".format(
            task_cfg["dataset_dir"], task_cfg["prompt_format"], task_cfg["n_shot"]
        ),
    }


def collect_prompts(samples: dict, limit_per_task: int | None) -> dict:
    """Task -> (n-shot prompts, targets), truncated deterministically."""
    from subspaces.step1.eval_gpu import manifest_prompts

    per_task = {}
    for task in samples["task_order"]:
        prompts, targets = manifest_prompts(samples, task, zero_shot=False)
        if limit_per_task is not None:
            prompts, targets = prompts[:limit_per_task], targets[:limit_per_task]
        per_task[task] = (prompts, targets)
    return per_task


def bootstrap_ci(correct: list, seed: int, n_boot: int = 2000) -> list:
    import numpy as np

    rng = np.random.default_rng(seed)
    outcomes = np.asarray(correct, dtype=float)
    means = rng.choice(outcomes, size=(n_boot, len(outcomes)), replace=True).mean(
        axis=1
    )
    low, high = np.percentile(means, [2.5, 97.5])
    return [float(low), float(high)]


def eval_condition(model, per_task_prompts: dict, head_values: dict | None, bs: int):
    """head_values None -> plain clean forward; {} -> null control."""
    import numpy as np

    from subspaces.step1.eval_gpu import eval_prompts
    from subspaces.step1.icl_ablation import ablated_generation_correctness

    per_task_acc = {}
    pooled: list[int] = []
    for task, (prompts, targets) in per_task_prompts.items():
        if head_values is None:
            correct, _bs = eval_prompts(model, prompts, targets, batch_size=bs)
        else:
            correct = ablated_generation_correctness(
                model, prompts, targets, head_values, batch_size=bs
            )
        per_task_acc[task] = float(np.mean(correct))
        pooled.extend(correct)
    return {
        "acc": float(np.mean(pooled)),
        "per_task": per_task_acc,
        "correct": pooled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", help="headset_eval manifest path")
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--set-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=None,
        help="smoke mode: evaluate only the first N examples per task",
    )
    parser.add_argument("--out-dir", default="log/fiveshot_necessity")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="override the recorded batch size"
    )
    args = parser.parse_args()

    import torch

    from subspaces.step1.eval_gpu import mean_z_over_tasks
    from subspaces.step1.icl_ablation import HOOK_SITE, IMPL, sample_head_subsets
    from subspaces.step1.zcache import load_zcache_dir
    from subspaces.utils.model import load_model_no_grad

    cell = load_manifest_cell(Path(args.manifest))
    bs = args.batch_size or cell["batch_size"]

    main_set = set(cell["main_heads"])
    other_heads = [h for h in cell["sig_heads"] if h not in main_set]
    random_sets = sample_head_subsets(
        other_heads, args.n_random, args.set_size, args.seed
    )

    z_train, meta_train = load_zcache_dir(Path("log/cache/z") / cell["z_train_fp"])
    h_bar = mean_z_over_tasks(z_train)  # [n_layers, n_heads, d_model]

    def ablation_values(heads: list) -> dict:
        return {(li, hi): h_bar[li, hi] for li, hi in heads}

    identity = meta_train["identity"]["model"]
    print(f"[model] loading {identity['name']} ({identity['dtype']})", flush=True)
    model, _n_layers, _n_heads = load_model_no_grad(
        identity["name"], args.device, dtype=getattr(torch, identity["dtype"])
    )

    per_task_prompts = collect_prompts(cell["samples"], args.limit_per_task)
    n_prompts = sum(len(p) for p, _ in per_task_prompts.values())
    print(
        f"[cell] {cell['label']}: {n_prompts} five-shot prompts, "
        f"mains={cell['main_heads']}, {len(other_heads)} other significant heads",
        flush=True,
    )

    conditions: dict[str, dict] = {}

    def run_condition(name: str, head_values: dict | None, heads: list | None):
        result = eval_condition(model, per_task_prompts, head_values, bs)
        result["heads"] = [list(h) for h in heads] if heads is not None else None
        result["ci95"] = bootstrap_ci(result["correct"], seed=args.seed)
        conditions[name] = result
        print(
            f"  {name}: acc={result['acc']:.4f} "
            f"ci95=[{result['ci95'][0]:.3f}, {result['ci95'][1]:.3f}]",
            flush=True,
        )

    run_condition("clean", None, None)
    run_condition("null_control", {}, [])
    run_condition("main_trio", ablation_values(cell["main_heads"]), cell["main_heads"])
    for index, subset in enumerate(random_sets):
        run_condition(
            f"random_{index:02d}",
            ablation_values([tuple(h) for h in subset]),
            subset,
        )

    clean_acc = conditions["clean"]["acc"]
    random_accs = [
        conditions[f"random_{i:02d}"]["acc"] for i in range(len(random_sets))
    ]
    summary = {
        "clean_recorded": cell["recorded_clean"],
        "clean_measured": clean_acc,
        "clean_matches_recorded": abs(clean_acc - cell["recorded_clean"]) < 1e-9,
        "null_control_acc": conditions["null_control"]["acc"],
        "trio_acc": conditions["main_trio"]["acc"],
        "trio_over_clean": (
            conditions["main_trio"]["acc"] / clean_acc if clean_acc else None
        ),
        "random_min": min(random_accs) if random_accs else None,
        "random_median": (
            float(sorted(random_accs)[len(random_accs) // 2]) if random_accs else None
        ),
        "random_max": max(random_accs) if random_accs else None,
        "frac_random_ge_0.86": (
            sum(a >= 0.86 for a in random_accs) / len(random_accs)
            if random_accs
            else None
        ),
        "frac_random_within_0.01_of_clean": (
            sum(a >= clean_acc - 0.01 for a in random_accs) / len(random_accs)
            if random_accs
            else None
        ),
    }

    result = {
        "kind": "fiveshot_necessity",
        "impl": IMPL,
        "hook_site": HOOK_SITE,
        "label": cell["label"],
        "model": identity,
        "task": cell["task_cfg"],
        "eval_manifest": cell["manifest_path"],
        "z_cache_train": cell["z_train_fp"],
        "h_bar_tasks": meta_train["tasks"],
        "main_heads": [list(h) for h in cell["main_heads"]],
        "n_significant": len(cell["sig_heads"]),
        "batch_size": bs,
        "limit_per_task": args.limit_per_task,
        "n_prompts": n_prompts,
        "seed": args.seed,
        "git": git_state(),
        "paper_reference": PAPER_REFERENCE,
        "summary": summary,
        "conditions": conditions,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.limit_per_task is None else f"__limit{args.limit_per_task}"
    out_path = (
        out_dir / f"{cell['label']}__{Path(cell['manifest_path']).stem}"
        f"__seed{args.seed}{suffix}.json"
    )
    out_path.write_text(json.dumps(result, indent=1))
    print(
        f"[done] clean={clean_acc:.3f} (recorded {cell['recorded_clean']}) "
        f"null={summary['null_control_acc']:.3f} trio={summary['trio_acc']:.3f} "
        f"random min/med/max={summary['random_min']}/{summary['random_median']}/"
        f"{summary['random_max']} -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
