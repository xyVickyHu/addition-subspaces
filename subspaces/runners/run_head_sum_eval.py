"""Paper phase 2 (subset-FV) — evaluate accuracy for subsets of selected heads.

Reads the per-head JSONL produced by ``run_head_eval``, picks the top-K heads
by either coefficient or accuracy, sweeps a shared scalar across the subset,
and reports best-scalar / best-accuracy for the chosen subset evaluated against
in-distribution + OOD tasks.

Ported from ``head_sum_eval.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from subspaces.runners.run_head_eval import infer_task_dir_name, precompute_head_vectors
from subspaces.utils.activations import load_z_results_dict
from subspaces.utils.intervene import compute_task_accuracy
from subspaces.utils.io import load_task_data
from subspaces.utils.model import load_model_no_grad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the top heads discovered by head_eval, sweep shared scalars over a range, "
            "and evaluate cumulative FVs for increasing subsets of heads."
        )
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Matrix log directory containing the head-eval JSONL (required).",
    )
    parser.add_argument("--head_eval_name", default="head_eval.jsonl")
    parser.add_argument("--top_signal", default="coefficient")
    parser.add_argument("--top_threshold", type=float, default=0.1)
    parser.add_argument("--subset_signal", default="accuracy")
    parser.add_argument("--subset_threshold", type=float, default=0.09)
    parser.add_argument("--scalar_min", type=int, default=0)
    parser.add_argument("--scalar_max", type=int, default=10)
    parser.add_argument("--project_dim", type=int, default=None)
    parser.add_argument("--other_signal", default="mean")
    parser.add_argument("--task_dir_name", default=None)
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--layer_name", default="blocks.10.hook_resid_mid")
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_example", type=int, default=100)
    parser.add_argument("--test_limit_per_task", type=int, default=10)
    parser.add_argument("--bs", type=int, default=None)
    parser.add_argument("--output_name", default=None)
    return parser.parse_args()


def load_head_eval_entries(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Head-eval file not found: {path}")
    entries: List[Dict] = []
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def select_top_heads(
    entries: List[Dict], threshold: float, order_signal: str
) -> List[Tuple[int, int]]:
    filtered = [
        entry
        for entry in entries
        if entry.get(order_signal) is not None and entry[order_signal] > threshold
    ]
    filtered.sort(key=lambda e: e[order_signal], reverse=True)
    return [tuple(entry["head position"]) for entry in filtered]


def project_head_vectors(head_vectors, dim, device):
    projected = {}
    for head, vectors in head_vectors.items():
        task_names = list(vectors.keys())
        matrices = {}
        for name in task_names:
            array = vectors[name].detach().to(torch.float32).cpu().numpy()
            matrices[name] = array.reshape(1, -1)

        feature_dim = matrices[task_names[0]].shape[1]
        n_components = min(dim, feature_dim)
        if n_components < 1:
            raise ValueError("project_dim must be at least 1.")

        X = np.concatenate([matrices[name] for name in task_names], axis=0)
        pca = PCA(n_components=n_components)
        pca.fit(X)

        head_proj = {}
        for name in task_names:
            transformed = pca.transform(matrices[name])
            reconstructed = pca.inverse_transform(transformed).reshape(-1)
            head_proj[name] = torch.tensor(
                reconstructed, dtype=vectors[name].dtype, device=device
            )
        projected[head] = head_proj

    return projected


def evaluate_head_subset(
    subset_heads,
    top_heads,
    scalar_values,
    other_signal,
    head_vectors,
    project_dim,
    mean_head_vectors,
    indist_task_names,
    ood_task_names,
    tasks,
    model,
    layer_name,
    n_shot,
    test_limit_per_task,
    batch_size,
    device,
):
    best_scalar = scalar_values[0]
    best_accuracy = float("-inf")
    indist_accuracy = float("-inf")
    ood_accuracy = float("-inf")
    task_names = indist_task_names + ood_task_names
    accuracy_per_task = {task_name: float("-inf") for task_name in task_names}

    if project_dim is not None:
        head_vectors = project_head_vectors(head_vectors, project_dim, device)

    for scalar in scalar_values:
        FVs_dict: Dict[str, torch.Tensor] = {}
        for task_name in task_names:
            base_vector = torch.zeros_like(head_vectors[top_heads[0]][task_name])
            for head in top_heads:
                if head in subset_heads:
                    base_vector = base_vector + scalar * head_vectors[head][task_name]
                else:
                    if other_signal == "mean":
                        base_vector = base_vector + mean_head_vectors[head]
            FVs_dict[task_name] = base_vector

        acc_total = 0.0
        indist_acc_total = 0.0
        ood_acc_total = 0.0
        acc_per_task = {}
        for task_name in task_names:
            (
                _,
                _,
                intervened_acc,
                _,
                _,
                *_,
            ) = compute_task_accuracy(
                task_name,
                tasks,
                n_shot,
                test_limit_per_task,
                batch_size,
                FVs_dict,
                None,
                None,
                None,
                model,
                layer_name,
                intervene=True,
                clean=False,
                corrupted=False,
                print_intervened_num=0,
            )
            acc_total += intervened_acc
            if task_name in indist_task_names:
                indist_acc_total += intervened_acc
            else:
                ood_acc_total += intervened_acc
            acc_per_task[task_name] = intervened_acc

        avg_acc = acc_total / len(task_names)
        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            best_scalar = scalar
            accuracy_per_task = acc_per_task.copy()
            indist_accuracy = indist_acc_total / len(indist_task_names)
            ood_accuracy = ood_acc_total / len(ood_task_names)

    return best_scalar, best_accuracy, accuracy_per_task, indist_accuracy, ood_accuracy


def main() -> None:
    args = parse_args()
    if args.log_dir is None:
        raise SystemExit(
            "--log_dir is required: pass the matrix log directory holding the "
            "head-eval JSONL (see docs/REPRODUCING.md, phase 2)."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.bs or args.test_limit_per_task
    scalar_values = list(range(args.scalar_min, args.scalar_max + 1))
    print(f"Scalar values: {scalar_values}")
    if not scalar_values:
        raise ValueError(
            "Scalar range is empty; please adjust scalar_min and scalar_max."
        )

    head_eval_path = os.path.join(args.log_dir, args.head_eval_name)
    head_entries = load_head_eval_entries(head_eval_path)
    top_heads = select_top_heads(head_entries, args.top_threshold, args.top_signal)
    subset_heads = select_top_heads(
        head_entries, args.subset_threshold, args.subset_signal
    )

    task_dir_name = args.task_dir_name or infer_task_dir_name(args.log_dir)
    print(f"Using task directory: {task_dir_name}")
    data_dir = PROJECT_ROOT / "dataset_files" / task_dir_name
    version_dir = PROJECT_ROOT / "log" / task_dir_name
    savevar_dir = version_dir / "savevars"
    config_path = Path(args.log_dir) / "args.json"
    with open(config_path, "r", encoding="utf-8") as infile:
        config = json.load(infile)
    indist_task_names = config["indist_keys"]
    ood_task_names = config["ood_keys"]

    tasks = load_task_data(str(data_dir))
    z_results_dict = load_z_results_dict(
        None,
        tasks,
        args.n_shot,
        args.n_example,
        str(savevar_dir),
        task_dir_name,
        device=device,
    )
    z_results_dict = {k: v.to(device) for k, v in z_results_dict.items()}
    mean_z_results = torch.mean(torch.stack(list(z_results_dict.values())), dim=0).to(
        device
    )

    model, _, _ = load_model_no_grad(args.model_name, device)

    head_vectors = precompute_head_vectors(top_heads, z_results_dict, device)
    mean_head_vectors = {
        head: mean_z_results[head[0], head[1]].to(device) for head in top_heads
    }

    output_path = Path(args.log_dir) / (args.output_name or "head_variants_eval.jsonl")
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as infile:
            results = [json.loads(line) for line in infile if line.strip()]
    else:
        results: List[Dict] = []

    best_scalar, best_accuracy, accuracy_per_task, indist_accuracy, ood_accuracy = (
        evaluate_head_subset(
            subset_heads,
            top_heads,
            scalar_values,
            args.other_signal,
            head_vectors,
            args.project_dim,
            mean_head_vectors,
            indist_task_names,
            ood_task_names,
            tasks,
            model,
            args.layer_name,
            args.n_shot,
            args.test_limit_per_task,
            batch_size,
            device,
        )
    )
    result_entry = {
        "heads": [list(head) for head in subset_heads],
        "project_dim": args.project_dim,
        "subset_signal": args.subset_signal,
        "subset_threshold": args.subset_threshold,
        "scalar_max": args.scalar_max,
        "other_heads": [list(head) for head in top_heads if head not in subset_heads],
        "top_signal": args.top_signal,
        "top_threshold": args.top_threshold,
        "other_signal": args.other_signal,
        "best_scalar": best_scalar,
        "best_accuracy": best_accuracy,
        "indist_accuracy": indist_accuracy,
        "ood_accuracy": ood_accuracy,
        "accuracy_per_task": accuracy_per_task,
    }
    results.append(result_entry)
    print(
        f"Top-{len(subset_heads)} heads: best scalar {best_scalar}, accuracy {best_accuracy:.4f}, "
        f"other_signal {args.other_signal}"
    )
    with open(output_path, "w", encoding="utf-8") as outfile:
        for entry in results:
            outfile.write(json.dumps(entry))
            outfile.write("\n")
    print(f"Saved {len(results)} aggregated evaluations to {output_path}.")


if __name__ == "__main__":
    main()
