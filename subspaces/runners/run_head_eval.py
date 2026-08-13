"""Paper phase 2 — per-head significance evaluation.

For each head selected above ``--threshold`` from the trained matrix, sweep a
scalar coefficient and measure intervention accuracy when that head is scaled
while every other selected head is held at zero (``--other_signal zero``) or
its across-task mean (``--other_signal mean``). The best (scalar, accuracy)
pair per head is logged into the JSONL artifact.

Ported from ``head_eval.py`` (kept as a backwards-compat shim).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from subspaces.runners.run_head_select import write_jsonl
from subspaces.utils.activations import load_z_results_dict
from subspaces.utils.heads import extract_heads_above_threshold, load_latest_matrix
from subspaces.utils.intervene import compute_task_accuracy
from subspaces.utils.io import load_task_data
from subspaces.utils.model import load_model_no_grad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end pipeline that selects large matrix heads, writes them to JSON, "
            "and evaluates each head to find the best scalar/accuracy pair."
        )
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Path to the log directory produced by subspaces.runners.train_matrix (required).",
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--output_name", default="head_eval.jsonl")
    parser.add_argument("--task_dir_name", default=None)
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--layer_name", default="blocks.10.hook_resid_mid")
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_example", type=int, default=100)
    parser.add_argument("--test_limit_per_task", type=int, default=10)
    parser.add_argument("--bs", type=int, default=None)
    parser.add_argument("--scalar_min", type=int, default=0)
    parser.add_argument("--scalar_max", type=int, default=14)
    parser.add_argument("--other_signal", choices=["zero", "mean"], default="zero")
    return parser.parse_args()


def infer_task_dir_name(log_dir: str) -> str:
    log_path = Path(log_dir)
    if log_path.parent.name != "log":
        return log_path.parent.name
    return log_path.name


def build_head_entries(
    heads_with_coeffs: Sequence[Tuple[Tuple[int, int], float]],
) -> List[Dict]:
    return [
        {"head position": [row, col], "coefficient": float(coeff)}
        for (row, col), coeff in heads_with_coeffs
    ]


def precompute_head_vectors(head_positions, z_results_dict, device):
    vectors: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
    for head in head_positions:
        per_task: Dict[str, torch.Tensor] = {}
        for task_name, z_results in z_results_dict.items():
            per_task[task_name] = z_results[head[0], head[1]].to(device)
        vectors[head] = per_task
    return vectors


def evaluate_single_head(
    target_head,
    selected_heads,
    head_vectors,
    mean_head_vectors,
    task_names,
    coefficient_values,
    other_signal,
    tasks,
    model,
    layer_name,
    n_shot,
    test_limit,
    batch_size,
):
    best_scalar = coefficient_values[0]
    best_accuracy = float("-inf")

    for coeff in coefficient_values:
        FVs_dict: Dict[str, torch.Tensor] = {}
        for task_name in task_names:
            base_vector = torch.zeros_like(head_vectors[target_head][task_name])
            for head in selected_heads:
                if head == target_head:
                    base_vector = base_vector + coeff * head_vectors[head][task_name]
                else:
                    if other_signal == "mean":
                        base_vector = base_vector + mean_head_vectors[head]
            FVs_dict[task_name] = base_vector

        acc_total = 0.0
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
                test_limit,
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
        avg_acc = acc_total / len(task_names)
        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            best_scalar = coeff

    return best_scalar, best_accuracy


def main() -> None:
    args = parse_args()
    if args.log_dir is None:
        raise SystemExit(
            "--log_dir is required: pass the matrix log directory produced by "
            "subspaces.runners.train_matrix (it must contain a checkpoints/ subdirectory), "
            "e.g. --log_dir log/<task>/<run_name>."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.bs or args.test_limit_per_task

    matrix, checkpoint_path = load_latest_matrix(args.log_dir)
    selected_heads = extract_heads_above_threshold(matrix, args.threshold)
    head_entries = build_head_entries(selected_heads)
    head_path = os.path.join(args.log_dir, args.output_name)
    write_jsonl(head_entries, head_path)
    print(
        f"Saved {len(head_entries)} heads above threshold {args.threshold} "
        f"from checkpoint {checkpoint_path} into {head_path}."
    )

    if not head_entries:
        print("No heads selected; skipping evaluation.")
        return

    task_dir_name = args.task_dir_name or infer_task_dir_name(args.log_dir)
    print(f"Task directory name: {task_dir_name}")
    data_dir = PROJECT_ROOT / "dataset_files" / task_dir_name
    version_dir = PROJECT_ROOT / "log" / task_dir_name
    savevar_dir = version_dir / "savevars"

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

    head_positions = [tuple(entry["head position"]) for entry in head_entries]
    head_vectors = precompute_head_vectors(head_positions, z_results_dict, device)
    mean_head_vectors = {
        head: mean_z_results[head[0], head[1]].to(device) for head in head_positions
    }
    task_names = sorted(tasks.keys())
    coefficient_values = list(range(args.scalar_min, args.scalar_max + 1))
    if not coefficient_values:
        raise ValueError("No scalar values to evaluate; adjust scalar_min/max.")

    results = {}
    for head in head_positions:
        scalar, accuracy = evaluate_single_head(
            head,
            head_positions,
            head_vectors,
            mean_head_vectors,
            task_names,
            coefficient_values,
            args.other_signal,
            tasks,
            model,
            args.layer_name,
            args.n_shot,
            args.test_limit_per_task,
            batch_size,
        )
        results[head] = (scalar, accuracy)
        print(f"Head {head}: best scalar {scalar}, accuracy {accuracy:.4f}")

    for entry in head_entries:
        head_tuple = tuple(entry["head position"])
        scalar, accuracy = results[head_tuple]
        entry["scalar"] = scalar
        entry["accuracy"] = accuracy

    write_jsonl(head_entries, head_path)
    print(
        f"Updated {head_path} with scalar and accuracy for {len(head_entries)} heads."
    )


if __name__ == "__main__":
    main()
