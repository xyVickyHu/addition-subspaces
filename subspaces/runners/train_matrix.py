"""Paper phase 1 — head-selection optimization training.

Trains a sparse coefficient matrix ``M`` ∈ ℝ^{n_layers × n_heads} so that
``FV_k = Σ M[l,h] · z_k[l,h]`` recovers per-task accuracy when injected at
``--layer_name``. L1 regularization is applied via proximal-gradient
(shrinkage) so coefficients converge to exact zeros, and ``--zero_one_interval``
clips to ``[0, 1]`` for the paper's canonical matrix.

Default invocation reproduces the paper-baseline training run on
``meta-llama/Meta-Llama-3-8B-Instruct`` + ``number_add``.

Ported from the legacy ``FV_matrix.py`` (which is now a backwards-compat
shim). Behavioral diffs vs. the legacy script: ``PROJECT_ROOT`` resolves
from this file's path instead of ``__file__``-relative; the WandB entity
comes from ``$WANDB_ENTITY`` (unset → wandb's default entity for the
logged-in account).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from subspaces.utils.activations import compute_FVs_dict, load_z_results_dict
from subspaces.utils.data import (
    LimitedTaskDataset,
    create_train_val_test_splits,
    load_tasks_split_indist_ood,
    process_batch_data_individual,
)
from subspaces.utils.intervene import (
    intervened_generation_with_accuracy,
    intervened_generation_with_nll,
)
from subspaces.utils.io import create_log_file, sort_files_by_epoch
from subspaces.utils.model import load_model_no_grad

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Matrix for Individual Tasks")
    parser.add_argument(
        "--version", type=str, default="1118_mixed_1008", help="Version"
    )
    parser.add_argument("--run_name", type=str, default=None, help="Run Name")
    parser.add_argument(
        "--savevar", action="store_true", help="Save z_results_dict or not"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume with wandb run ID"
    )
    parser.add_argument(
        "--nonnegative", action="store_true", help="Nonnegative Coefficients"
    )
    parser.add_argument(
        "--zero_one_interval", action="store_true", help="Coefficients are all in [0,1]"
    )
    parser.add_argument(
        "--add_mean",
        type=int,
        default=0,
        help="The mode for adding mean_z_result to FVs. 0: FV_k=c*h_k; 1: c*h_k + α(1-c)·h̄ (scalar α); 2: c*h_k + d(1-c)·h̄ (matrix d)",
    )
    parser.add_argument(
        "--lambdaL1", type=float, default=0.05, help="L1 regularization weight"
    )
    parser.add_argument("--anneal", type=int, default=1, help="Anneal schedule for λ")
    parser.add_argument(
        "--task_dir", type=str, default="number_add", help="Task Directory Name"
    )
    parser.add_argument(
        "--indist_keys", type=str, default=None, help="Indist Keys (JSON list)"
    )
    parser.add_argument(
        "--ood_keys", type=str, default=None, help="OOD Keys (JSON list)"
    )
    parser.add_argument(
        "--ood_task_num", type=int, default=5, help="Number of OOD Tasks"
    )
    parser.add_argument("--indist_limit", type=int, default=2560)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--ood_limit", type=int, default=512)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epoch_num", type=int, default=50)
    parser.add_argument("--test_gap", type=int, default=5)
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_example", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corrupted", action="store_true")
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument("--layer_name", type=str, default="blocks.10.hook_resid_mid")
    return parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = parse_args()
    if (
        args.indist_keys
        and isinstance(args.indist_keys, str)
        and args.indist_keys not in ("None", "", "null")
    ):
        args.indist_keys = json.loads(args.indist_keys)
    else:
        args.indist_keys = None
    if (
        args.ood_keys
        and isinstance(args.ood_keys, str)
        and args.ood_keys not in ("None", "", "null")
    ):
        args.ood_keys = json.loads(args.ood_keys)
    else:
        args.ood_keys = None
    random.seed(args.seed)
    torch.manual_seed(seed=args.seed)

    data_dir = PROJECT_ROOT / "dataset_files"
    version_dir = PROJECT_ROOT / "log" / args.version
    run_dir = version_dir if args.run_name is None else version_dir / args.run_name
    savevar_dir = str(version_dir / "savevars") if args.savevar else None
    create_log_file(str(run_dir), "output.txt")

    task_dir = str(data_dir / args.task_dir)
    tasks, args_dict = load_tasks_split_indist_ood(
        task_dir,
        args.ood_task_num,
        args,
        new_args_path=str(run_dir / "args.json"),
        pre_args_path=None,
    )

    indist_tasks = {key: tasks[key] for key in args_dict["indist_keys"]}
    ood_tasks = {key: tasks[key] for key in args_dict["ood_keys"]}
    dataset = LimitedTaskDataset(indist_tasks, args.n_shot, args.indist_limit)
    train_loader, val_loader, test_loader = create_train_val_test_splits(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=0,
        test_ratio=1 - args.train_ratio,
        batch_size=args.bs,
    )
    ood_dataset = LimitedTaskDataset(ood_tasks, args.n_shot, args.ood_limit)
    ood_loader = DataLoader(ood_dataset, batch_size=args.bs, shuffle=True)
    print(
        "train_loader",
        len(train_loader),
        "test_loader",
        len(test_loader),
        "ood_loader",
        len(ood_loader),
    )
    torch.cuda.empty_cache()

    model, n_layers, n_heads = load_model_no_grad(args.model_name, device)
    try:
        model.set_use_checkpoint(True)
        print("Activation checkpointing enabled.")
    except AttributeError:
        print("Warning: this model does not support set_use_checkpoint, skipping.")

    torch.cuda.empty_cache()
    z_results_dict = load_z_results_dict(
        model,
        tasks,
        args.n_shot,
        args.n_example,
        savevar_dir,
        args.task_dir,
        device=device,
    )

    if args.resume:
        checkpoint_dir = version_dir / "checkpoints"
        if not checkpoint_dir.exists():
            raise ValueError("Please provide a correct matrix_dir")
        matrix_file = sort_files_by_epoch(os.listdir(str(checkpoint_dir)))[-1]
        matrix = torch.load(str(checkpoint_dir / matrix_file), map_location=device)
        start_epoch = int(matrix_file[len("matrix_epoch") : -len(".pth")]) + 1
        print("Loaded", matrix_file, "from", checkpoint_dir)
    else:
        matrix = torch.rand(n_layers, n_heads, device=device, requires_grad=True)
        start_epoch = 0
        print("Randomly initialized matrix in [0,1)")

    matrix.requires_grad = True
    if args.add_mean == 1:
        mean_ratio = torch.rand(1, device=device, requires_grad=True)
        mean_matrix = mean_ratio * torch.ones(
            n_layers, n_heads, device=device, requires_grad=True
        )
    elif args.add_mean == 2:
        mean_matrix = torch.randn(n_layers, n_heads, device=device, requires_grad=True)
    else:
        mean_matrix = None

    if mean_matrix is not None:
        optimizer = torch.optim.AdamW(
            [{"params": [matrix, mean_matrix], "lr": args.lr, "weight_decay": 0.0}],
            betas=(0.9, 0.98),
            eps=1e-8,
            amsgrad=False,
        )
    else:
        optimizer = torch.optim.AdamW(
            [{"params": [matrix], "lr": args.lr, "weight_decay": 0.0}],
            betas=(0.9, 0.98),
            eps=1e-8,
            amsgrad=False,
        )

    if args.resume:
        # resume="allow" (not "must") so checkpoint-resume works under WANDB_MODE=offline
        # even when the original (silent/offline) run id isn't recoverable — the training
        # restart point (start_epoch) comes from the checkpoint, independent of wandb.
        wandb.init(
            project="function-vector-subspace",
            id=args.resume,
            resume="allow",
            entity=WANDB_ENTITY,
        )
    else:
        wandb.login()
        if args.run_name:
            wandb.init(
                project="function-vector-subspace",
                name=f"{args.version}-{args.run_name}",
                config=args_dict,
                entity=WANDB_ENTITY,
            )
        else:
            wandb.init(
                project="function-vector-subspace",
                name=args.version,
                config=args_dict,
                entity=WANDB_ENTITY,
            )

    lambda_target = args.lambdaL1

    def get_lambda(epoch, anneal):
        if anneal == 2:
            if epoch < int(2 / 3 * args.epoch_num):
                return 0
            else:
                return (
                    lambda_target
                    * (epoch - int(2 / 3 * args.epoch_num))
                    / (args.epoch_num - int(2 / 3 * args.epoch_num))
                )
        elif anneal == 3:
            if epoch < int(1 / 3 * args.epoch_num):
                return 0
            else:
                return (
                    lambda_target
                    * (epoch - int(1 / 3 * args.epoch_num))
                    / (args.epoch_num - int(1 / 3 * args.epoch_num))
                )
        else:
            return lambda_target

    epoch_clean_accuracy_list = []
    epoch_corrupted_accuracy_list = []
    epoch_intervened_accuracy_list = []
    epoch_clean_accuracy_ood_list = []
    epoch_corrupted_accuracy_ood_list = []
    epoch_intervened_accuracy_ood_list = []
    for epoch_cnt in range(start_epoch, args.epoch_num):
        time0 = time.time()
        batch_cnt = 0
        eval_cnt = 0
        epoch_clean_accuracy = 0
        epoch_corrupted_accuracy = 0
        epoch_intervened_accuracy = 0
        epoch_clean_accuracy_ood = 0
        epoch_corrupted_accuracy_ood = 0
        epoch_intervened_accuracy_ood = 0
        for batch in train_loader:
            time1 = time.time()
            optimizer.zero_grad()
            FVs_dict = compute_FVs_dict(
                z_results_dict, matrix, device=device, mean_matrix=mean_matrix
            )
            batch_FV, batch_prompt, batch_zero_shot_prompt, batch_target = (
                process_batch_data_individual(batch, tasks, FVs_dict)
            )
            intervened_nll = intervened_generation_with_nll(
                model,
                batch_zero_shot_prompt,
                batch_target,
                args.layer_name,
                batch_FV,
                intervention_mode=0,
            )
            loss = intervened_nll + get_lambda(epoch_cnt, args.anneal) * torch.sum(
                torch.abs(matrix)
            )
            if mean_matrix is not None:
                loss += get_lambda(epoch_cnt, args.anneal) * torch.sum(
                    torch.abs(mean_matrix)
                )
            torch.cuda.empty_cache()
            if batch_cnt % args.test_gap == 0:
                with torch.no_grad():
                    test_batch_cnt = 0
                    ood_batch_cnt = 0
                    batch_clean_accuracy = 0
                    batch_corrupted_accuracy = 0
                    batch_intervened_accuracy = 0
                    batch_clean_accuracy_ood = 0
                    batch_corrupted_accuracy_ood = 0
                    batch_intervened_accuracy_ood = 0
                    for test_batch in test_loader:
                        (
                            test_batch_FV,
                            test_batch_prompt,
                            test_batch_zero_shot_prompt,
                            test_batch_target,
                        ) = process_batch_data_individual(test_batch, tasks, FVs_dict)
                        clean_accuracy = intervened_generation_with_accuracy(
                            model, test_batch_prompt, test_batch_target
                        )
                        if args.corrupted:
                            corrupted_accuracy = intervened_generation_with_accuracy(
                                model, test_batch_zero_shot_prompt, test_batch_target
                            )
                        else:
                            corrupted_accuracy = 0
                        intervened_accuracy = intervened_generation_with_accuracy(
                            model,
                            test_batch_zero_shot_prompt,
                            test_batch_target,
                            args.layer_name,
                            test_batch_FV,
                            intervention_mode=0,
                        )
                        batch_clean_accuracy += clean_accuracy
                        batch_corrupted_accuracy += corrupted_accuracy
                        batch_intervened_accuracy += intervened_accuracy
                        test_batch_cnt += 1
                    for ood_batch in ood_loader:
                        (
                            ood_batch_FV,
                            ood_batch_prompt,
                            ood_batch_zero_shot_prompt,
                            ood_batch_target,
                        ) = process_batch_data_individual(
                            ood_batch, ood_tasks, FVs_dict
                        )
                        clean_accuracy_ood = intervened_generation_with_accuracy(
                            model, ood_batch_prompt, ood_batch_target
                        )
                        if args.corrupted:
                            corrupted_accuracy_ood = (
                                intervened_generation_with_accuracy(
                                    model, ood_batch_zero_shot_prompt, ood_batch_target
                                )
                            )
                        else:
                            corrupted_accuracy_ood = 0
                        intervened_accuracy_ood = intervened_generation_with_accuracy(
                            model,
                            ood_batch_zero_shot_prompt,
                            ood_batch_target,
                            args.layer_name,
                            ood_batch_FV,
                            intervention_mode=0,
                        )
                        batch_clean_accuracy_ood += clean_accuracy_ood
                        batch_corrupted_accuracy_ood += corrupted_accuracy_ood
                        batch_intervened_accuracy_ood += intervened_accuracy_ood
                        ood_batch_cnt += 1
                    batch_clean_accuracy /= test_batch_cnt
                    batch_corrupted_accuracy /= test_batch_cnt
                    batch_intervened_accuracy /= test_batch_cnt
                    batch_clean_accuracy_ood /= ood_batch_cnt
                    batch_corrupted_accuracy_ood /= ood_batch_cnt
                    batch_intervened_accuracy_ood /= ood_batch_cnt
                    epoch_clean_accuracy += batch_clean_accuracy
                    epoch_corrupted_accuracy += batch_corrupted_accuracy
                    epoch_intervened_accuracy += batch_intervened_accuracy
                    epoch_clean_accuracy_ood += batch_clean_accuracy_ood
                    epoch_corrupted_accuracy_ood += batch_corrupted_accuracy_ood
                    epoch_intervened_accuracy_ood += batch_intervened_accuracy_ood
                    time2 = time.time()
                    print(
                        f"epoch-batch {epoch_cnt} {batch_cnt} "
                        f"intervened_nll {intervened_nll.item():.4f} "
                        f"intervened_acc {batch_intervened_accuracy:.4f} "
                        f"clean_acc {batch_clean_accuracy:.4f} "
                        f"corrupted_acc {batch_corrupted_accuracy:.4f} "
                        f"ood_clean_acc {batch_clean_accuracy_ood:.4f} "
                        f"ood_corrupted_acc {batch_corrupted_accuracy_ood:.4f} "
                        f"ood_intervened_acc {batch_intervened_accuracy_ood:.4f} "
                        f"batch time {(time2 - time1):.4f}",
                        flush=True,
                    )
                    wandb.log(
                        {
                            "intervened_nll": intervened_nll.item(),
                            "intervened_acc": batch_intervened_accuracy,
                            "clean_acc": batch_clean_accuracy,
                            "corrupted_acc": batch_corrupted_accuracy,
                            "ood_clean_acc": batch_clean_accuracy_ood,
                            "ood_corrupted_acc": batch_corrupted_accuracy_ood,
                            "ood_intervened_acc": batch_intervened_accuracy_ood,
                        }
                    )
                    eval_cnt += 1
                    torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                # Proximal-gradient L1 shrinkage: pushes small coefficients to exact zero.
                shrink = (
                    get_lambda(epoch_cnt, args.anneal) * optimizer.param_groups[0]["lr"]
                )
                matrix.data.copy_(
                    matrix.data.sign()
                    * torch.clamp(matrix.data.abs() - shrink, min=0.0)
                )
                if mean_matrix is not None:
                    mean_matrix.data.copy_(
                        mean_matrix.data.sign()
                        * torch.clamp(mean_matrix.data.abs() - shrink, min=0.0)
                    )
                    mean_matrix.data.clamp_(0.0, 1.0)
                if args.nonnegative:
                    matrix.data.clamp_(min=0.0)
                if args.zero_one_interval:
                    matrix.data.clamp_(0.0, 1.0)
            batch_cnt += 1
            torch.cuda.empty_cache()

        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(matrix, str(checkpoint_dir / f"matrix_epoch{epoch_cnt}.pth"))
        if mean_matrix is not None:
            torch.save(
                mean_matrix, str(checkpoint_dir / f"mean_matrix_epoch{epoch_cnt}.pth")
            )
        with torch.no_grad():
            sparsity = 1.0 - (matrix != 0).float().mean().item()
            sparsity_mean = (
                1.0 - (mean_matrix != 0).float().mean().item()
                if mean_matrix is not None
                else 0
            )
        time3 = time.time()
        print(
            f"epoch {epoch_cnt} "
            f"epoch_clean_acc {(epoch_clean_accuracy / eval_cnt):.4f} "
            f"epoch_corrupted_acc {(epoch_corrupted_accuracy / eval_cnt):.4f} "
            f"epoch_intervened_acc {(epoch_intervened_accuracy / eval_cnt):.4f} "
            f"epoch_clean_acc_ood {(epoch_clean_accuracy_ood / eval_cnt):.4f} "
            f"epoch_corrupted_acc_ood {(epoch_corrupted_accuracy_ood / eval_cnt):.4f} "
            f"epoch_intervened_acc_ood {(epoch_intervened_accuracy_ood / eval_cnt):.4f} "
            f"lambda {get_lambda(epoch_cnt, args.anneal):.4f} "
            f"sparsity {sparsity:.4f} "
            f"sparsity_mean {sparsity_mean:.4f} "
            f"epoch time {(time3 - time0):.4f}",
            flush=True,
        )
        wandb.log(
            {
                "epoch": epoch_cnt,
                "epoch_clean_acc": (epoch_clean_accuracy / eval_cnt),
                "epoch_corrupted_acc": (epoch_corrupted_accuracy / eval_cnt),
                "epoch_intervened_acc": (epoch_intervened_accuracy / eval_cnt),
                "epoch_clean_acc_ood": (epoch_clean_accuracy_ood / eval_cnt),
                "epoch_corrupted_acc_ood": (epoch_corrupted_accuracy_ood / eval_cnt),
                "epoch_intervened_acc_ood": (epoch_intervened_accuracy_ood / eval_cnt),
                "lambda": get_lambda(epoch_cnt, args.anneal),
                "sparsity": sparsity,
                "sparsity_mean": sparsity_mean,
                "epoch_time": (time3 - time0),
            }
        )
        epoch_clean_accuracy_list.append(epoch_clean_accuracy / eval_cnt)
        epoch_corrupted_accuracy_list.append(epoch_corrupted_accuracy / eval_cnt)
        epoch_intervened_accuracy_list.append(epoch_intervened_accuracy / eval_cnt)
        epoch_clean_accuracy_ood_list.append(epoch_clean_accuracy_ood / eval_cnt)
        epoch_corrupted_accuracy_ood_list.append(
            epoch_corrupted_accuracy_ood / eval_cnt
        )
        epoch_intervened_accuracy_ood_list.append(
            epoch_intervened_accuracy_ood / eval_cnt
        )
        torch.cuda.empty_cache()

    accuracy_dict = {
        "epoch_clean_accuracy_list": epoch_clean_accuracy_list,
        "epoch_corrupted_accuracy_list": epoch_corrupted_accuracy_list,
        "epoch_intervened_accuracy_list": epoch_intervened_accuracy_list,
        "epoch_clean_accuracy_ood_list": epoch_clean_accuracy_ood_list,
        "epoch_corrupted_accuracy_ood_list": epoch_corrupted_accuracy_ood_list,
        "epoch_intervened_accuracy_ood_list": epoch_intervened_accuracy_ood_list,
    }
    torch.save(accuracy_dict, str(run_dir / "accuracy_dict.pth"))


if __name__ == "__main__":
    main()
