"""ICL datasets, splits, prompt formatting, and per-batch processing."""

from __future__ import annotations

import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .io import load_task_data


def load_tasks_split_indist_ood(
    task_dir, ood_task_num, args, new_args_path, pre_args_path
):
    """Load tasks under ``task_dir`` and (re)compute in-distribution / OOD splits.

    Mirrors the legacy ``load_tasks_split_indist_ood`` loader byte-for-byte: if
    ``pre_args_path`` is given the dict at that path overrides ``args``; if
    ``indist_keys`` / ``ood_keys`` are missing they are inferred (random sample
    of ``ood_task_num`` tasks for OOD; complement for indist). Final args dict
    is written to ``new_args_path`` when provided.
    """
    tasks = load_task_data(task_dir)
    args_dict = vars(args)
    if pre_args_path:
        if not os.path.exists(pre_args_path):
            raise ValueError(f"Cannot find {pre_args_path} while resume is not None.")
        with open(pre_args_path, "r") as f:
            pre_args_dict = json.load(f)
        for key in pre_args_dict:
            args_dict[key] = pre_args_dict[key]
        print("Rewritten args from", pre_args_path)

    if (
        "indist_keys" in args_dict
        and "ood_keys" in args_dict
        and "ood_task_num" in args_dict
    ):
        if args_dict["indist_keys"] is None and args_dict["ood_keys"] is None:
            args_dict["ood_keys"] = random.sample(
                list(tasks.keys()), args_dict["ood_task_num"]
            )
            args_dict["indist_keys"] = [
                key for key in list(tasks.keys()) if key not in args_dict["ood_keys"]
            ]
            print("Randomly generated ood_keys", args_dict["ood_keys"])
        elif args_dict["indist_keys"] is None and args_dict["ood_keys"] is not None:
            args_dict["indist_keys"] = [
                key for key in list(tasks.keys()) if key not in args_dict["ood_keys"]
            ]
            print("Generated indist_keys from ood_keys", args_dict["indist_keys"])
        elif args_dict["indist_keys"] is not None and args_dict["ood_keys"] is None:
            args_dict["ood_keys"] = [
                key for key in list(tasks.keys()) if key not in args_dict["indist_keys"]
            ]
            print("Generated ood_keys from indist_keys", args_dict["ood_keys"])
        else:
            print("indist_keys and ood_keys are both in args_dict.")
        args_dict["ood_task_num"] = ood_task_num

    if new_args_path:
        with open(new_args_path, "w") as f:
            json.dump(args_dict, f)
        print("Saved args to", new_args_path)
    return tasks, args_dict


class LimitedTaskDataset(Dataset):
    def __init__(self, tasks, n_shot, limit=-1, datapoint_filter=None):
        self.tasks = tasks
        self.n_shot = n_shot
        self.limit = limit
        # Optional predicate ``f(task_name, x_q, task) -> bool``. When given, only
        # query examples for which it returns True are kept (demos x_icl are still
        # sampled from the full task pool). Used by the single-token-answer control.
        self.datapoint_filter = datapoint_filter
        self.datapoints = self.create_all_datapoints()

    def create_all_datapoints(self):
        all_datapoints = []
        while len(all_datapoints) < self.limit or self.limit == -1:
            added_this_pass = 0
            for task_name, task in self.tasks.items():
                task_input = [item["input"] for item in task]
                for i in task:
                    x_q = i["input"]
                    remaining_input = [item for item in task_input if item != x_q]
                    if len(remaining_input) < self.n_shot:
                        continue
                    if self.datapoint_filter is not None and not self.datapoint_filter(
                        task_name, x_q, task
                    ):
                        continue
                    x_icl = random.sample(remaining_input, self.n_shot)
                    all_datapoints.append(
                        {"task_name": task_name, "x_q": x_q, "x_icl": x_icl}
                    )
                    added_this_pass += 1
            if self.limit == -1:
                break
            # Guard against an infinite loop when a filter (or short tasks) leaves
            # fewer passing datapoints than ``limit``: stop once a full pass adds none.
            if added_this_pass == 0:
                break
        if self.limit != -1:
            limit = min(self.limit, len(all_datapoints))
            all_datapoints = random.sample(all_datapoints, limit)
        return all_datapoints

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        return self.datapoints[idx]


def create_train_val_test_splits(
    dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=64
):
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader, val_loader, test_loader = None, None, None
    if len(train_dataset) > 0:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    if len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader


def format_input(x_icl, x_q, task):
    """Render an ICL prompt under the active prompt format (``$FV_PROMPT_FORMAT``).

    Default ``arrow`` reproduces the paper-of-record format byte-for-byte:
    ``input->output#`` per demo, then ``x_q->``. Other formats vary the
    input/output prefixes and separators per Todd et al. (see
    ``subspaces.utils.prompt_formats``).
    """
    from .prompt_formats import get_format, render_demo, render_query

    fmt = get_format()
    formatted = ""
    for example in x_icl:
        example_dict = next((item for item in task if item["input"] == example), None)
        if example_dict:
            formatted += render_demo(example_dict["input"], example_dict["output"], fmt)
    formatted += render_query(x_q, fmt)
    return formatted


def find_output_for_input(task, input_value):
    for item in task:
        if item["input"] == input_value:
            return item["output"]
    return None


def process_batch_data_individual(batch, tasks, FVs_dict):
    batch_FV = []
    batch_prompt = []
    batch_zero_shot_prompt = []
    batch_target = []
    for i in torch.arange(0, len(batch["x_q"])):
        x_icl = []
        for x in batch["x_icl"]:
            x_icl.append(x[i])
        x_q = batch["x_q"][i]
        task_name = batch["task_name"][i]
        task = tasks[task_name]

        prompt = format_input(x_icl, x_q, task)
        zero_shot_prompt = format_input([], x_q, task)
        target = find_output_for_input(task, x_q)

        if target is None:
            continue
        if FVs_dict is not None:
            batch_FV.append(FVs_dict[task_name])
        batch_prompt.append(prompt)
        batch_zero_shot_prompt.append(zero_shot_prompt)
        batch_target.append(target)
    if FVs_dict is not None:
        batch_FV = torch.stack(batch_FV, dim=0)
    else:
        batch_FV = None
    return batch_FV, batch_prompt, batch_zero_shot_prompt, batch_target
