"""File-level helpers: log files, task-JSON loading, checkpoint discovery."""

from __future__ import annotations

import json
import os
import re
import sys

import torch


def create_log_file(log_dir, log_file_name="output.txt"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Create logging directory {log_dir}.")
    else:
        print(f"Continue log into directory {log_dir}.")
    log_path = os.path.join(log_dir, log_file_name)
    sys.stdout = open(log_path, "a")
    sys.stderr = sys.stdout


def load_task_data(task_path):
    """Load every ``*.json`` task under ``task_path`` (or a single file)."""
    tasks = {}
    if os.path.isdir(task_path):
        for filename in os.listdir(task_path):
            if filename.endswith(".json"):
                with open(os.path.join(task_path, filename), "r") as f:
                    tasks[filename[:-5]] = json.load(f)
    elif os.path.isfile(task_path) and task_path.endswith(".json"):
        with open(task_path, "r") as f:
            task_name = os.path.basename(task_path)[:-5]
            tasks[task_name] = json.load(f)
    else:
        raise ValueError("Provided path is neither a directory nor a valid JSON file.")
    return tasks


def sort_files_by_epoch(files):
    """Sort ``matrix_epoch{N}.pth``-style filenames by epoch number."""
    try:
        return sorted(files, key=lambda x: int(re.search(r"epoch(\d+)", x).group(1)))
    except Exception:
        return files


def get_No_checkpoint(run_dir, No, device):
    """Load the No-th checkpoint (``No=-1`` for the latest)."""
    matrix_dir = os.path.join(run_dir, "checkpoints")
    matrix_name = sort_files_by_epoch(os.listdir(matrix_dir))[No]
    matrix_path = os.path.join(matrix_dir, matrix_name)
    matrix = torch.load(matrix_path, map_location=torch.device(device))
    print("Loaded", matrix_path, "to", device)
    return matrix


def get_last_checkpoint(log_dir, device):
    matrix_dir = os.path.join(log_dir, "checkpoints")
    matrix_name = sort_files_by_epoch(os.listdir(matrix_dir))[-1]
    matrix_path = os.path.join(matrix_dir, matrix_name)
    matrix = torch.load(matrix_path, map_location=torch.device(device))
    print("Loaded", matrix_path, "to", device)
    return matrix
