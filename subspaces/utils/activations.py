"""Per-task head-output (``z``) extraction, caching, and FV builders."""

from __future__ import annotations

import os
import random

import torch

from .data import format_input


def compute_z_result_per_task(model, tasks, task_name, n_shot, n_example):
    """Mean last-token attn output per layer/head, averaged over n_example ICL prompts."""
    task = tasks[task_name]
    del tasks
    task_input = [item["input"] for item in task]
    batch_prompt = []

    for _ in range(n_example):
        x_q = random.choice(task_input)
        remaining_input = [item for item in task_input if item != x_q]
        x_icl = random.sample(remaining_input, n_shot)
        prompt = format_input(x_icl, x_q, task)
        batch_prompt.append(prompt)

    chunk_size = 1  # tuned for memory headroom on A100s
    all_chunks_activations = []

    for i in range(0, n_example, chunk_size):
        chunk_prompts = batch_prompt[i : i + chunk_size]
        chunk_tokens = model.to_tokens(
            chunk_prompts, prepend_bos=True, padding_side="left"
        )
        chunk_layer_activations = []

        def head_activation_hook(activation, hook):
            last_token_mean_activation = torch.mean(activation[:, -1, :, :], dim=0)
            chunk_layer_activations.append(last_token_mean_activation)

        hook_name_filter = lambda name: name.endswith("attn.hook_result")

        with torch.no_grad():
            model.cfg.use_attn_result = True
            model.run_with_hooks(
                chunk_tokens,
                return_type=None,
                fwd_hooks=[(hook_name_filter, head_activation_hook)],
                attention_mask=(chunk_tokens != model.tokenizer.pad_token_id),
            )
            model.cfg.use_attn_result = False
            del chunk_tokens
            torch.cuda.empty_cache()

        chunk_stacked = torch.stack(chunk_layer_activations, dim=0)
        all_chunks_activations.append(chunk_stacked)

    all_chunks_stacked = torch.stack(all_chunks_activations, dim=0)
    all_layer_mean_activation = torch.mean(all_chunks_stacked, dim=0)
    del all_chunks_stacked
    torch.cuda.empty_cache()
    return all_layer_mean_activation


def compute_z_results_dict(model, tasks, n_shot, n_example, device):
    z_results_dict = {}
    for task_name in tasks.keys():
        z_results_dict[task_name] = compute_z_result_per_task(
            model, tasks, task_name, n_shot, n_example
        ).to(device)
        print(
            "Computed z results for task",
            task_name,
            "with shape",
            z_results_dict[task_name].shape,
        )
    return z_results_dict


def load_z_results_dict(
    model, tasks, n_shot, n_example, savevar_dir, task_dir_name, device="cpu"
):
    """Cache-aware z-results loader. Cache file is task+shot+format-keyed (no model info!).

    The active prompt format (``$FV_PROMPT_FORMAT``) is appended to the cache
    filename so per-format activations never collide; the default ``arrow``
    format keeps its legacy untagged name for backwards compatibility.
    """
    from .prompt_formats import format_tag

    if savevar_dir:
        os.makedirs(savevar_dir, exist_ok=True)
        z_results_dict_path = os.path.join(
            savevar_dir,
            f"z_results_dict_{task_dir_name}_shot{n_shot}{format_tag()}.pth",
        )
        print("z_results_dict_path", z_results_dict_path)
        if os.path.exists(z_results_dict_path):
            z_results_dict = torch.load(z_results_dict_path, map_location=device)
            print("Loaded", z_results_dict_path)
        else:
            z_results_dict = compute_z_results_dict(
                model, tasks, n_shot, n_example, device
            )
            torch.save(z_results_dict, z_results_dict_path)
            print(
                "No cached z_results_dict found; recomputed and saved",
                z_results_dict_path,
            )
    else:
        z_results_dict = compute_z_results_dict(model, tasks, n_shot, n_example, device)
    return z_results_dict


def compute_mean_z_result(z_results_dict):
    """h̄ — average of per-task z-results, used by the mean-ablation FV."""
    return torch.mean(torch.stack(list(z_results_dict.values()), dim=0), dim=0)


def compute_FVs_dict(z_results_dict, M, device="cuda", mean_matrix=None):
    """FV_k = Σ_{l,h} M[l,h] · z_k[l,h]  (+ Σ mean_matrix · (1−M) · h̄ if mean_matrix is given)."""
    FVs_dict = {}
    if mean_matrix is not None:
        mean_z_result = compute_mean_z_result(z_results_dict).to(device)
    for task_name, z_result in z_results_dict.items():
        M_expand = M.unsqueeze(-1).to(device)
        FV = (M_expand * z_result.to(device)).sum(dim=[0, 1])
        if mean_matrix is not None:
            FV = FV + (
                (mean_matrix * (torch.ones_like(M) - M)).unsqueeze(-1).to(device)
                * mean_z_result
            ).sum(dim=[0, 1])
        FVs_dict[task_name] = FV
    return FVs_dict
