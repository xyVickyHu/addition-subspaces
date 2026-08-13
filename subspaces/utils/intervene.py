"""Intervention-time generation: NLL loss, greedy-accuracy, multi-task driver."""

from __future__ import annotations

from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from .data import LimitedTaskDataset, process_batch_data_individual
from .pca import compute_project_vectors


def compute_nll(logits, target_tokens, pad_token_id):
    """Mean NLL over target tokens (pad-masked)."""
    max_new_tokens = len(target_tokens[0])
    log_probs = torch.log_softmax(logits[:, -max_new_tokens - 1 : -1, :], dim=-1)
    mask = target_tokens != pad_token_id
    nll0 = torch.gather(log_probs, 2, target_tokens.unsqueeze(2)).squeeze(2)
    nll0[~mask] = 0
    nll0 = nll0.sum(1).mean()
    return -nll0


def intervened_generation_with_nll(
    model: HookedTransformer,
    input_text,
    target_text,
    layer_name: str = None,
    intervention_vector: torch.Tensor = None,
    intervention_mode: int = 1,
) -> float:
    """Joint-tokenize input+target, optionally intervene at the last input token, return NLL."""
    model.eval()
    if isinstance(input_text, str):
        input_text = [input_text]
        target_text = [target_text]
    pad_id = model.tokenizer.pad_token_id
    per_item_input = []
    per_item_target = []
    for inp, tgt in zip(input_text, target_text):
        in_t = model.to_tokens(inp, prepend_bos=True)[0]
        full_t = model.to_tokens(inp + tgt, prepend_bos=True)[0]
        per_item_input.append(in_t)
        per_item_target.append(full_t[in_t.shape[0] :])
    max_input_len = max(int(t.shape[0]) for t in per_item_input)
    max_target_len = max(int(t.shape[0]) for t in per_item_target)
    device = per_item_input[0].device
    dtype = per_item_input[0].dtype
    input_tokens = torch.stack(
        [
            torch.cat(
                [
                    torch.full(
                        (max_input_len - int(t.shape[0]),),
                        pad_id,
                        dtype=dtype,
                        device=device,
                    ),
                    t,
                ]
            )
            for t in per_item_input
        ]
    )
    target_tokens = torch.stack(
        [
            torch.cat(
                [
                    t,
                    torch.full(
                        (max_target_len - int(t.shape[0]),),
                        pad_id,
                        dtype=dtype,
                        device=device,
                    ),
                ]
            )
            for t in per_item_target
        ]
    )
    full_tokens = torch.cat([input_tokens, target_tokens], dim=1)
    full_mask = full_tokens != pad_id
    target_length = target_tokens.size(1)
    if layer_name and intervention_vector is not None:

        def intervention_hook(activation, hook, intevrention_mode):
            if intervention_mode == 1:
                activation[:, -target_length - 1, :] = intervention_vector
            else:
                activation[:, -target_length - 1, :] += intervention_vector
            return activation

        tmp_hook_fn = partial(intervention_hook, intevrention_mode=intervention_mode)
        final_logits = model.run_with_hooks(
            full_tokens,
            fwd_hooks=[(layer_name, tmp_hook_fn)],
            attention_mask=full_mask,
            return_type="logits",
        )
    else:
        with torch.no_grad():
            final_logits = model(full_tokens, attention_mask=full_mask)
    nll = compute_nll(final_logits, target_tokens, model.tokenizer.pad_token_id)
    del final_logits
    return nll


def intervened_generation_with_accuracy(
    model: HookedTransformer,
    input_text,
    target_text,
    layer_name: Optional[str] = None,
    intervention_vector: Optional[torch.Tensor] = None,
    intervention_mode: int = 1,
    print_text_num: int = 0,
    print_correct_list: bool = False,
) -> Union[float, Tuple[float, List[str], List[str]], List[bool]]:
    """Greedy-decode target tokens, optionally intervening at the prompt-tail residual stream."""
    model.eval()
    with torch.no_grad():
        input_tokens = model.to_tokens(
            input_text, prepend_bos=True, padding_side="left"
        )
        target_tokens = model.to_tokens(
            target_text, prepend_bos=False, padding_side="right"
        )
        input_mask = input_tokens != model.tokenizer.pad_token_id
        max_new_tokens = len(target_tokens[0])

        if layer_name and intervention_vector is not None:

            def intervention_hook(
                activation: torch.Tensor,
                hook: HookPoint,
                position: int,
                intervention_mode: int,
            ) -> torch.Tensor:
                if intervention_mode == 1:
                    activation[:, position, :] = intervention_vector
                else:
                    activation[:, position, :] += intervention_vector
                return activation

            temp_hook_fn = partial(
                intervention_hook, position=-1, intervention_mode=intervention_mode
            )
            input_logits = model.run_with_hooks(
                input_tokens,
                fwd_hooks=[(layer_name, temp_hook_fn)],
                attention_mask=input_mask,
                return_type="logits",
            )
            next_token_logits = input_logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1)
            all_tokens = input_tokens.clone()
            all_mask = input_mask.clone()
            del input_logits, input_mask, input_tokens
            torch.cuda.empty_cache()
            for pos in range(2, max_new_tokens + 2):
                all_tokens = torch.cat([all_tokens, next_token.unsqueeze(1)], dim=1)
                all_mask = torch.cat(
                    [all_mask, torch.ones_like(all_mask[:, :1])], dim=1
                )
                temp_hook_fn = partial(
                    intervention_hook,
                    position=-pos,
                    intervention_mode=intervention_mode,
                )
                next_token_logits = model.run_with_hooks(
                    all_tokens,
                    fwd_hooks=[(layer_name, temp_hook_fn)],
                    attention_mask=all_mask,
                    return_type="logits",
                )[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1)
                del next_token_logits
                torch.cuda.empty_cache()
        else:
            input_logits = model(
                input_tokens, attention_mask=input_mask, return_type="logits"
            )
            next_token_logits = input_logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1)
            all_tokens = input_tokens.clone()
            all_mask = input_mask.clone()
            del input_logits, input_mask, next_token_logits
            torch.cuda.empty_cache()

            for _ in range(max_new_tokens):
                all_mask = torch.cat(
                    [all_mask, torch.ones_like(all_mask[:, :1])], dim=1
                )
                all_tokens = torch.cat([all_tokens, next_token.unsqueeze(1)], dim=1)
                torch.cuda.empty_cache()
                next_token_logits = model(all_tokens, attention_mask=all_mask)[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1)
                del next_token_logits
                torch.cuda.empty_cache()

        correct = 0
        correct_list = []
        newly_generated_tokens = all_tokens[:, -max_new_tokens:]
        masks = target_tokens != model.tokenizer.pad_token_id
        masked_generated_tokens = newly_generated_tokens * masks
        masked_target_tokens = target_tokens * masks

        for i in range(len(target_tokens)):
            correct += torch.equal(masked_generated_tokens[i], masked_target_tokens[i])
            correct_list.append(
                torch.equal(masked_generated_tokens[i], masked_target_tokens[i])
            )
        accuracy = correct / len(target_tokens)
        del all_tokens
        torch.cuda.empty_cache()
        if print_text_num:
            generated_strings = model.to_string(
                masked_generated_tokens[:print_text_num, :]
            )
            target_stings = model.to_string(masked_target_tokens[:print_text_num, :])
            print("accuracy", accuracy)
            print(
                "generated",
                masked_generated_tokens[:print_text_num, :].shape,
                generated_strings,
            )
            print(
                "target", masked_target_tokens[:print_text_num, :].shape, target_stings
            )
            return accuracy, generated_strings, target_stings
        if print_correct_list:
            return (
                accuracy,
                model.to_string(masked_generated_tokens[:, :]),
                correct_list,
            )
        return accuracy


def compute_task_accuracy(
    test_name,
    tasks,
    n_shot,
    test_limit,
    bs,
    FVs_dict,
    trans_FVs_dict,
    subspace_components,
    mean_train_FVs,
    model,
    layer_name,
    intervene=True,
    clean=True,
    corrupted=True,
    print_intervened_num=0,
    datapoint_filter=None,
):
    """Evaluate clean / corrupted / intervened accuracy on a single task, optionally projecting FVs."""
    dataset = LimitedTaskDataset(
        {test_name: tasks[test_name]},
        n_shot,
        test_limit,
        datapoint_filter=datapoint_filter,
    )
    test_loader = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=True)

    test_batch_cnt = 0
    batch_clean_accuracy = 0
    batch_corrupted_accuracy = 0
    batch_intervened_accuracy = 0
    batch_proj_accuracy = 0
    batch_trans_accuracy = 0
    all_intervened_generated_strings = []
    all_intervened_target_strings = []
    all_corrupted_generated_strings = []
    all_corrupted_target_strings = []
    all_clean_generated_strings = []
    all_clean_target_strings = []
    for test_batch in test_loader:
        (
            test_batch_FV,
            test_batch_prompt,
            test_batch_zero_shot_prompt,
            test_batch_target,
        ) = process_batch_data_individual(test_batch, tasks, FVs_dict)
        if clean:
            if print_intervened_num > 0:
                clean_accuracy, clean_generated_strings, clean_target_strings = (
                    intervened_generation_with_accuracy(
                        model,
                        test_batch_prompt,
                        test_batch_target,
                        print_text_num=print_intervened_num,
                    )
                )
                all_clean_generated_strings.extend(clean_generated_strings)
                all_clean_target_strings.extend(clean_target_strings)
            else:
                clean_accuracy = intervened_generation_with_accuracy(
                    model, test_batch_prompt, test_batch_target
                )
            batch_clean_accuracy += clean_accuracy
        if corrupted:
            if print_intervened_num > 0:
                (
                    corrupted_accuracy,
                    corrupted_generated_strings,
                    corrupted_target_strings,
                ) = intervened_generation_with_accuracy(
                    model,
                    test_batch_zero_shot_prompt,
                    test_batch_target,
                    print_text_num=print_intervened_num,
                )
                all_corrupted_generated_strings.extend(corrupted_generated_strings)
                all_corrupted_target_strings.extend(corrupted_target_strings)
            else:
                corrupted_accuracy = intervened_generation_with_accuracy(
                    model, test_batch_zero_shot_prompt, test_batch_target
                )
            batch_corrupted_accuracy += corrupted_accuracy
        if intervene:
            if print_intervened_num > 0:
                (
                    intervened_accuracy,
                    intervened_generated_strings,
                    intervened_target_strings,
                ) = intervened_generation_with_accuracy(
                    model,
                    test_batch_zero_shot_prompt,
                    test_batch_target,
                    layer_name,
                    test_batch_FV,
                    intervention_mode=0,
                    print_text_num=print_intervened_num,
                )
                all_intervened_generated_strings.extend(intervened_generated_strings)
                all_intervened_target_strings.extend(intervened_target_strings)
            else:
                intervened_accuracy = intervened_generation_with_accuracy(
                    model,
                    test_batch_zero_shot_prompt,
                    test_batch_target,
                    layer_name,
                    test_batch_FV,
                    intervention_mode=0,
                )
            batch_intervened_accuracy += intervened_accuracy

        if subspace_components is not None and mean_train_FVs is not None:
            print("computing proj_accuracy")
            test_batch_FV_proj = compute_project_vectors(
                test_batch_FV, subspace_components, mean_train_FVs
            )
            proj_accuracy = intervened_generation_with_accuracy(
                model,
                test_batch_zero_shot_prompt,
                test_batch_target,
                layer_name,
                test_batch_FV_proj,
                intervention_mode=0,
            )
            batch_proj_accuracy += proj_accuracy
        if trans_FVs_dict is not None:
            print("computing trans_accuracy")
            (
                test_batch_FV_trans,
                test_batch_prompt,
                test_batch_zero_shot_prompt,
                test_batch_target,
            ) = process_batch_data_individual(test_batch, tasks, trans_FVs_dict)
            trans_accuracy = intervened_generation_with_accuracy(
                model,
                test_batch_zero_shot_prompt,
                test_batch_target,
                layer_name,
                test_batch_FV_trans,
                intervention_mode=0,
            )
            batch_trans_accuracy += trans_accuracy

        test_batch_cnt += 1
    if test_batch_cnt == 0:
        # No full batch was produced (e.g. a datapoint_filter left this task with
        # fewer examples than the batch size). Report NaN rather than crashing;
        # callers that aggregate across tasks should skip NaNs.
        nan = float("nan")
        return (
            nan,
            nan,
            nan,
            nan,
            nan,
            all_clean_generated_strings,
            all_clean_target_strings,
            all_corrupted_generated_strings,
            all_corrupted_target_strings,
            all_intervened_target_strings,
            all_intervened_generated_strings,
        )
    batch_clean_accuracy /= test_batch_cnt
    batch_corrupted_accuracy /= test_batch_cnt
    batch_intervened_accuracy /= test_batch_cnt
    batch_proj_accuracy /= test_batch_cnt
    batch_trans_accuracy /= test_batch_cnt
    return (
        batch_clean_accuracy,
        batch_corrupted_accuracy,
        batch_intervened_accuracy,
        batch_proj_accuracy,
        batch_trans_accuracy,
        all_clean_generated_strings,
        all_clean_target_strings,
        all_corrupted_generated_strings,
        all_corrupted_target_strings,
        all_intervened_target_strings,
        all_intervened_generated_strings,
    )


def eval_fv_per_example_correctness(
    task_name: str,
    tasks: Dict,
    n_shot: int,
    test_limit: int,
    bs: int,
    FVs_dict: Dict[str, torch.Tensor],
    model: HookedTransformer,
    layer_name: str,
) -> List[int]:
    """Intervened greedy-decode eval on a single task — returns per-example 0/1 correctness.

    Counterpart of ``compute_task_accuracy`` that only does the intervened
    branch and exposes the per-prompt correctness vector (needed for paired
    bootstrap CIs over example-level outcomes). Uses ``drop_last=True`` to
    match the rest of the pipeline's batch handling.
    """
    dataset = LimitedTaskDataset({task_name: tasks[task_name]}, n_shot, test_limit)
    test_loader = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=True)
    correct_list_all: List[int] = []
    for test_batch in test_loader:
        test_batch_FV, _, test_batch_zero_shot_prompt, test_batch_target = (
            process_batch_data_individual(test_batch, tasks, FVs_dict)
        )
        _acc, _genstr, correct_list = intervened_generation_with_accuracy(
            model,
            test_batch_zero_shot_prompt,
            test_batch_target,
            layer_name,
            test_batch_FV,
            intervention_mode=0,
            print_correct_list=True,
        )
        correct_list_all.extend(int(bool(b)) for b in correct_list)
    return correct_list_all
