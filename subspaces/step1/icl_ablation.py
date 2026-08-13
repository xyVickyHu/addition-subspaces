"""Head-output mean-ablation during the CLEAN n-shot forward pass.

The Step-1 headset evaluation intervenes on ZERO-SHOT prompts by adding an
FV to the residual stream; this module implements the complementary
NECESSITY direction of paper §3.2 ("Validating necessity of main heads in
the five-shot setting"): run the ordinary n-shot ICL prompt and REPLACE the
``attn.hook_result`` output of selected heads at the final prompt token with
a fixed per-head vector (canonically the cross-task mean ``h_bar`` from a
train z cache, the same basis the z caches record — see
``subspaces.step1.zcache.HOOK_SITE``).

The generation/scoring numerics deliberately mirror
``subspaces.utils.intervene.intervened_generation_with_accuracy`` (left-padded
prompts, greedy decode of exactly the target token count re-running the full
sequence each step, pad-masked exact token match); only the intervention
differs. During decode steps the replacement stays pinned to the ORIGINAL
final prompt token, matching the legacy convention for its residual-stream
intervention.

Replacement requires ``use_attn_result=True`` (TransformerLens only exposes
per-head outputs on that path). Because the flag changes the bf16 summation
order of the attention output even with no hook attached, callers comparing
against a plain clean run should also measure the empty-``head_values`` NULL
CONTROL this module supports.
"""

from __future__ import annotations

from subspaces.artifacts import ArtifactError

IMPL = {"module": "subspaces.step1.icl_ablation", "algorithm_version": 1}
HOOK_SITE = "attn.hook_result[last_prompt_token]"


def result_hook_name(layer: int) -> str:
    return f"blocks.{int(layer)}.attn.hook_result"


def head_ablation_hooks(head_values: dict, position: int) -> list:
    """Build ``(hook_name, fn)`` pairs replacing each selected head's output.

    ``head_values`` maps ``(layer, head) -> tensor[d_model]``; ``position``
    is the (negative) sequence index to replace. Heads sharing a layer share
    one hook. The hook mutates ONLY the listed head rows at ``position``;
    every other head/position passes through untouched.
    """
    by_layer: dict[int, list] = {}
    for (layer, head), value in sorted(head_values.items()):
        by_layer.setdefault(int(layer), []).append((int(head), value))

    hooks = []
    for layer, entries in sorted(by_layer.items()):

        def replace_heads(activation, hook, _entries=entries, _position=position):
            # activation: [batch, seq, n_heads, d_model]
            for head_idx, value in _entries:
                activation[:, _position, head_idx, :] = value
            return activation

        hooks.append((result_hook_name(layer), replace_heads))
    return hooks


def ablated_generation_correctness(
    model,
    prompts: list,
    targets: list,
    head_values: dict,
    *,
    batch_size: int,
) -> list:
    """Per-example 0/1 correctness with selected head outputs replaced.

    Every listed prompt is evaluated (final partial batch included). An empty
    ``head_values`` still runs the ``use_attn_result=True`` code path with no
    replacement — the null control for the flag's bf16 summation-order
    change.
    """
    if len(prompts) != len(targets):
        raise ValueError("prompts/targets length mismatch")
    correct: list = []
    for start in range(0, len(prompts), batch_size):
        correct.extend(
            _ablated_batch_correctness(
                model,
                prompts[start : start + batch_size],
                targets[start : start + batch_size],
                head_values,
            )
        )
    if len(correct) != len(prompts):
        raise ArtifactError(
            f"evaluated {len(correct)} of {len(prompts)} prompts — an example "
            "was dropped; refusing."
        )
    return correct


def _ablated_batch_correctness(model, prompts, targets, head_values) -> list:
    import torch  # heavy

    device_values = {
        key: value.to(device=model.cfg.device, dtype=model.cfg.dtype)
        for key, value in head_values.items()
    }
    with torch.no_grad():
        input_tokens = model.to_tokens(prompts, prepend_bos=True, padding_side="left")
        target_tokens = model.to_tokens(
            targets, prepend_bos=False, padding_side="right"
        )
        max_new_tokens = int(target_tokens.shape[1])
        all_tokens = input_tokens
        all_mask = input_tokens != model.tokenizer.pad_token_id

        previous_flag = model.cfg.use_attn_result
        model.cfg.use_attn_result = True
        try:
            next_token = None
            for step in range(max_new_tokens):
                if next_token is not None:
                    all_tokens = torch.cat([all_tokens, next_token.unsqueeze(1)], dim=1)
                    all_mask = torch.cat(
                        [all_mask, torch.ones_like(all_mask[:, :1])], dim=1
                    )
                # the original final prompt token sits `step` appended tokens
                # from the end (legacy position convention)
                hooks = head_ablation_hooks(device_values, position=-(step + 1))
                next_token_logits = model.run_with_hooks(
                    all_tokens,
                    fwd_hooks=hooks,
                    attention_mask=all_mask,
                    return_type="logits",
                )[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1)
                del next_token_logits
            all_tokens = torch.cat([all_tokens, next_token.unsqueeze(1)], dim=1)
        finally:
            model.cfg.use_attn_result = previous_flag

        generated = all_tokens[:, -max_new_tokens:]
        masks = target_tokens != model.tokenizer.pad_token_id
        masked_generated = generated * masks
        masked_targets = target_tokens * masks
        return [
            int(torch.equal(masked_generated[i], masked_targets[i]))
            for i in range(len(targets))
        ]


def sample_head_subsets(pool: list, n_sets: int, set_size: int, seed: int) -> list:
    """``n_sets`` DISTINCT size-``set_size`` subsets of ``pool``, deterministic
    in ``seed``; each subset is returned sorted. Refuses impossible requests
    instead of looping forever."""
    import math
    import random

    unique_pool = sorted(set(pool))
    if len(unique_pool) != len(pool):
        raise ValueError("pool contains duplicate heads")
    if set_size > len(unique_pool):
        raise ValueError(
            f"cannot draw size-{set_size} subsets from {len(unique_pool)} heads"
        )
    total = math.comb(len(unique_pool), set_size)
    if n_sets > total:
        raise ValueError(f"requested {n_sets} distinct subsets but only {total} exist")
    rng = random.Random(seed)
    seen: set = set()
    subsets: list = []
    while len(subsets) < n_sets:
        candidate = tuple(sorted(rng.sample(unique_pool, set_size)))
        if candidate in seen:
            continue
        seen.add(candidate)
        subsets.append(list(candidate))
    return subsets
