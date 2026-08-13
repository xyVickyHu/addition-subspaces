"""``subspaces.utils.signals`` — paper §5 "Signal Extractors of ICL Demonstrations".

Per-demonstration *extracted-signal* decomposition behind §5.1/5.2, ported into
the canonical package (a clean re-derivation of the original exploratory
notebook). Reuses the package loaders
(:func:`subspaces.utils.model.load_model_no_grad`, :func:`subspaces.utils.io.load_task_data`,
:func:`subspaces.utils.activations.load_z_results_dict`); fully self-contained (no
imports from legacy research-tree code).

Definitions, for a head ``h`` at layer ``l`` and final-token position:

    h(p) = Σ_t α_t · O_h V_h z_t            (head output at the final token)

For each demonstration label token ``y_i`` the per-demo **extracted signal** is
``s_i = O_h V_h z_{y_i}`` (α-FREE) and ``α_i`` its final-token attention weight.

§5.1/5.2 — *label-token peaking*: the attention budget ``{α_t}`` concentrates on
the demo label tokens, and ``s_i`` aligns (in the head's 6-D PCA subspace) with
the add-k task direction ``h_k = z_results_dict["number-add{k}"][l,h]``
(:func:`signal_cosine`). The *label-token aggregation* analysis
(:func:`label_relationship`) then asks, WITHIN the labels, whether attention
prefers the labels whose extracted signal is better aligned with ``h_k``, and
whether the attention-weighted average over labels aligns with ``h_k`` better
than a uniform average.

Numbers 1..99 are single tokens in Llama-3, so each label ``y_i`` and the query
are single positions. All projections are float32.
"""

from __future__ import annotations

import random
from functools import partial
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

MAIN_HEADS: List[Tuple[int, int]] = [(15, 2), (15, 1), (13, 6)]
K_RANGE = range(1, 31)


# --------------------------------------------------------------------------- #
# add-k prompt construction (the "x->y#...#xq->" format the cache/paper use)   #
# --------------------------------------------------------------------------- #


# Task families: a label function y = f(x, k). add covers add-k (k>0) AND
# subtraction (k<0, dataset keys ``number-add{-k}``); mul covers ``number-mul{k}``.
def _add_label(x: int, k: int) -> int:
    return x + k


def _mul_label(x: int, k: int) -> int:
    return x * k


LABEL_FNS = {"add": _add_label, "mul": _mul_label}


def valid_pool(
    k: int, label_fn=_add_label, lo_in: int = 1, hi_in: int = 99
) -> List[int]:
    """Inputs x in [lo_in,hi_in] whose label y=label_fn(x,k) is also in [1,99]
    (so both x and y are single tokens on Llama-3 / phi-4)."""
    return [x for x in range(lo_in, hi_in + 1) if 1 <= label_fn(x, k) <= 99]


def addk_pool(k: int, lo_in: int = 1, hi_in: int = 99) -> List[int]:
    """Back-compat add-k pool (= ``valid_pool(k, _add_label)``)."""
    return valid_pool(k, _add_label, lo_in, hi_in)


def build_task_prompt(
    rng: random.Random,
    k: int,
    n_shot: int,
    pool: List[int],
    label_fn=_add_label,
) -> Tuple[str, List[int], int]:
    """Sample one clean n-shot prompt 'x1->y1#...#xn->yn#xq->' for task y=f(x,k).

    Returns (prompt, [x1..xn], x_q). All inputs distinct.
    """
    if len(pool) < n_shot + 1:
        raise ValueError(f"pool too small for k={k}: |pool|={len(pool)}")
    picks = rng.sample(pool, n_shot + 1)
    xs, x_q = picks[:n_shot], picks[n_shot]
    demos = "#".join(f"{x}->{label_fn(x, k)}" for x in xs)
    return f"{demos}#{x_q}->", xs, x_q


def build_addk_prompt(
    rng: random.Random,
    k: int,
    n_shot: int = 5,
    pool: List[int] = None,
) -> Tuple[str, List[int], int]:
    """Back-compat add-k prompt builder."""
    if pool is None:
        pool = addk_pool(k)
    return build_task_prompt(rng, k, n_shot, pool, _add_label)


def find_label_positions_and_final(model, prompt: str) -> Tuple[List[int], int, int]:
    """Token positions of each demo label y_i (token just before each '#') and
    the final token position. Returns (label_positions, final_pos, seq_len)."""
    tokens = model.to_tokens(prompt, prepend_bos=True, padding_side="left")[0]
    hash_id = model.to_tokens("#", prepend_bos=False)[0, -1].item()
    seq = tokens.tolist()
    label_positions = [i - 1 for i, tid in enumerate(seq) if tid == hash_id and i > 0]
    return label_positions, len(seq) - 1, len(seq)


def token_types(model, prompt: str) -> List[str]:
    """Tag each token position by role for the label-peaking attention profile:
    {bos, input, arrow, label, hash, final, other}. Used to bucket the
    final-token attention mass. The 'final' position is the trailing '->' query
    cue; the query input itself is tagged 'input'."""
    tokens = model.to_tokens(prompt, prepend_bos=True, padding_side="left")[0].tolist()
    hash_id = model.to_tokens("#", prepend_bos=False)[0, -1].item()
    arrow_ids = set(model.to_tokens("->", prepend_bos=False)[0].tolist())
    n = len(tokens)
    types = ["other"] * n
    types[0] = "bos"
    for i, tid in enumerate(tokens):
        if i == 0:
            continue
        if i == n - 1:
            types[i] = "final"
        elif tid == hash_id:
            types[i] = "hash"
        elif tid in arrow_ids:
            types[i] = "arrow"
        elif i + 1 < n and tokens[i + 1] == hash_id:
            types[i] = "label"  # token just before a '#'
        elif i + 1 < n and tokens[i + 1] in arrow_ids:
            types[i] = "input"  # token just before an '->'
    return types


# --------------------------------------------------------------------------- #
# task directions + per-head PCA subspace (from the cached z_results / h_k)    #
# --------------------------------------------------------------------------- #


def get_hk_directions(
    z_results_dict: Dict[str, torch.Tensor],
    k: int,
    heads: Sequence[Tuple[int, int]],
    device,
    k_range=K_RANGE,
    task_prefix: str = "number-add",
) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
    """Per-head task-k directions: h_k (raw), unit_raw, unit_contrast
    (k-specific = h_k − mean_{k'} h_{k'}, normalized). ``task_prefix`` selects the
    parametric family (``number-add`` for add/sub, ``number-mul`` for mul)."""
    task_names = [
        f"{task_prefix}{kp}" for kp in k_range if f"{task_prefix}{kp}" in z_results_dict
    ]
    if f"{task_prefix}{k}" not in z_results_dict:
        raise KeyError(f"{task_prefix}{k} not in z_results_dict")
    out: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
    for l, h in heads:
        h_all = (
            torch.stack([z_results_dict[t][l, h] for t in task_names])
            .to(device)
            .to(torch.float32)
        )
        h_k = z_results_dict[f"{task_prefix}{k}"][l, h].to(device).to(torch.float32)
        contrast = h_k - h_all.mean(dim=0)
        out[(l, h)] = {
            "h_k": h_k,
            "unit_raw": h_k / (h_k.norm() + 1e-12),
            "unit_contrast": contrast / (contrast.norm() + 1e-12),
            "h_k_norm": float(h_k.norm().item()),
        }
    return out


def head_pca_subspace(
    z_results_dict,
    l: int,
    h: int,
    n_comp: int = 6,
    k_range=K_RANGE,
    device="cpu",
    task_prefix: str = "number-add",
) -> torch.Tensor:
    """Top-``n_comp`` PCA subspace of the task-k vectors for head (l, h)
    (centered SVD principal axes — the paper §4 ``heads_components``).
    Returns (d_model, n_comp) float32."""
    task_names = [
        f"{task_prefix}{kp}" for kp in k_range if f"{task_prefix}{kp}" in z_results_dict
    ]
    X = (
        torch.stack([z_results_dict[t][l, h] for t in task_names])
        .to(torch.float32)
        .cpu()
        .numpy()
    )
    Xc = X - X.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    return torch.tensor(vt[:n_comp].T, dtype=torch.float32, device=device)


# --------------------------------------------------------------------------- #
# forward pass + per-demo extracted info                                      #
# --------------------------------------------------------------------------- #


def value_head_for_query_head(model, query_head: int, n_value_heads: int) -> int:
    """Map a query head index to its GQA/MQA value (KV) head index."""
    n_query_heads = int(model.cfg.n_heads)  # avoid model.W_O (restacks all layers)
    if n_value_heads == n_query_heads:
        return query_head
    if n_query_heads % n_value_heads != 0:
        raise ValueError(
            f"{n_query_heads} query heads not divisible by {n_value_heads} kv heads"
        )
    return query_head // (n_query_heads // n_value_heads)


def forward_capture(model, prompt: str, layers: Sequence[int], device):
    """One forward pass capturing the attention pattern + value vectors at the
    given layers. Returns (cache, label_positions, final_pos, seq_len)."""
    tokens = model.to_tokens(prompt, prepend_bos=True, padding_side="left").to(device)
    label_positions, final_pos, seq_len = find_label_positions_and_final(model, prompt)
    cache: Dict[Tuple[int, str], torch.Tensor] = {}

    def grab(activation, hook, key):
        cache[key] = activation.detach()

    hooks = []
    for l in layers:
        hooks.append(
            (f"blocks.{l}.attn.hook_pattern", partial(grab, key=(l, "pattern")))
        )
        hooks.append((f"blocks.{l}.attn.hook_v", partial(grab, key=(l, "v"))))
    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=hooks)
    return cache, label_positions, final_pos, seq_len


def extracted_info(
    cache, model, l: int, h: int, label_positions: List[int], final_pos: int
) -> Dict[str, object]:
    """Per-demo extracted info for head (l, h), direction-agnostic.

    Returns:
        alpha     : (n_shot,) — final-token attention to each demo label
        alpha_full: (seq,)    — final-token attention to every position
        s         : (n_shot, d_model) float32 — O_h V_h z_{y_i}
        s_norm    : (n_shot,) — ‖s_i‖
    """
    pat = cache[(l, "pattern")][0, h]  # (q, k)
    v = cache[(l, "v")][0]  # (seq, n_kv_heads, d_head)
    alpha_full = pat[final_pos].to(torch.float32)
    alphas = alpha_full[label_positions]
    vh = value_head_for_query_head(model, h, int(v.shape[1]))
    v_lab = v[label_positions, vh, :].to(torch.float32)  # (n_shot, d_head)
    W_O = (
        model.blocks[l].attn.W_O[h].to(torch.float32)
    )  # (d_head, d_model); per-block (no full stack)
    s = v_lab @ W_O  # (n_shot, d_model)
    return {
        "alpha": alphas.cpu().numpy().astype(np.float64),
        "alpha_full": alpha_full.cpu().numpy().astype(np.float64),
        "s": s,
        "s_norm": s.norm(dim=-1).cpu().numpy().astype(np.float64),
    }


# --------------------------------------------------------------------------- #
# §5.2 per-demo signal alignment (α-free subspace cosine)                      #
# --------------------------------------------------------------------------- #


def _subspace_proj_and_taskdir(
    s: torch.Tensor, subspace: torch.Tensor, hk: torch.Tensor
):
    """sc = proj(s) (n_shot, n_comp); hc_unit = unit(proj(h_k)) (n_comp,)."""
    sc = s.to(torch.float32) @ subspace
    hc = hk.to(torch.float32) @ subspace
    return sc, hc / (hc.norm() + 1e-12)


def signal_cosine(
    s: torch.Tensor, subspace: torch.Tensor, hk: torch.Tensor
) -> np.ndarray:
    """α-free subspace cosine: ⟨proj(s_i), unit(proj(h_k))⟩ / ‖proj(s_i)‖.
    The attention scalar cancels. (n_shot,)."""
    sc, hc_unit = _subspace_proj_and_taskdir(s, subspace, hk)
    cos = (sc @ hc_unit) / (sc.norm(dim=-1) + 1e-12)
    return cos.cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------- #
# attention × activation relationship — LABEL TOKENS ONLY                       #
#   Refactor of the follow-up aggregation study's mode-3 "relationship" analysis #
#   (``agg_hypothesis/pa_aggregate.extract_prompt_scalars`` /                    #
#   ``agg_observe``), which ran over ALL token positions. Here it is restricted #
#   to the demo label tokens: rather than "does attention route onto the label  #
#   tokens" (that is the label-peaking ratio in collect_signals), this asks,    #
#   WITHIN the labels, whether attention prefers the labels whose extracted      #
#   signal is better aligned with h_k, and whether the attention-weighted        #
#   average over labels aligns with h_k better than a uniform average.          #
# --------------------------------------------------------------------------- #


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(u @ v / (nu * nv))


def label_relationship(
    s: torch.Tensor,
    alpha: np.ndarray,
    hk_unit: torch.Tensor,
    hk_raw: torch.Tensor,
    n_shuffle: int = 30,
    seed: int = 0,
) -> Dict[str, float]:
    """Attention × activation relationship over the demo LABEL tokens, for one
    prompt at one head. Refactored (label-only) from the follow-up
    aggregation study's ``pa_aggregate.extract_prompt_scalars``.

    s        : (n_shot, d_model) per-label extracted signals O_h V_h z_{y_i}
    alpha    : (n_shot,) final-token attention to each label
    hk_unit  : h_k / ‖h_k‖ ; hk_raw : h_k     (full-space task direction)

    Returns per-prompt scalars:
        ondemo_cos_mean : mean full-space cos(s_i, h_k) over labels — the §5.1/5.2
                          "label-token output aligns with h_k" quantity
        routing_corr     : Pearson(α_i, ⟨s_i, hk_unit⟩) — does attention prefer
                           labels with larger on-task projection?
        routing_corr_cos : Pearson(α_i, cos(s_i, h_k)) — …with larger alignment?
        cos_weighted     : cos(Σ_i α̃_i s_i, h_k)  (α̃ = α renormalized over labels)
        cos_uniform      : cos(mean_i s_i, h_k)
        wgain            : cos_weighted − cos_uniform — does within-label
                           attention-weighting improve aggregate alignment?
        wgain_shuf       : mean over α-permutations of (cos_weighted_shuf − cos_uniform)
    """
    s32 = s.to(torch.float32)
    proj = (s32 @ hk_unit.to(torch.float32)).cpu().numpy().astype(np.float64)
    snorm = s32.norm(dim=-1).cpu().numpy().astype(np.float64)
    cos_i = proj / (snorm + 1e-12)
    a = np.asarray(alpha, dtype=np.float64)
    a_norm = a / (a.sum() + 1e-12)
    s_np = s32.cpu().numpy()
    hk_np = hk_raw.to(torch.float32).cpu().numpy()
    cos_w = _cos((a_norm[:, None] * s_np).sum(axis=0), hk_np)
    cos_u = _cos(s_np.mean(axis=0), hk_np)
    rng = np.random.default_rng(seed)
    gains = [
        (
            _cos(
                (a_norm[rng.permutation(len(a_norm))][:, None] * s_np).sum(axis=0),
                hk_np,
            )
            - cos_u
        )
        for _ in range(n_shuffle)
    ]
    # routing null: permute α among the labels, recompute Pearson(α, cos_i). Breaks
    # the α↔alignment pairing while preserving both marginals → the noise band for
    # routing_corr_cos (mean over prompts should bracket 0 under independence).
    rcos_null = [_pearson(a[rng.permutation(a.size)], cos_i) for _ in range(n_shuffle)]
    return {
        "ondemo_cos_mean": float(np.mean(cos_i)),
        "routing_corr": _pearson(a, proj),
        "routing_corr_cos": _pearson(a, cos_i),
        "routing_corr_cos_shuf": (
            float(np.nanmean(rcos_null)) if rcos_null else float("nan")
        ),
        "cos_weighted": cos_w,
        "cos_uniform": cos_u,
        "wgain": cos_w - cos_u,
        "wgain_shuf": float(np.nanmean(gains)) if gains else float("nan"),
    }


# --------------------------------------------------------------------------- #
# bootstrap helpers                                                            #
# --------------------------------------------------------------------------- #


def bootstrap_mean_ci(values, n_boot: int = 2000, seed: int = 0):
    """Mean + 95% percentile CI by resampling a value list."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samp = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(samp, 0.025)),
        float(np.quantile(samp, 0.975)),
    )


# --------------------------------------------------------------------------- #
# end-to-end §5 driver (used by run_pipeline stage 4 and run_signal_extraction) #
# --------------------------------------------------------------------------- #


def collect_signals(
    model,
    z_results_dict,
    heads: Sequence[Tuple[int, int]],
    device,
    *,
    n_prompts_per_k: int = 40,
    n_shot: int = 5,
    k_range=K_RANGE,
    n_comp: int = 6,
    seed: int = 0,
    task_prefix: str = "number-add",
    label_fn=None,
) -> Dict[Tuple[int, int], Dict[str, object]]:
    """Run forward passes over a seed-pinned prompt sample and accumulate, per
    head: the §5.1/5.2 attention profile + signal alignment and the label-token
    attention×activation relationship.

    ``task_prefix`` + ``label_fn`` select the parametric family: ``number-add`` /
    add (k>0 = add-k, k<0 = subtraction) or ``number-mul`` / mul. Prompts whose
    label count ≠ n_shot (e.g. a tokenizer that splits the separator) are skipped
    and counted in ``acc[head]["n_skipped"]``.

    Returns a dict keyed by head with raw arrays; statistics are computed by
    :func:`summarize_signals`."""
    if label_fn is None:
        label_fn = _add_label
    layers = sorted({l for (l, _h) in heads})
    subspaces = {
        (l, h): head_pca_subspace(
            z_results_dict, l, h, n_comp, k_range, device, task_prefix
        )
        for (l, h) in heads
    }
    rel_keys = [
        "ondemo_cos_mean",
        "routing_corr",
        "routing_corr_cos",
        "routing_corr_cos_shuf",
        "cos_weighted",
        "cos_uniform",
        "wgain",
        "wgain_shuf",
    ]
    acc = {
        (l, h): {
            "align": [],
            "attn_lab": [],
            "attn_off": [],
            "postype": {},
            "postype_n": {},
            "task_mass": [],
            "sink_mass": [],
            "n_skipped": 0,
            "n_shot": n_shot,
            **{f"rel_{kk}": [] for kk in rel_keys},
        }
        for (l, h) in heads
    }
    rng = random.Random(seed)
    sink_types = ("bos", "hash", "arrow", "final")
    pc = 0
    for k in k_range:
        pool = valid_pool(k, label_fn)
        if len(pool) < n_shot + 1:
            continue
        hk_dirs = get_hk_directions(
            z_results_dict, k, heads, device, k_range, task_prefix
        )
        for _ in range(n_prompts_per_k):
            prompt, _xs, _xq = build_task_prompt(rng, k, n_shot, pool, label_fn)
            cache, label_pos, final_pos, _seq = forward_capture(
                model, prompt, layers, device
            )
            if (
                len(label_pos) != n_shot
            ):  # robustness guard for cross-model tokenization
                for l, h in heads:
                    acc[(l, h)]["n_skipped"] += 1
                continue
            ttypes = token_types(model, prompt)
            sink_pos = [p for p, t in enumerate(ttypes) if t in sink_types]
            for l, h in heads:
                info = extracted_info(cache, model, l, h, label_pos, final_pos)
                hk = hk_dirs[(l, h)]["h_k"]
                sub = subspaces[(l, h)]
                a = acc[(l, h)]
                # §5.2 α-free subspace alignment of the extracted signal
                a["align"].append(float(signal_cosine(info["s"], sub, hk).mean()))
                # §5.1/5.2 label-token peaking (attention mass)
                alpha_full = info["alpha_full"]
                a["attn_lab"].append(
                    float(alpha_full[label_pos].sum()) / len(label_pos)
                )
                off_mask = np.ones(alpha_full.shape[0], dtype=bool)
                off_mask[label_pos] = False
                a["attn_off"].append(float(alpha_full[off_mask].mean()))
                a["task_mass"].append(float(info["alpha"].sum()))
                a["sink_mass"].append(
                    float(alpha_full[sink_pos].sum()) if sink_pos else 0.0
                )
                for pos, t in enumerate(ttypes):
                    a["postype"][t] = a["postype"].get(t, 0.0) + float(alpha_full[pos])
                    a["postype_n"][t] = a["postype_n"].get(t, 0) + 1
                # mode-3 attention × activation relationship over label tokens
                rel = label_relationship(
                    info["s"],
                    info["alpha"],
                    hk_dirs[(l, h)]["unit_raw"],
                    hk,
                    seed=seed + pc,
                )
                for kk in rel_keys:
                    a[f"rel_{kk}"].append(rel[kk])
            pc += 1
    return acc


def _mean_ci(vals, n_boot, seed):
    """Bootstrap mean + 95% CI of a value list, dropping non-finite entries."""
    arr = np.asarray(
        [v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64
    )
    m, lo, hi = bootstrap_mean_ci(arr, n_boot=n_boot, seed=seed)
    return {"mean": m, "ci": [lo, hi], "n": int(arr.size)}


def collect_signals_discrete(
    model,
    z_results_dict,
    tasks_data: Dict[str, list],
    heads: Sequence[Tuple[int, int]],
    device,
    *,
    n_prompts_per_task: int = 40,
    n_shot: int = 5,
    seed: int = 0,
) -> Dict[Tuple[int, int], Dict[str, object]]:
    """Reduced §5 for DISCRETE (non-parametric) ICL tasks (antonym, capitalize, …).

    Each task's own function vector ``h_task = z_results[task][l,h]`` is the task
    direction; there is NO periodic family, so there is NO 6-D subspace — all
    readings are FULL-space projections onto ``h_task``. Prompts are real ICL
    prompts (``subspaces.utils.data.format_input`` → 'in1->out1#…#xq->') sampled from
    ``tasks_data``. Returns the same acc structure as :func:`collect_signals`, so
    :func:`summarize_signals` consumes it unchanged — but note ``mean_signal_cosine``
    is full-space here (not subspace). Tasks present in both ``tasks_data`` and
    ``z_results_dict`` are used."""
    from subspaces.utils.data import format_input

    task_names = [t for t in z_results_dict if t in tasks_data]
    layers = sorted({l for (l, _h) in heads})
    rel_keys = [
        "ondemo_cos_mean",
        "routing_corr",
        "routing_corr_cos",
        "routing_corr_cos_shuf",
        "cos_weighted",
        "cos_uniform",
        "wgain",
        "wgain_shuf",
    ]
    acc = {
        (l, h): {
            "align": [],
            "attn_lab": [],
            "attn_off": [],
            "postype": {},
            "postype_n": {},
            "task_mass": [],
            "sink_mass": [],
            "n_skipped": 0,
            "n_shot": n_shot,
            **{f"rel_{kk}": [] for kk in rel_keys},
        }
        for (l, h) in heads
    }
    rng = random.Random(seed)
    sink_types = ("bos", "hash", "arrow", "final")
    pc = 0
    for task_name in task_names:
        task = tasks_data[task_name]
        inputs = list(
            dict.fromkeys(ex["input"] for ex in task)
        )  # unique, order-preserving
        if len(inputs) < n_shot + 1:
            continue
        hk = {
            (l, h): z_results_dict[task_name][l, h].to(device).to(torch.float32)
            for (l, h) in heads
        }
        hk_unit = {kh: v / (v.norm() + 1e-12) for kh, v in hk.items()}
        for _ in range(n_prompts_per_task):
            picks = rng.sample(inputs, n_shot + 1)
            prompt = format_input(picks[:n_shot], picks[n_shot], task)
            cache, label_pos, final_pos, _seq = forward_capture(
                model, prompt, layers, device
            )
            if (
                len(label_pos) != n_shot
            ):  # multi-token outputs can desync label/separator counts
                for l, h in heads:
                    acc[(l, h)]["n_skipped"] += 1
                continue
            ttypes = token_types(model, prompt)
            sink_pos = [p for p, t in enumerate(ttypes) if t in sink_types]
            for l, h in heads:
                info = extracted_info(cache, model, l, h, label_pos, final_pos)
                s = info["s"]
                alpha = info["alpha"]
                hku, hkr = hk_unit[(l, h)], hk[(l, h)]
                proj = (s.to(torch.float32) @ hku).cpu().numpy().astype(np.float64)
                cos_i = proj / (info["s_norm"] + 1e-12)
                a = acc[(l, h)]
                a["align"].append(float(cos_i.mean()))  # full-space alignment
                alpha_full = info["alpha_full"]
                a["attn_lab"].append(
                    float(alpha_full[label_pos].sum()) / len(label_pos)
                )
                off_mask = np.ones(alpha_full.shape[0], dtype=bool)
                off_mask[label_pos] = False
                a["attn_off"].append(float(alpha_full[off_mask].mean()))
                a["task_mass"].append(float(info["alpha"].sum()))
                a["sink_mass"].append(
                    float(alpha_full[sink_pos].sum()) if sink_pos else 0.0
                )
                for pos, t in enumerate(ttypes):
                    a["postype"][t] = a["postype"].get(t, 0.0) + float(alpha_full[pos])
                    a["postype_n"][t] = a["postype_n"].get(t, 0) + 1
                rel = label_relationship(s, alpha, hku, hkr, seed=seed + pc)
                for kk in rel_keys:
                    a[f"rel_{kk}"].append(rel[kk])
            pc += 1
    return acc


def summarize_signals(
    acc: Dict[Tuple[int, int], Dict[str, object]], *, n_boot: int = 2000, seed: int = 0
) -> Dict[str, Dict]:
    """Turn raw accumulated signals into the §5.1/5.2 + relationship summary,
    per head."""
    out = {}
    for (l, h), a in acc.items():
        align_mean, align_lo, align_hi = bootstrap_mean_ci(
            a["align"], n_boot=n_boot, seed=seed
        )
        lab_mean = float(np.mean(a["attn_lab"]))
        off_mean = float(np.mean(a["attn_off"]))
        postype = {t: a["postype"][t] / a["postype_n"][t] for t in a["postype"]}
        out[f"({l},{h})"] = {
            "n_prompts": len(a["align"]),
            "n_skipped": int(a.get("n_skipped", 0)),
            "n_shot": int(a["n_shot"]),
            # §5.1/5.2 — label-token peaking + signal alignment
            "mean_attn_per_label": lab_mean,
            "mean_attn_per_offlabel": off_mean,
            "label_peaking_ratio": (lab_mean / off_mean) if off_mean else float("nan"),
            "mean_signal_cosine": align_mean,
            "mean_signal_cosine_ci": [align_lo, align_hi],
            "attn_by_postype": postype,
            # attention × activation relationship over label tokens (aggregation-study mode-3)
            "ondemo_cos_full": _mean_ci(a["rel_ondemo_cos_mean"], n_boot, seed),
            "routing_corr": _mean_ci(a["rel_routing_corr"], n_boot, seed),
            "routing_corr_cos": _mean_ci(a["rel_routing_corr_cos"], n_boot, seed),
            "routing_corr_cos_null": _mean_ci(
                a["rel_routing_corr_cos_shuf"], n_boot, seed
            ),
            "cos_weighted_label": _mean_ci(a["rel_cos_weighted"], n_boot, seed),
            "cos_uniform_label": _mean_ci(a["rel_cos_uniform"], n_boot, seed),
            "weighting_gain": _mean_ci(a["rel_wgain"], n_boot, seed),
            "weighting_gain_shuffled_null": _mean_ci(a["rel_wgain_shuf"], n_boot, seed),
            "task_mass": float(np.mean(a["task_mass"])),
            "sink_mass": float(np.mean(a["sink_mass"])),
        }
    return out
