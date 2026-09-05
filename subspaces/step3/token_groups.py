"""Step-3 analysis v1: per-source-token FV-share decomposition by token group.

For a head ``(layer_idx, head_idx)`` at the final token position of an n-shot
ICL prompt, the head's output decomposes linearly over source tokens:

    h(p) = sum_t alpha_t * O_h V_h z_t

where ``alpha_t`` is the final-row attention weight (``hook_pattern``) and
``O_h V_h z_t`` is realized as ``hook_v[t, kv_head] @ W_O[head]`` (GQA-mapped;
Llama attention has no biases, so the decomposition is exact — checked by a
``hook_z`` reconstruction gate). For each source token the contribution is
projected onto the head's task direction — the per-head FV component
``h_task = z_results[task][layer_idx, head_idx]`` from the activation z-cache —
and normalized by the total:

    share_t = alpha_t * <O_h V_h z_t, h_task> / <h(p), h_task>

Shares are SIGNED and sum to 1 exactly (linearity); they are invariant to the
scale of ``h_task``, so the raw (un-normalized) direction is used. Tokens are
grouped PER DEMONSTRATION — ``bos``, ``demo{i}_input`` / ``_arrow`` /
``_output`` / ``_sep`` for each demo i, then ``query_input``, ``query_arrow``
(the final position) — via the sample manifest's ``prompt_format`` segments
(``arrow`` maps to out_pre ``->``; ``sep`` to out_sep ``#``), never hard-coded
delimiters. Group spans come from the fast tokenizer's character offsets
(grouping v2: each token joins the group owning its first character — an
exact partition for any tokenizer, bit-identical to the v1 prefix-consistent
spans wherever those existed); prompts with malformed offsets are skipped and
counted.

Per-group shares (the sum of the group's token shares) are accumulated over
all prompts of all tasks shared by the sample manifest and the z-cache; the
summary records mean / variance / [min, max] / 95% bootstrap CI per group,
plus per-task means and denominator diagnostics. A signed share can exceed 1
or go negative when other groups contribute with opposite sign; prompts with
a negative or near-zero denominator are counted separately.

Artifact: kind ``step3_token_groups``, one manifest per complete identity
(``step3-tokengroups-<h10>.json``) with an npz sidecar of per-prompt group
shares and per-head bar charts rendered from the same arrays. Plots are
presentation-only side outputs (not part of the manifest identity/payload).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np

from subspaces.artifacts import (
    ArtifactError,
    identity_of,
    make_manifest,
    manifest_ref,
    read_manifest,
    reuse_or_refuse,
    write_json_atomic,
)
from subspaces.context import AnalysisContext
from subspaces.head_sets import Head, read_heads_manifest
from subspaces.paths import ProjectPaths

TOKEN_GROUPS_KIND = "step3_token_groups"
TOKEN_GROUPS_SCHEMA_VERSION = 1
STEM_PREFIX = "step3-tokengroups"
IMPL = {"module": "subspaces.step3.token_groups", "algorithm_version": 1}
# v2 (2026-08-04): char-offset grouping — each token joins the group owning
# its FIRST character (HF fast-tokenizer offset_mapping). Exact partition for
# ANY format/tokenizer; reproduces v1's prefix-tokenization spans bit-exactly
# wherever v1 succeeded (a token inside one segment starts in it), and closes
# v1's failure mode (boundary-crossing BPE merges: the ab/fx/textlabel
# trailing-space in_sep merges 100% of prompts; '#'+word merges skip 12-29%
# of the text-family prompts). Boundary tokens follow reading order: a merged
# token's chars beyond its first belong to later segments but are counted
# with the first char's group. v1 stays available as token_group_spans for
# fixture equivalence.
GROUPING = {
    "method": "format_segments_offsets",
    "version": 2,
    "per_demo": True,
    "assignment": "first_char",
}
DIRECTION = {"source": "z_cache_task_mean", "space": "d_model", "normalized": False}
DENOMINATOR_ABS_EPS = 1e-10
SMALL_DENOMINATOR_FRACTION = 0.05  # diagnostic flag only, never excludes prompts
RECONSTRUCTION_MIN_COSINE = 0.999
INTERVALS = ("var", "std", "minmax")

# Bar colors by token role (validated categorical palette, slots 1-4 + muted).
ROLE_COLORS = {
    "input": "#2a78d6",
    "arrow": "#eb6834",
    "output": "#1baf7a",
    "sep": "#eda100",
    "bos": "#898781",
}


class GroupingError(ValueError):
    """A prompt whose token-group segmentation cannot be established."""


# --------------------------------------------------------------------------- #
# token grouping (pure string/token logic — no torch)                          #
# --------------------------------------------------------------------------- #


def prompt_parts(
    record: dict, prompt_format: dict, n_shot: int
) -> list[tuple[str, str]]:
    """Split one sample record's prompt into named format segments.

    Segment roles follow the format's template ``in_pre + x + in_sep + out_pre
    + y + out_sep``: the *input* group is ``in_pre + x + in_sep``, *arrow* is
    ``out_pre``, *output* is ``y``, *sep* is ``out_sep``. The query contributes
    ``query_input`` and ``query_arrow`` (no output). Raises
    :class:`GroupingError` when the record does not carry ``n_shot`` demos or
    the segments do not reassemble the stored prompt byte-for-byte.
    """
    demos = record.get("demos") or []
    if len(demos) != n_shot:
        raise GroupingError(f"record has {len(demos)} demos, expected n_shot={n_shot}")
    try:
        in_pre = prompt_format["in_pre"]
        out_pre = prompt_format["out_pre"]
        in_sep = prompt_format["in_sep"]
        out_sep = prompt_format["out_sep"]
    except KeyError as err:
        raise GroupingError(f"prompt_format lacks segment field {err}") from err
    parts: list[tuple[str, str]] = []
    try:
        for demo_num, demo in enumerate(demos, start=1):
            parts.append((f"demo{demo_num}_input", f"{in_pre}{demo['input']}{in_sep}"))
            parts.append((f"demo{demo_num}_arrow", out_pre))
            parts.append((f"demo{demo_num}_output", str(demo["output"])))
            parts.append((f"demo{demo_num}_sep", out_sep))
        parts.append(("query_input", f"{in_pre}{record['query']['input']}{in_sep}"))
        parts.append(("query_arrow", out_pre))
        prompt = record["prompt"]
    except (KeyError, TypeError) as err:
        raise GroupingError(f"malformed sample record: {err}") from err
    reassembled = "".join(text for _name, text in parts)
    if reassembled != prompt:
        raise GroupingError(
            "format segments do not reassemble the stored prompt "
            f"({reassembled!r} != {record['prompt']!r})"
        )
    return parts


def token_group_spans(
    to_tokens: Callable[[str], list[int]],
    parts: list[tuple[str, str]],
    prompt: str,
) -> list[tuple[str, int, int]]:
    """Token spans ``(group, start, end)`` for each segment plus ``bos``.

    ``to_tokens`` must map a string to the FULL token-id list the model sees
    (BOS included). Boundaries come from incremental prefix tokenization; each
    prefix must tokenize to an exact prefix of the full prompt's ids
    (prefix-consistency), otherwise a BPE merge crosses a segment boundary and
    the prompt is unusable for grouping (:class:`GroupingError`).
    """
    full = to_tokens(prompt)
    bos_ids = to_tokens("")
    if bos_ids and full[: len(bos_ids)] != bos_ids:
        raise GroupingError("prompt token ids do not start with the BOS prefix")
    spans: list[tuple[str, int, int]] = []
    if bos_ids:
        spans.append(("bos", 0, len(bos_ids)))
    prev = len(bos_ids)
    cumulative = ""
    for name, text in parts:
        cumulative += text
        ids = to_tokens(cumulative)
        if ids != full[: len(ids)]:
            raise GroupingError(
                f"tokenization is not prefix-consistent at segment {name!r} "
                "(a BPE merge crosses the segment boundary)"
            )
        if len(ids) < prev:
            raise GroupingError(f"segment {name!r} shrank the token prefix")
        spans.append((name, prev, len(ids)))
        prev = len(ids)
    if prev != len(full):
        raise GroupingError(
            f"segments cover {prev} tokens but the prompt has {len(full)}"
        )
    return spans


def token_group_spans_offsets(
    parts: list[tuple[str, str]],
    prompt: str,
    offsets: list[tuple[int, int]],
    n_bos: int,
) -> list[tuple[str, int, int]]:
    """Token spans ``(group, start, end)`` via char offsets (grouping v2).

    ``offsets`` are the fast-tokenizer character spans of the NON-BOS tokens
    (aligned with the model's token ids after the ``n_bos`` BOS prefix). Each
    token joins the segment owning its FIRST character, so the spans are an
    exact, order-preserving partition of the token axis for any tokenizer —
    segments whose characters were fully swallowed by a preceding boundary
    merge get an EMPTY span. Bit-identical to :func:`token_group_spans` on
    prefix-consistent prompts. Malformed offsets (empty, non-monotone, not
    covering the prompt) raise :class:`GroupingError`.
    """
    if not offsets:
        raise GroupingError("tokenizer returned no offsets for the prompt")
    starts = [start for start, _end in offsets]
    if any(start >= end for start, end in offsets):
        raise GroupingError("tokenizer produced an empty/degenerate token offset")
    if starts != sorted(starts):
        raise GroupingError("token offsets are not monotone")
    if starts[0] != 0 or offsets[-1][1] != len(prompt):
        raise GroupingError(
            f"token offsets cover [{starts[0]}, {offsets[-1][1]}) but the "
            f"prompt has {len(prompt)} characters"
        )
    boundaries: list[tuple[str, int]] = []  # (name, char_start)
    cursor = 0
    for name, text in parts:
        boundaries.append((name, cursor))
        cursor += len(text)
    if cursor != len(prompt):
        raise GroupingError(
            f"segments cover {cursor} characters but the prompt has {len(prompt)}"
        )
    spans: list[tuple[str, int, int]] = []
    if n_bos:
        spans.append(("bos", 0, n_bos))
    token = 0
    for index, (name, char_start) in enumerate(boundaries):
        char_end = (
            boundaries[index + 1][1] if index + 1 < len(boundaries) else len(prompt)
        )
        start_token = token
        while token < len(starts) and char_start <= starts[token] < char_end:
            token += 1
        spans.append((name, n_bos + start_token, n_bos + token))
    if token != len(offsets):
        raise GroupingError(
            f"assigned {token} tokens to segments but the prompt has " f"{len(offsets)}"
        )
    return spans


def group_order_from_spans(spans: list[tuple[str, int, int]]) -> list[str]:
    return [name for name, _start, _end in spans]


def span_sums(values: np.ndarray, spans: list[tuple[str, int, int]]) -> np.ndarray:
    """Sum a per-token vector within each span (empty spans sum to 0)."""
    return np.array(
        [float(values[start:end].sum()) for _name, start, end in spans],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# statistics (pure numpy)                                                      #
# --------------------------------------------------------------------------- #


def _bootstrap_mean_ci(
    values: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean (signals.py convention:
    ``np.random.default_rng``, resample with replacement, 2.5/97.5)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    resampled = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return float(np.quantile(resampled, 0.025)), float(np.quantile(resampled, 0.975))


def summarize_groups(
    shares: np.ndarray,
    group_order: list[str],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, dict]:
    """Per-group mean / variance (ddof=1) / [min, max] / bootstrap CI / n over
    the prompt axis of a ``(n_prompts, n_groups)`` share matrix."""
    if shares.ndim != 2 or shares.shape[1] != len(group_order):
        raise ValueError(
            f"share matrix {shares.shape} does not match {len(group_order)} groups"
        )
    out: dict[str, dict] = {}
    for column, group in enumerate(group_order):
        col = shares[:, column]
        n_prompts = int(col.size)
        if n_prompts == 0:
            # None (not NaN) keeps the manifest strict-JSON parseable.
            out[group] = {
                "mean": None,
                "var": None,
                "min": None,
                "max": None,
                "ci": None,
                "n": 0,
            }
            continue
        variance = float(col.var(ddof=1)) if n_prompts > 1 else 0.0
        ci_lo, ci_hi = _bootstrap_mean_ci(col, n_boot=n_boot, seed=seed)
        out[group] = {
            "mean": float(col.mean()),
            "var": variance,
            "min": float(col.min()),
            "max": float(col.max()),
            "ci": [ci_lo, ci_hi],
            "n": n_prompts,
        }
    return out


def group_role(group: str) -> str:
    """Map a group name to its token role (for colors/legend)."""
    if group == "bos":
        return "bos"
    suffix = group.rsplit("_", 1)[-1]
    return suffix if suffix in ("input", "arrow", "output", "sep") else "input"


def group_display_label(group: str) -> str:
    """Compact x-tick label: demo3_output -> out3, query_arrow -> ar_q."""
    if group == "bos":
        return "bos"
    short = {"input": "in", "arrow": "ar", "output": "out", "sep": "sep"}
    if group.startswith("query_"):
        return f"{short[group.split('_', 1)[1]]}_q"
    head, _, role = group.partition("_")
    return f"{short[role]}{head.removeprefix('demo')}"


# --------------------------------------------------------------------------- #
# model-facing computation (torch imported at call time)                       #
# --------------------------------------------------------------------------- #


def _capture(model, prompt: str, layers: list[int], *, with_z: bool):
    """One forward pass capturing hook_pattern + hook_v (and hook_z when
    gating) at the given layers. Batch of 1; ``final_pos = seq_len - 1``."""
    import torch  # heavy

    tokens = model.to_tokens(prompt, prepend_bos=True).to(model.cfg.device)
    cache: dict[tuple[int, str], object] = {}

    def grab(activation, hook, key):
        cache[key] = activation.detach()

    hooks = []
    for layer_idx in layers:
        hooks.append(
            (
                f"blocks.{layer_idx}.attn.hook_pattern",
                partial(grab, key=(layer_idx, "pattern")),
            )
        )
        hooks.append(
            (f"blocks.{layer_idx}.attn.hook_v", partial(grab, key=(layer_idx, "v")))
        )
        if with_z:
            hooks.append(
                (
                    f"blocks.{layer_idx}.attn.hook_z",
                    partial(grab, key=(layer_idx, "z")),
                )
            )
    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=hooks)
    return cache, int(tokens.shape[1])


def head_source_contributions(
    cache, model, layer_idx: int, head_idx: int, final_pos: int, direction
) -> np.ndarray:
    """Per-source-token signed contribution ``alpha_t * <O_h V_h z_t, dir>``
    at the final position, float64, shape (seq,). Sums to ``<h(p), dir>``."""
    import torch  # heavy

    from subspaces.utils.signals import value_head_for_query_head

    pattern = cache[(layer_idx, "pattern")][0, head_idx]  # (q, k)
    alpha = pattern[final_pos].to(torch.float32)
    v_all = cache[(layer_idx, "v")][0]  # (seq, n_kv_heads, d_head)
    kv_idx = value_head_for_query_head(model, head_idx, int(v_all.shape[1]))
    w_o = model.blocks[layer_idx].attn.W_O[head_idx].to(torch.float32)
    s_full = v_all[:, kv_idx, :].to(torch.float32) @ w_o  # (seq, d_model)
    contribs = alpha * (s_full @ direction.to(torch.float32))
    return contribs.detach().cpu().numpy().astype(np.float64)


def reconstruction_cosine(
    cache, model, layer_idx: int, head_idx: int, final_pos: int
) -> float:
    """Gate: cos(sum_t alpha_t O_h V_h z_t, hook_z[final] @ W_O) ~ 1.0 —
    verifies the GQA mapping and the linearity of the decomposition."""
    import torch  # heavy

    from subspaces.utils.signals import value_head_for_query_head

    pattern = cache[(layer_idx, "pattern")][0, head_idx]
    alpha = pattern[final_pos].to(torch.float32)
    v_all = cache[(layer_idx, "v")][0]
    kv_idx = value_head_for_query_head(model, head_idx, int(v_all.shape[1]))
    w_o = model.blocks[layer_idx].attn.W_O[head_idx].to(torch.float32)
    s_full = v_all[:, kv_idx, :].to(torch.float32) @ w_o
    recomposed = (alpha[:, None] * s_full).sum(dim=0)
    z_final = cache[(layer_idx, "z")][0, final_pos, head_idx].to(torch.float32)
    direct = z_final @ w_o
    denom = recomposed.norm() * direct.norm()
    if float(denom) < 1e-20:
        return float("nan")
    return float((recomposed @ direct) / denom)


def effective_model_identity(context: AnalysisContext, zcache_model: dict) -> dict:
    """The model identity the analysis ACTUALLY runs with: context values win,
    the z-cache's recorded values are the fallback pins. This exact dict goes
    into the artifact identity, so a dtype/revision override forks the
    identity instead of silently sharing it with the default run."""
    return {
        "name": context.model["name"],
        "revision": context.model.get("revision") or zcache_model.get("revision"),
        "dtype": (
            context.model.get("dtype") or zcache_model.get("dtype") or "bfloat16"
        ),
    }


def _load_model_pinned(effective_model: dict, device_override: str | None):
    """Offline model load pinned to the EFFECTIVE identity recorded in the
    artifact. Mirrors subspaces.step1.eval_gpu.load_model."""
    import torch  # heavy

    from subspaces.step1.resolve import resolve_model_identity
    from subspaces.utils.model import load_model_no_grad  # heavy

    name = effective_model["name"]
    pinned = effective_model.get("revision")
    if pinned is not None:
        observed = resolve_model_identity(name)
        if observed["revision"] != pinned:
            raise ArtifactError(
                f"the local cache resolves {name} to revision "
                f"{observed['revision']} but this analysis pins {pinned}; "
                "refusing to load drifted weights."
            )
    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _n_layers, _n_heads = load_model_no_grad(
        name, device, dtype=getattr(torch, effective_model["dtype"])
    )
    return model


def _load_zcache_dir(z_dir: Path):
    """Load a z-cache directory (``meta.json`` + ``z_results.pth``), verifying
    the recorded tensor content digest. Returns ``(z_results, meta)``."""
    import torch  # heavy

    from subspaces.step1.zcache import ZCACHE_SCHEMA_VERSION, tensors_content_sha

    meta = read_manifest(
        z_dir / "meta.json",
        expect_kind="z_cache",
        max_schema_version=ZCACHE_SCHEMA_VERSION,
    )
    tensor_path = z_dir / "z_results.pth"
    if not tensor_path.is_file():
        raise ArtifactError(f"{tensor_path} is missing; incomplete z cache")
    z_results = torch.load(tensor_path, map_location="cpu")
    recorded = meta.get("content_sha256")
    if recorded is not None and tensors_content_sha(z_results) != recorded:
        raise ArtifactError(
            f"{z_dir}: z tensor payload does not match the recorded content "
            "digest; refusing a corrupted cache."
        )
    return z_results, meta


# --------------------------------------------------------------------------- #
# artifact plumbing                                                            #
# --------------------------------------------------------------------------- #


def _head_key(head: Head) -> str:
    return f"({head[0]},{head[1]})"


def _npz_prefix(head: Head) -> str:
    return f"L{head[0]}H{head[1]}"


def build_identity(
    *,
    cfg,
    heads: list[Head],
    inputs: dict,
    task: dict,
    model: dict,
) -> dict:
    """The complete reuse identity (kind, schema_version, inputs, config, impl)
    of one token-group analysis. Presentation options (plot interval) and input
    locator names are deliberately excluded."""
    config = {
        "analysis": {"name": "token_groups", "version": IMPL["algorithm_version"]},
        "heads": [list(head) for head in sorted(heads)],
        "task": {
            "family": task.get("family"),
            "prompt_format": task.get("prompt_format"),
            "n_shot": task.get("n_shot"),
        },
        "model": {
            "name": model.get("name"),
            "revision": model.get("revision"),
            "dtype": model.get("dtype"),
        },
        "n_prompts_per_task": cfg.n_prompts_per_task,
        "seed": cfg.seed,
        "n_boot": cfg.n_boot,
        "direction": dict(DIRECTION),
        "grouping": dict(GROUPING),
        "proportion": {
            "kind": "signed_share",
            "denominator": "sum_over_source_tokens",
            "abs_eps": DENOMINATOR_ABS_EPS,
        },
    }
    return {
        "kind": TOKEN_GROUPS_KIND,
        "schema_version": TOKEN_GROUPS_SCHEMA_VERSION,
        "inputs": inputs,
        "config": config,
        "impl": IMPL,
    }


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Same atomic-write discipline as the JSON manifests (unique same-dir
    temp file + os.replace) so an interrupted run never leaves a truncated
    sidecar next to a valid manifest."""
    import os
    import tempfile

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        with open(tmp_name, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _verify_npz(base_dir: Path, manifest: dict) -> None:
    """A reused manifest must still have its npz sidecar, content-intact."""
    from subspaces.step1.recovery import outcomes_content_sha

    npz_path = base_dir / manifest["outcomes"]["file"]
    if not npz_path.is_file():
        raise ArtifactError(
            f"{npz_path} is missing but the manifest references it; "
            "refusing to reuse an incomplete artifact."
        )
    try:
        loaded = np.load(npz_path)
        arrays = {key: loaded[key] for key in loaded.files}
    except Exception as err:  # zipfile.BadZipFile, OSError, ValueError, ...
        raise ArtifactError(
            f"{npz_path}: unreadable outcomes sidecar ({err}); "
            "refusing to reuse a corrupted artifact."
        ) from err
    actual = outcomes_content_sha(arrays)
    if actual != manifest["outcomes"]["content_sha256"]:
        raise ArtifactError(
            f"{npz_path}: content digest mismatch "
            f"({actual[:12]} vs recorded {manifest['outcomes']['content_sha256'][:12]})"
        )


# --------------------------------------------------------------------------- #
# plots (presentation-only side outputs; rendered from the summary + npz)      #
# --------------------------------------------------------------------------- #


def render_plots(manifest: dict, out_dir: Path, interval: str = "var") -> list[Path]:
    """One bar chart per head: mean group share (bar) with an interval band
    (``var`` = ±variance, per the requested convention; ``std`` = ±sd;
    ``minmax`` = the full [min, max] range). Files land next to the manifest
    under ``<stem>-plots/`` and are NOT part of the artifact identity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if interval not in INTERVALS:
        raise ValueError(f"interval must be one of {INTERVALS}: {interval!r}")
    stem = Path(manifest["outcomes"]["file"]).name.removesuffix("-outcomes.npz")
    plots_dir = out_dir / f"{stem}-plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    group_order = manifest["group_order"]
    written: list[Path] = []
    for head_key, record in manifest["per_head"].items():
        if not record["n_prompts"]:
            continue  # nothing to plot for a fully-degenerate head
        layer_idx, head_idx = (int(x) for x in head_key.strip("()").split(","))
        groups = record["groups"]
        means = np.array([groups[g]["mean"] for g in group_order])
        if interval == "var":
            err_lo = err_hi = np.array([groups[g]["var"] for g in group_order])
        elif interval == "std":
            err_lo = err_hi = np.array([np.sqrt(groups[g]["var"]) for g in group_order])
        else:  # minmax
            err_lo = means - np.array([groups[g]["min"] for g in group_order])
            err_hi = np.array([groups[g]["max"] for g in group_order]) - means
        colors = [ROLE_COLORS[group_role(g)] for g in group_order]
        n_groups = len(group_order)
        fig, ax = plt.subplots(figsize=(max(7.0, 0.38 * n_groups + 2.0), 3.6))
        ax.bar(
            range(n_groups),
            means,
            color=colors,
            yerr=np.vstack([np.clip(err_lo, 0, None), np.clip(err_hi, 0, None)]),
            capsize=2,
            error_kw={"elinewidth": 0.9, "ecolor": "#52514e"},
        )
        ax.axhline(0.0, color="#c3c2b7", linewidth=0.8)
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(
            [group_display_label(g) for g in group_order],
            rotation=60,
            ha="right",
            fontsize=8,
        )
        interval_label = {
            "var": "± variance",
            "std": "± std",
            "minmax": "[min, max]",
        }[interval]
        ax.set_ylabel(f"mean share of final-token\nFV projection ({interval_label})")
        ax.set_title(
            f"L{layer_idx}H{head_idx} — FV share by token group "
            f"(n={record['n_prompts']})"
        )
        roles_present = []
        for role in ("input", "arrow", "output", "sep", "bos"):
            if any(group_role(g) == role for g in group_order):
                roles_present.append(role)
        ax.legend(
            handles=[
                Patch(facecolor=ROLE_COLORS[role], label=role) for role in roles_present
            ],
            fontsize=8,
            frameon=False,
            ncol=len(roles_present),
            loc="upper left",
        )
        fig.tight_layout()
        out_png = plots_dir / f"token_groups_L{layer_idx}H{head_idx}_{interval}.png"
        fig.savefig(str(out_png), dpi=120)
        plt.close(fig)
        written.append(out_png)
    return written


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #


def run_token_groups(
    cfg,
    heads: list[Head],
    paths: ProjectPaths,
    context: AnalysisContext,
    *,
    heads_artifact: str | Path | None,
    out_dir: Path,
) -> dict:
    """Compute (or reuse) one token-group decomposition artifact.

    ``context.samples[cfg.samples_name]`` supplies the analysis prompts;
    ``context.activations`` the z-cache holding the per-task head directions.
    ``out_dir`` is the artifact home (the heads artifact's node dir by
    default). Returns the manifest dict.
    """
    from subspaces.step1.recovery import outcomes_content_sha

    samples_ref = context.samples[cfg.samples_name]
    samples_path = paths.resolve(samples_ref["path"])
    samples_manifest = read_manifest(
        samples_path, expect_kind="samples", max_schema_version=1
    )
    prompt_format = samples_manifest["prompt_format"]
    manifest_task = samples_manifest.get("config", {}).get("task", {})
    manifest_n_shot = manifest_task.get("n_shot")
    n_shot = int(
        manifest_n_shot if manifest_n_shot is not None else context.task["n_shot"]
    )
    for field, manifest_value in (
        ("prompt_format", prompt_format.get("name")),
        ("n_shot", manifest_n_shot),
    ):
        context_value = context.task.get(field)
        if manifest_value is not None and context_value != manifest_value:
            raise ArtifactError(
                f"context.task.{field}={context_value!r} does not match the "
                f"sample manifest ({manifest_value!r}); refusing mismatched "
                "protocol inputs."
            )

    activation_refs = context.activation_refs()
    if len(activation_refs) != 1:
        raise ArtifactError(
            "step3 needs exactly ONE activations reference (the z cache "
            "holding the head directions); context declares "
            f"{sorted(activation_refs) or 'none'}. Use the single "
            "`activations: {path: log/cache/z/<fp>}` form."
        )
    (z_ref,) = activation_refs.values()
    z_dir = paths.resolve(z_ref["path"])
    z_meta = read_manifest(
        z_dir / "meta.json", expect_kind="z_cache", max_schema_version=1
    )
    zcache_model = z_meta["identity"]["model"]
    if zcache_model["name"] != context.model["name"]:
        raise ArtifactError(
            f"context.model.name={context.model['name']!r} does not match the "
            f"z cache ({zcache_model['name']!r}); refusing cross-model directions."
        )
    context_revision = context.model.get("revision")
    if context_revision and context_revision != zcache_model.get("revision"):
        raise ArtifactError(
            f"context.model.revision={context_revision} does not match the "
            f"z cache ({zcache_model.get('revision')}); refusing."
        )

    inputs: dict[str, dict] = {
        "samples": manifest_ref(samples_path, paths, samples_manifest),
        "z_cache": {"content_fingerprint": z_meta["cache_fingerprint"]},
    }
    head_provenance: dict = {"source": "explicit"}
    if heads_artifact is not None:
        heads_path = paths.resolve(heads_artifact)
        heads_manifest, heads_kind = read_heads_manifest(heads_path)
        inputs["heads"] = manifest_ref(heads_path, paths, heads_manifest)
        if heads_kind == "heads":
            selector = heads_manifest.get("selector")
        else:
            # selector output (kind main_heads, e.g. a significant node): the
            # selector fields live at the manifest top level
            selector = {
                "name": heads_manifest.get("selector_name"),
                "version": heads_manifest.get("selector_version"),
                "params": heads_manifest.get("params"),
            }
        head_provenance = {
            "source": "heads_artifact",
            "selector": selector,
        }

    effective_model = effective_model_identity(context, zcache_model)
    expected = build_identity(
        cfg=cfg,
        heads=heads,
        inputs=inputs,
        task=context.task,
        model=effective_model,
    )
    stem = f"{STEM_PREFIX}-{identity_of(expected)[:10]}"
    out_path = out_dir / f"{stem}.json"
    existing = reuse_or_refuse(
        out_path,
        expected,
        expect_kind=TOKEN_GROUPS_KIND,
        max_schema_version=TOKEN_GROUPS_SCHEMA_VERSION,
    )
    if existing is not None:
        _verify_npz(out_dir, existing)
        render_plots(existing, out_dir, interval=cfg.interval)
        print(f"[step3] reusing {out_path}")
        return existing

    import torch  # heavy

    z_results, _ = _load_zcache_dir(z_dir)
    model = _load_model_pinned(effective_model, context.model.get("device"))
    device = model.cfg.device

    def to_tokens(text: str) -> list[int]:
        return model.to_tokens(text, prepend_bos=True)[0].tolist()

    n_bos = len(to_tokens(""))

    def spans_for(parts: list[tuple[str, str]], prompt: str):
        """Grouping v2: char-offset spans, cross-checked against the ids the
        model actually sees (a fast-tokenizer/`to_tokens` disagreement must
        skip the prompt, not silently mis-span it)."""
        encoded = model.tokenizer(
            prompt, return_offsets_mapping=True, add_special_tokens=False
        )
        full = to_tokens(prompt)
        if full[:n_bos] != to_tokens("")[:n_bos] or full[n_bos:] != list(
            encoded["input_ids"]
        ):
            raise GroupingError(
                "offset-tokenization ids do not match the model's to_tokens ids"
            )
        offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        return token_group_spans_offsets(parts, prompt, offsets, n_bos)

    tasks = [t for t in samples_manifest["task_order"] if t in z_results]
    tasks_missing_direction = [
        t for t in samples_manifest["task_order"] if t not in z_results
    ]
    if not tasks:
        raise ArtifactError(
            "no task of the sample manifest has a direction in the z cache; "
            "the samples and activations references are inconsistent."
        )
    layers = sorted({layer_idx for layer_idx, _head_idx in heads})

    group_order: list[str] | None = None
    shares_rows: dict[Head, list[np.ndarray]] = {head: [] for head in heads}
    raw_rows: dict[Head, list[np.ndarray]] = {head: [] for head in heads}
    denoms: dict[Head, list[float]] = {head: [] for head in heads}
    task_idx_rows: dict[Head, list[int]] = {head: [] for head in heads}
    n_degenerate: dict[Head, int] = {head: 0 for head in heads}
    gate_min: dict[Head, float] = {head: float("inf") for head in heads}
    gated_tasks: set[str] = set()
    grouped_counts: dict[str, int] = {}
    n_skipped_grouping = 0
    n_prompts_seen = 0

    for task_number, task_id in enumerate(tasks):
        directions = {
            head: z_results[task_id][head[0], head[1]].to(device, torch.float32)
            for head in heads
        }
        samples = samples_manifest["tasks"][task_id]["samples"][
            : cfg.n_prompts_per_task
        ]
        for record in samples:
            n_prompts_seen += 1
            try:
                parts = prompt_parts(record, prompt_format, n_shot)
                spans = spans_for(parts, record["prompt"])
            except GroupingError:
                n_skipped_grouping += 1
                continue
            order = group_order_from_spans(spans)
            if group_order is None:
                group_order = order
            elif order != group_order:
                n_skipped_grouping += 1
                continue
            grouped_counts[task_id] = grouped_counts.get(task_id, 0) + 1
            # gate the first prompt of each task that SURVIVES grouping (a
            # skipped first record must not leave the task ungated)
            with_gate = task_id not in gated_tasks
            cache, seq_len = _capture(model, record["prompt"], layers, with_z=with_gate)
            if seq_len != spans[-1][2]:
                raise ArtifactError(
                    f"span/forward token count mismatch ({spans[-1][2]} vs "
                    f"{seq_len}) for task {task_id}; tokenization drifted "
                    "between grouping and capture."
                )
            final_pos = seq_len - 1
            for head in heads:
                contribs = head_source_contributions(
                    cache, model, head[0], head[1], final_pos, directions[head]
                )
                if with_gate:
                    cosine = reconstruction_cosine(
                        cache, model, head[0], head[1], final_pos
                    )
                    gate_min[head] = min(gate_min[head], cosine)
                    if not (cosine > RECONSTRUCTION_MIN_COSINE):
                        raise ArtifactError(
                            f"reconstruction gate failed for head {head} on "
                            f"task {task_id}: cos={cosine:.6f} <= "
                            f"{RECONSTRUCTION_MIN_COSINE} — the per-token "
                            "decomposition does not recompose the head output."
                        )
                denominator = float(contribs.sum())
                if abs(denominator) < DENOMINATOR_ABS_EPS:
                    n_degenerate[head] += 1
                    continue
                shares_rows[head].append(span_sums(contribs / denominator, spans))
                raw_rows[head].append(span_sums(contribs, spans))
                denoms[head].append(denominator)
                task_idx_rows[head].append(task_number)
            if with_gate:
                gated_tasks.add(task_id)

    if group_order is None:
        raise ArtifactError(
            "every prompt was skipped during token grouping; the prompt "
            "format cannot be segmented under this tokenizer."
        )

    per_head: dict[str, dict] = {}
    npz_arrays: dict[str, np.ndarray] = {}
    for head in heads:
        shares = np.array(shares_rows[head], dtype=np.float64).reshape(
            len(shares_rows[head]), len(group_order)
        )
        raw = np.array(raw_rows[head], dtype=np.float64).reshape(
            len(raw_rows[head]), len(group_order)
        )
        denom_arr = np.array(denoms[head], dtype=np.float64)
        task_arr = np.array(task_idx_rows[head], dtype=np.int32)
        prefix = _npz_prefix(head)
        npz_arrays[f"{prefix}_shares"] = shares
        npz_arrays[f"{prefix}_raw"] = raw
        npz_arrays[f"{prefix}_denominators"] = denom_arr
        npz_arrays[f"{prefix}_task_index"] = task_arr
        groups = summarize_groups(shares, group_order, n_boot=cfg.n_boot, seed=cfg.seed)
        per_task_mean: dict[str, dict[str, float]] = {}
        for task_number, task_id in enumerate(tasks):
            mask = task_arr == task_number
            if mask.any():
                per_task_mean[task_id] = {
                    group: float(shares[mask, column].mean())
                    for column, group in enumerate(group_order)
                }
        n_kept = int(denom_arr.size)
        median_abs = float(np.median(np.abs(denom_arr))) if n_kept else 0.0
        per_head[_head_key(head)] = {
            "groups": groups,
            "per_task_mean": per_task_mean,
            "n_prompts": n_kept,
            "n_degenerate_denominator": n_degenerate[head],
            "n_negative_denominator": int((denom_arr < 0).sum()),
            "n_small_denominator": int(
                (np.abs(denom_arr) < SMALL_DENOMINATOR_FRACTION * median_abs).sum()
            ),
            "mean_denominator": float(denom_arr.mean()) if n_kept else None,
            "reconstruction_cosine_min": (
                gate_min[head] if np.isfinite(gate_min[head]) else None
            ),
        }

    from subspaces.step1.eval_gpu import gpu_peak_memory

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_name = f"{stem}-outcomes.npz"
    _write_npz_atomic(out_dir / npz_name, npz_arrays)
    payload = {
        "impl": IMPL,
        "group_order": group_order,
        "task_order_analyzed": tasks,
        "tasks_missing_direction": tasks_missing_direction,
        "head_set_provenance": head_provenance,
        "n_prompts_seen": n_prompts_seen,
        "n_skipped_grouping": n_skipped_grouping,
        # truth-in-payload: config.n_prompts_per_task is the REQUESTED cap;
        # these record what each task actually supplied / passed grouping
        "n_prompts_grouped_per_task": grouped_counts,
        "n_tasks_gated": len(gated_tasks),
        "per_head": per_head,
        "outcomes": {
            "file": npz_name,
            "content_sha256": outcomes_content_sha(npz_arrays),
        },
        "gpu_memory": gpu_peak_memory(),
    }
    manifest = make_manifest(
        kind=TOKEN_GROUPS_KIND,
        schema_version=TOKEN_GROUPS_SCHEMA_VERSION,
        paths=paths,
        config=expected["config"],
        inputs=inputs,
        payload=payload,
    )
    if identity_of(manifest) != identity_of(expected):
        raise ArtifactError(
            "internal error: the written manifest's identity drifted from the "
            "pre-computed identity; refusing to write an unreusable artifact."
        )
    write_json_atomic(out_path, manifest)
    render_plots(manifest, out_dir, interval=cfg.interval)
    print(f"[step3] wrote {out_path}")
    return manifest
