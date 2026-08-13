"""AIE (Average Indirect Effect) head-selection baseline — Todd et al.,
*Function Vectors in Large Language Models* (arXiv:2310.15213), adapted to
this repo's corrected protocol.

A self-contained stage family per (model × task-family) cell:

1. **scores** (GPU): per-head AIE over the committed TRAIN tasks only. Two
   versioned methods (each rides in artifact identity via ``impl_for``):

   - ``cie_replace`` v2 — paper-faithful CIE: forward corrupted n-shot
     prompts (demo labels permuted within-prompt; the ``aie_corrupted``
     sample kind), REPLACE one head's last-token post-W_O contribution with
     its task-conditioned mean z̄_t from the standard activation z-cache,
     and read the change in the FIRST expected-target-token probability
     (float32 softmax at the last position). CIE(head | t) = mean over the
     task's prompts; AIE = unweighted mean over train tasks (paper Eq. 4).
     Implemented as an additive delta at ``blocks.{l}.hook_attn_out``
     (``cie_patched_last_logits``) — algebraically identical to replacing
     ``act[:, -1, head, :]`` on ``attn.hook_result`` under
     ``use_attn_result=True`` (CPU-equivalence-tested), WITHOUT that flag's
     per-layer ``[batch, pos, head, d_head, d_model]`` intermediate, which
     OOMs on the long-sequence cells. v1 used the flag and never produced
     artifacts (superseded before any GPU run).
   - ``zs_add_proxy`` v1 — the historical proxy from the original research
     code (provenance of the paper's "Todd 0.31" lineage; legacy
     the legacy AIE notebook): the same
     first-token readout on the activation manifest's ZERO-SHOT prompts,
     with z̄_t[l, h] ADDED at the cell's inject site
     (``cfg.sites.inject_layer``) instead of replaced at its own head.

2. **top-k selection** (CPU): rank by SIGNED AIE descending with the
   deterministic tie-break (AIE desc, layer_idx asc, head_idx asc). Signed,
   never abs — the paper's ranking (a later abs() re-ranking in the legacy
   code was a deviation).

3. **eval** (GPU): per held-out task the Todd FV = Σ_{top-k} z̄heldout_t is
   ADDED at the inject site on the final-evaluation zero-shot prompts
   (intervention mode 0, versioned scorer) with a paired clean n-shot arm —
   identical protocol and z tensors (heldout_activation cache, seed 1043) to
   the framework's headset evaluation.

The chain deliberately does NOT enter the significant→scan→main lineage: no
scan is needed, artifacts are hash-named under ``<journal run dir>/aie/``,
and reuse follows the shared refuse-on-mismatch rule. This module's IMPL is
deliberately NOT added to ``subspaces.step1.resolve.algorithm_versions()`` — that
would fork every existing run identity; it joins the aie artifacts' own
identities via ``impl_for`` instead. Heavy imports are confined to call
time; importing this module stays light.
"""

from __future__ import annotations

from pathlib import Path

from subspaces.artifacts import (
    ArtifactError,
    identity_of,
    make_manifest,
    manifest_ref,
    reuse_or_refuse,
    write_json_atomic,
)
from subspaces.config import ConfigError, Step1Config, to_dict
from subspaces.paths import ProjectPaths

AIE_SCORES_SCHEMA_VERSION = 1
AIE_EVAL_SCHEMA_VERSION = 1
# Shared-engine version (tokenization, batching, artifact assembly); each
# method's own (name, version) joins identity via impl_for, so bumping one
# method never forks the other's artifacts.
ENGINE_VERSION = 1
IMPL = {"module": "subspaces.step1.aie", "algorithm_version": ENGINE_VERSION}
# cie_replace v2: delta-at-hook_attn_out implementation (v1's
# use_attn_result replacement OOMed on long-sequence cells and never
# produced artifacts).
METHOD_VERSIONS: dict[str, int] = {"cie_replace": 2, "zs_add_proxy": 1}
READOUT = "first_token_prob"
RANKING = "signed_desc"
HEAD_ORDER = "row_major(layer_idx,head_idx)"


def impl_for(method: str) -> dict:
    """Implementation identity for one AIE method: shared-engine version plus
    the method's own (name, version) — mirrors subspaces.step1.significant."""
    if method not in METHOD_VERSIONS:
        raise ConfigError(
            f"unknown AIE method {method!r}; registered: {sorted(METHOD_VERSIONS)}"
        )
    return {**IMPL, "method": {"name": method, "version": METHOD_VERSIONS[method]}}


# -- pure CPU math -------------------------------------------------------------


def first_token_probs(last_logits, target_token_ids):
    """Probability of each prompt's first expected-target token.

    ``last_logits`` are LAST-POSITION logits ``[batch, vocab]`` (any float
    dtype; upcast to float32 BEFORE the softmax), ``target_token_ids`` is
    ``[batch]``. Returns a float32 ``[batch]`` tensor.
    """
    import torch  # heavy

    if last_logits.ndim != 2:
        raise ArtifactError(
            f"expected [batch, vocab] logits, got shape {tuple(last_logits.shape)}"
        )
    if target_token_ids.ndim != 1 or target_token_ids.shape[0] != last_logits.shape[0]:
        raise ArtifactError(
            "target_token_ids must be [batch] aligned with the logits: got "
            f"{tuple(target_token_ids.shape)} for batch {last_logits.shape[0]}"
        )
    index = target_token_ids.view(-1, 1).to(device=last_logits.device)
    probs = torch.softmax(last_logits.to(torch.float32), dim=-1)
    return probs.gather(1, index).squeeze(1)


def first_token_prob_deltas(baseline_logits, patched_logits, target_token_ids):
    """P_patch − P_base of the first expected-target token, per prompt.

    Pure function over (baseline last-position logits, patched last-position
    logits, target first-token ids); float32 softmax, first-token gather.
    The per-(head, task) CIE is the MEAN of these deltas over the task's
    prompts (taken by the caller).
    """
    if baseline_logits.shape != patched_logits.shape:
        raise ArtifactError(
            "baseline/patched logits shape mismatch: "
            f"{tuple(baseline_logits.shape)} vs {tuple(patched_logits.shape)}"
        )
    base = first_token_probs(baseline_logits, target_token_ids)
    patched = first_token_probs(patched_logits, target_token_ids)
    return patched - base


def head_grid(
    n_layers: int, n_heads: int, head_limit: int | None = None
) -> list[tuple[int, int]]:
    """Row-major (layer_idx, head_idx) grid, optionally capped by a flat
    prefix (``aie.head_limit`` — smoke configs only; recorded in identity)."""
    if n_layers < 1 or n_heads < 1:
        raise ArtifactError(f"degenerate head grid {n_layers}x{n_heads}")
    heads = [
        (layer_idx, head_idx)
        for layer_idx in range(n_layers)
        for head_idx in range(n_heads)
    ]
    return heads[:head_limit] if head_limit else heads


def select_top_k(aie_matrix, scored_heads, k: int) -> list[tuple[int, int, float]]:
    """Top-k heads by SIGNED AIE descending (the paper's ranking; never abs).

    Deterministic tie-break: (AIE desc, layer_idx asc, head_idx asc).
    ``aie_matrix`` is indexable as ``aie_matrix[layer_idx][head_idx]``;
    ``scored_heads`` restricts ranking to the heads the scores artifact
    actually computed (the full grid unless head_limit was set).
    """
    if k < 1:
        raise ConfigError(f"k must be >= 1: {k}")
    if k > len(scored_heads):
        raise ConfigError(
            f"k={k} exceeds the {len(scored_heads)} scored heads "
            "(head grid, or aie.head_limit)"
        )
    ranked: list[tuple[int, int, float]] = []
    for layer_idx, head_idx in scored_heads:
        value = float(aie_matrix[layer_idx][head_idx])
        if value != value:  # NaN
            raise ArtifactError(
                f"AIE score for head ({layer_idx},{head_idx}) is NaN; refusing "
                "to rank an incompletely scored grid"
            )
        ranked.append((layer_idx, head_idx, value))
    ranked.sort(key=lambda item: (-item[2], item[0], item[1]))
    return ranked[:k]


def arrays_content_sha(arrays: dict) -> str:
    """Deterministic digest of named arrays (key + dtype + shape + bytes;
    npz bytes embed zip timestamps and are not reproducible — the
    recovery.py outcomes precedent, extended with dtype/shape for the
    non-uint8 score matrices)."""
    import hashlib

    import numpy as np

    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


# -- artifact identities -------------------------------------------------------


def scores_identity(
    cfg: Step1Config,
    method: str,
    samples_ref: dict,
    z_fingerprint: str,
    examples_per_task: int,
) -> dict:
    """Identity envelope of one ``aie_scores`` artifact (computable before
    any GPU work, so reuse decisions stay CPU-only on cache hits).

    The sites block joins the identity ONLY for ``zs_add_proxy`` (it patches
    AT the inject site); ``cie_replace`` patches each head at its own layer
    and never reads ``inject_layer`` — including it would over-fork the
    expensive artifact on an irrelevant setting."""
    batch_size = (
        cfg.aie.batch_size if method == "cie_replace" else cfg.aie.proxy_batch_size
    )
    config = {
        "method": {"name": method, "version": METHOD_VERSIONS[method]},
        "batch_size": batch_size,
        "readout": READOUT,
        "examples_per_task": examples_per_task,
        "head_limit": cfg.aie.head_limit,
    }
    if method == "zs_add_proxy":
        config["sites"] = to_dict(cfg.sites)
    return {
        "kind": "aie_scores",
        "schema_version": AIE_SCORES_SCHEMA_VERSION,
        "inputs": {
            "samples": samples_ref,
            "z_cache": {"content_fingerprint": z_fingerprint},
        },
        "config": config,
        "impl": impl_for(method),
    }


def eval_identity(
    cfg: Step1Config,
    method: str,
    k: int,
    scores_ref: dict,
    final_eval_ref: dict,
    z_heldout_fingerprint: str,
) -> dict:
    """Identity envelope of one ``aie_eval`` artifact (method × k)."""
    from subspaces.step1.eval_gpu import effective_batch_size

    return {
        "kind": "aie_eval",
        "schema_version": AIE_EVAL_SCHEMA_VERSION,
        "inputs": {
            "aie_scores": scores_ref,
            "final_eval_samples": final_eval_ref,
            "z_cache_heldout": {"content_fingerprint": z_heldout_fingerprint},
        },
        "config": {
            "k": k,
            "ranking": RANKING,
            "sites": to_dict(cfg.sites),
            "intervention_mode": 0,
            "batch_size": effective_batch_size(cfg),
            "scoring": to_dict(cfg.scoring),
        },
        "impl": impl_for(method),
    }


def _verify_scores_arrays(base_dir: Path, manifest: dict) -> None:
    """A reused aie_scores artifact must still have its npz, content-intact."""
    import numpy as np

    npz_path = Path(base_dir) / manifest["arrays"]["file"]
    if not npz_path.is_file():
        raise ArtifactError(
            f"{npz_path} is missing but the manifest references it; refusing "
            "to reuse an incomplete artifact."
        )
    loaded = np.load(npz_path)
    actual = arrays_content_sha({key: loaded[key] for key in loaded.files})
    expected = manifest["arrays"]["content_sha256"]
    if actual != expected:
        raise ArtifactError(
            f"{npz_path}: score-array content digest mismatch "
            f"({actual[:12]} vs recorded {expected[:12]}); refusing."
        )


# -- GPU score computation -----------------------------------------------------


def _target_first_token_ids(model, targets: list[str]):
    """First token of each target under the standard scorer convention:
    tokenized STANDALONE (``prepend_bos=False``), first token taken. An
    empty-tokenizing target has no first token — the readout is undefined,
    so it refuses cleanly (the known abstractive empty-target datapoint)."""
    import torch  # heavy

    token_ids = []
    for target in targets:
        tokens = model.to_tokens(target, prepend_bos=False)
        if tokens.shape[1] == 0:
            raise ArtifactError(
                f"target {target!r} tokenizes to zero tokens; the "
                "first-token-probability readout is undefined for it — "
                "refusing (validity overlay: handle the datapoint explicitly)"
            )
        token_ids.append(int(tokens[0, 0].item()))
    return torch.tensor(token_ids, dtype=torch.long, device=model.cfg.device)


def _forward_last_logits(model, tokens, mask, fwd_hooks=None):
    import torch  # heavy

    with torch.no_grad():
        if fwd_hooks:
            logits = model.run_with_hooks(
                tokens, fwd_hooks=fwd_hooks, attention_mask=mask, return_type="logits"
            )
        else:
            logits = model(tokens, attention_mask=mask, return_type="logits")
    # clone: keep only the last position, freeing the [batch, seq, vocab] block
    return logits[:, -1, :].detach().clone()


def _capture_last_z_hooks(n_layers: int, sink: dict) -> list:
    """Forward hooks capturing every layer's LAST-TOKEN ``attn.hook_z`` slice
    (``[batch, n_heads, d_head]`` — tiny), available without
    ``use_attn_result``."""
    hooks = []
    for layer_idx in range(n_layers):

        def capture(activation, hook, *, _layer=layer_idx, _sink=sink):
            _sink[_layer] = activation[:, -1, :, :].detach().clone()
            return activation

        hooks.append((f"blocks.{layer_idx}.attn.hook_z", capture))
    return hooks


def cie_baseline_forward(model, tokens, mask):
    """Baseline forward for ``cie_replace``: last-position logits plus the
    per-layer last-token z slices needed by ``cie_patched_last_logits``.
    Returns ``(last_logits, z_base)`` with ``z_base[layer_idx]`` of shape
    ``[batch, n_heads, d_head]``."""
    z_base: dict[int, object] = {}
    hooks = _capture_last_z_hooks(int(model.cfg.n_layers), z_base)
    last_logits = _forward_last_logits(model, tokens, mask, fwd_hooks=hooks)
    return last_logits, z_base


def cie_patched_last_logits(
    model,
    tokens,
    mask,
    layer_idx: int,
    head_idx: int,
    z_vector,
    z_base: dict,
    contrib_cache: dict,
):
    """One patched forward for ``cie_replace`` v2.

    A single hook on ``blocks.{layer_idx}.hook_attn_out`` adds, at the last
    token,

        delta = z_vector − z_base[layer_idx][:, head_idx, :] @ W_O[layer_idx, head_idx]

    which REPLACES head (layer_idx, head_idx)'s last-token post-W_O
    contribution with ``z_vector``: algebraically identical to setting
    ``act[:, -1, head_idx, :] = z_vector`` on ``attn.hook_result`` under
    ``use_attn_result=True`` — ``b_O`` and the other heads' terms cancel,
    and within the patched forward the layer's OWN ``hook_z`` equals the
    baseline because the patch applies after z is computed (both properties
    pinned by ``tests/test_aie_cie_equivalence.py``). Unlike the
    ``use_attn_result`` formulation it never materializes the per-layer
    ``[batch, pos, head, d_head, d_model]`` intermediate (≈18–190 GiB/layer
    on the real cells — OOM). Only bf16 association-order differences
    remain. ``contrib_cache`` memoizes the per-layer baseline head
    contributions ``[batch, n_heads, d_model]`` across heads of one batch.
    """
    import torch  # heavy

    if layer_idx not in contrib_cache:
        contrib_cache[layer_idx] = torch.einsum(
            "bhd,hdm->bhm", z_base[layer_idx], model.W_O[layer_idx]
        )
    delta = z_vector - contrib_cache[layer_idx][:, head_idx, :]
    hook_name = f"blocks.{layer_idx}.hook_attn_out"

    def add_delta(activation, hook, *, _delta=delta):
        activation[:, -1, :] += _delta
        return activation

    return _forward_last_logits(model, tokens, mask, fwd_hooks=[(hook_name, add_delta)])


def _proxy_patched_last_logits(model, tokens, mask, hook_name: str, vector):
    """One patched forward for ``zs_add_proxy``: ADD the head's z̄ at the
    inject site's last token."""

    def add_at_last(activation, hook, *, _vector=vector):
        activation[:, -1, :] += _vector
        return activation

    return _forward_last_logits(
        model, tokens, mask, fwd_hooks=[(hook_name, add_at_last)]
    )


def compute_method_scores(
    model_loader, method: str, samples_manifest: dict, z_results: dict, cfg: Step1Config
) -> dict:
    """GPU: per-task CIE matrices + the AIE matrix for one method.

    ``cie_replace`` scores the manifest's corrupted N-SHOT prompts with
    replace-at-own-head patches (delta at ``hook_attn_out``, v2 — peak
    memory ≈ plain inference); ``zs_add_proxy`` scores the manifest's
    ZERO-SHOT prompts with add-at-inject-site patches. Unscored heads (under
    ``aie.head_limit``) stay NaN in the returned float32 matrices.

    Returns ``{"per_task_cie", "aie", "per_task_base", "task_order",
    "heads", "grid"}`` (numpy payloads; ``per_task_base`` holds per-prompt
    baseline first-token probabilities).
    """
    import numpy as np

    from subspaces.step1.eval_gpu import manifest_prompts

    if method not in METHOD_VERSIONS:
        raise ConfigError(
            f"unknown AIE method {method!r}; registered: {sorted(METHOD_VERSIONS)}"
        )
    model = model_loader()
    task_order = list(samples_manifest["task_order"])
    missing = [task for task in task_order if task not in z_results]
    if missing:
        raise ArtifactError(f"activation cache lacks task z for: {missing}")
    shapes = {tuple(z_results[task].shape) for task in task_order}
    if len(shapes) != 1:
        raise ArtifactError(f"inconsistent z shapes across tasks: {sorted(shapes)}")
    n_layers, n_heads, _d_model = shapes.pop()
    model_dims = (int(model.cfg.n_layers), int(model.cfg.n_heads))
    if (n_layers, n_heads) != model_dims:
        raise ArtifactError(
            f"z-cache head grid {n_layers}x{n_heads} does not match the model "
            f"grid {model_dims[0]}x{model_dims[1]}; refusing"
        )
    heads = head_grid(n_layers, n_heads, cfg.aie.head_limit)
    zero_shot = method == "zs_add_proxy"
    batch_size = (
        cfg.aie.batch_size if method == "cie_replace" else cfg.aie.proxy_batch_size
    )

    # startup pre-validation (CPU tokenizer pass over EVERY task's targets):
    # a target whose first-token readout is undefined must refuse BEFORE any
    # GPU hours are burned (insurance for reruns/foreign datasets).
    for task in task_order:
        _prompts, targets = manifest_prompts(
            samples_manifest, task, zero_shot=zero_shot
        )
        _target_first_token_ids(model, targets)

    per_task_cie: dict[str, np.ndarray] = {}
    per_task_base: dict[str, np.ndarray] = {}
    for task in task_order:
        prompts, targets = manifest_prompts(samples_manifest, task, zero_shot=zero_shot)
        z_task = z_results[task].to(device=model.cfg.device, dtype=model.cfg.dtype)
        delta_sums = [0.0] * len(heads)
        base_probs: list[float] = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            batch_targets = targets[start : start + batch_size]
            tokens = model.to_tokens(
                batch_prompts, prepend_bos=True, padding_side="left"
            )
            mask = tokens != model.tokenizer.pad_token_id
            target_ids = _target_first_token_ids(model, batch_targets)
            if method == "cie_replace":
                baseline_logits, z_base = cie_baseline_forward(model, tokens, mask)
                contrib_cache: dict = {}
            else:
                baseline_logits = _forward_last_logits(model, tokens, mask)
            base_probs.extend(
                first_token_probs(baseline_logits, target_ids).cpu().tolist()
            )
            for head_pos, (layer_idx, head_idx) in enumerate(heads):
                if method == "cie_replace":
                    patched_logits = cie_patched_last_logits(
                        model,
                        tokens,
                        mask,
                        layer_idx,
                        head_idx,
                        z_task[layer_idx, head_idx],
                        z_base,
                        contrib_cache,
                    )
                else:
                    patched_logits = _proxy_patched_last_logits(
                        model,
                        tokens,
                        mask,
                        cfg.sites.inject_layer,
                        z_task[layer_idx, head_idx],
                    )
                deltas = first_token_prob_deltas(
                    baseline_logits, patched_logits, target_ids
                )
                delta_sums[head_pos] += float(deltas.sum().item())
        n_prompts = len(prompts)
        if n_prompts != len(base_probs):
            raise ArtifactError(
                f"scored {len(base_probs)} of {n_prompts} prompts for "
                f"{task} — a prompt was dropped; refusing."
            )
        cie = np.full((n_layers, n_heads), np.nan, dtype=np.float32)
        for head_pos, (layer_idx, head_idx) in enumerate(heads):
            cie[layer_idx, head_idx] = delta_sums[head_pos] / n_prompts
        per_task_cie[task] = cie
        per_task_base[task] = np.asarray(base_probs, dtype=np.float32)
    aie = np.mean(np.stack([per_task_cie[task] for task in task_order]), axis=0).astype(
        np.float32
    )
    return {
        "per_task_cie": per_task_cie,
        "aie": aie,
        "per_task_base": per_task_base,
        "task_order": task_order,
        "heads": heads,
        "grid": {"n_layers": int(n_layers), "n_heads": int(n_heads)},
    }


# -- orchestration -------------------------------------------------------------


def resolve_overrides(
    cfg: Step1Config, methods: list[str] | None, k_values: list[int] | None
) -> tuple[list[str], list[int]]:
    """CLI overrides may NARROW the configured lists; anything not in the
    config REFUSES (repo convention: explicit contradiction refuses)."""
    if methods is None:
        methods = list(cfg.aie.methods)
    else:
        if not methods:
            raise ConfigError("--methods must name at least one method")
        if len(set(methods)) != len(methods):
            raise ConfigError(f"--methods contains duplicates: {methods}")
        rogue = [method for method in methods if method not in cfg.aie.methods]
        if rogue:
            raise ConfigError(
                f"--methods {rogue} contradicts the config's aie.methods "
                f"{cfg.aie.methods}; refusing (narrowing is allowed, adding "
                "is not)"
            )
    if k_values is None:
        k_values = list(cfg.aie.k_values)
    else:
        if not k_values:
            raise ConfigError("--k must name at least one k value")
        if len(set(k_values)) != len(k_values):
            raise ConfigError(f"--k contains duplicates: {k_values}")
        rogue_k = [k for k in k_values if k not in cfg.aie.k_values]
        if rogue_k:
            raise ConfigError(
                f"--k {rogue_k} contradicts the config's aie.k_values "
                f"{cfg.aie.k_values}; refusing (narrowing is allowed, adding "
                "is not)"
            )
    return methods, k_values


def _write_scores(
    cfg: Step1Config,
    paths: ProjectPaths,
    expected: dict,
    result: dict,
    aie_dir: Path,
    stem: str,
) -> dict:
    import numpy as np

    from subspaces.step1.eval_gpu import gpu_peak_memory

    arrays: dict = {"aie": result["aie"]}
    for task in result["task_order"]:
        arrays[f"cie__{task}"] = result["per_task_cie"][task]
        arrays[f"pbase__{task}"] = result["per_task_base"][task]
    aie_dir.mkdir(parents=True, exist_ok=True)
    npz_path = aie_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, **arrays)
    preview = select_top_k(
        result["aie"], result["heads"], min(10, len(result["heads"]))
    )
    manifest = make_manifest(
        kind="aie_scores",
        schema_version=AIE_SCORES_SCHEMA_VERSION,
        paths=paths,
        config=expected["config"],
        inputs=expected["inputs"],
        payload={
            "impl": expected["impl"],
            "protocol": cfg.protocol,
            "task_order": result["task_order"],
            "grid": result["grid"],
            "head_order": HEAD_ORDER,
            "n_scored_heads": len(result["heads"]),
            "head_limit_applied": cfg.aie.head_limit,
            "n_prompts_per_task": {
                task: int(result["per_task_base"][task].shape[0])
                for task in result["task_order"]
            },
            "baseline_mean_prob": {
                task: float(result["per_task_base"][task].mean())
                for task in result["task_order"]
            },
            "top_heads_preview": [
                [layer_idx, head_idx, score] for layer_idx, head_idx, score in preview
            ],
            "gpu_memory": gpu_peak_memory(),
            "arrays": {
                "file": npz_path.name,
                "content_sha256": arrays_content_sha(arrays),
                "keys": sorted(arrays),
            },
        },
    )
    write_json_atomic(aie_dir / f"{stem}.json", manifest)
    return manifest


def _load_aie_matrix(aie_dir: Path, scores_manifest: dict):
    """AIE matrix + the scored-head list from a persisted scores artifact
    (npz digest re-verified on every load)."""
    import numpy as np

    _verify_scores_arrays(aie_dir, scores_manifest)
    loaded = np.load(Path(aie_dir) / scores_manifest["arrays"]["file"])
    grid = scores_manifest["grid"]
    scored_heads = head_grid(
        int(grid["n_layers"]),
        int(grid["n_heads"]),
        scores_manifest.get("head_limit_applied"),
    )
    return loaded["aie"], scored_heads


def _evaluate_top_k(
    cfg: Step1Config,
    paths: ProjectPaths,
    expected: dict,
    top_heads: list[tuple[int, int, float]],
    z_heldout: dict,
    final_samples: dict,
    aie_dir: Path,
    stem: str,
    *,
    resolved: dict,
    model_loader,
) -> dict:
    """Todd-FV evaluation of one (method, k): paired clean (n-shot, no hook)
    and aie_unit (zero-shot + additive FV at the inject site) arms on the
    final-evaluation prompts — the headset_eval pattern."""
    import numpy as np

    from subspaces.step1.eval_gpu import (
        effective_batch_size,
        eval_prompts,
        gpu_peak_memory,
        manifest_prompts,
        require_resolved_model,
    )
    from subspaces.step1.recovery import outcomes_content_sha

    require_resolved_model(resolved)
    task_order = list(final_samples["task_order"])
    missing = [task for task in task_order if task not in z_heldout]
    if missing:
        raise ArtifactError(f"held-out activation cache lacks z for: {missing}")

    model = model_loader()
    batch_size = effective_batch_size(cfg)
    layer_name = cfg.sites.inject_layer
    outcomes: dict[str, list[int]] = {"clean": [], "aie_unit": []}
    per_task: dict[str, dict] = {}
    all_batch_sizes: list[int] = []
    for task in task_order:
        nshot = manifest_prompts(final_samples, task)
        zeroshot = manifest_prompts(final_samples, task, zero_shot=True)
        vector = sum(
            z_heldout[task][layer_idx, head_idx]
            for layer_idx, head_idx, _score in top_heads
        )
        task_metrics: dict = {}
        for variant, (prompts, targets), arm_vector in (
            ("clean", nshot, None),
            ("aie_unit", zeroshot, vector),
        ):
            correct, batch_sizes = eval_prompts(
                model,
                prompts,
                targets,
                layer_name=layer_name if arm_vector is not None else None,
                vector=arm_vector,
                batch_size=batch_size,
            )
            outcomes[variant].extend(correct)
            all_batch_sizes.extend(batch_sizes)
            task_metrics[variant] = float(np.mean(correct))
        task_metrics["n"] = len(nshot[0])
        per_task[task] = task_metrics

    aie_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = aie_dir / f"{stem}-outcomes.npz"
    np.savez_compressed(
        outcomes_path,
        **{key: np.asarray(values, dtype=np.uint8) for key, values in outcomes.items()},
    )
    metrics = {variant: float(np.mean(values)) for variant, values in outcomes.items()}
    metrics_macro = {
        variant: float(np.mean([per_task[task][variant] for task in task_order]))
        for variant in outcomes
    }
    manifest = make_manifest(
        kind="aie_eval",
        schema_version=AIE_EVAL_SCHEMA_VERSION,
        paths=paths,
        config=expected["config"],
        inputs=expected["inputs"],
        payload={
            "impl": expected["impl"],
            "protocol": cfg.protocol,
            "heads": [
                [layer_idx, head_idx, score] for layer_idx, head_idx, score in top_heads
            ],
            "metrics": metrics,
            "metrics_macro": metrics_macro,
            "per_task": per_task,
            "task_order": task_order,
            "n_eval_total": sum(data["n"] for data in per_task.values()),
            "effective_batch_sizes": sorted(set(all_batch_sizes)),
            "gpu_memory": gpu_peak_memory(),
            "outcomes": {
                "file": outcomes_path.name,
                "content_sha256": outcomes_content_sha(outcomes),
            },
        },
    )
    write_json_atomic(aie_dir / f"{stem}.json", manifest)
    return manifest


def run_aie(
    cfg: Step1Config,
    paths: ProjectPaths,
    *,
    journal_dir: Path,
    split,
    resolved: dict,
    methods: list[str] | None = None,
    k_values: list[int] | None = None,
) -> dict:
    """The whole AIE chain for one cell, resumable via reuse_or_refuse:
    ensure samples → ensure z caches → aie_scores per method → top-k per k →
    aie_eval per (method × k). Scores/selection consume TRAIN-task kinds
    only; held-out manifests/caches are materialized by the eval part only.
    """
    if cfg.aie is None:
        raise ConfigError(
            "this config has no aie block; the aie stage requires one "
            "(see AIEConfig in subspaces/config.py)"
        )
    methods, k_values = resolve_overrides(cfg, methods, k_values)

    from subspaces.step1 import pipeline
    from subspaces.step1.zcache import cache_fingerprint, cache_identity, ensure_zcache

    aie_dir = Path(journal_dir) / "aie"
    model_loader = pipeline.make_model_loader(cfg, resolved)

    # -- selection-side inputs (train tasks only; holdout guard in samples) --
    activation_manifest, activation_path = pipeline._ensure_samples(
        cfg, cfg.samples.activation, "activation", split, paths
    )
    corrupted: tuple[dict, Path] | None = None
    if "cie_replace" in methods:
        corrupted = pipeline._ensure_samples(
            cfg, cfg.samples.aie_corrupted, "aie_corrupted", split, paths
        )
    z_train_fp = cache_fingerprint(cache_identity(cfg, resolved, activation_manifest))

    # -- per-method scores --
    scores: dict[str, tuple[dict, Path]] = {}
    for method in methods:
        if method == "cie_replace":
            assert corrupted is not None
            samples_manifest, samples_path = corrupted
            examples_per_task = cfg.samples.aie_corrupted.examples_per_task
        else:
            samples_manifest, samples_path = activation_manifest, activation_path
            examples_per_task = cfg.samples.activation.examples_per_task
        expected = scores_identity(
            cfg,
            method,
            manifest_ref(samples_path, paths, samples_manifest),
            z_train_fp,
            examples_per_task,
        )
        stem = f"scores-{method}-{identity_of(expected)[:8]}"
        out_path = aie_dir / f"{stem}.json"
        existing = reuse_or_refuse(
            out_path,
            expected,
            expect_kind="aie_scores",
            max_schema_version=AIE_SCORES_SCHEMA_VERSION,
        )
        if existing is not None:
            _verify_scores_arrays(aie_dir, existing)
            scores[method] = (existing, out_path)
            continue
        z_results, fingerprint = ensure_zcache(
            model_loader, activation_manifest, cfg, paths, resolved
        )
        if fingerprint != z_train_fp:
            raise ArtifactError(
                "train z-cache fingerprint drifted during the run; refusing"
            )
        result = compute_method_scores(
            model_loader, method, samples_manifest, z_results, cfg
        )
        manifest = _write_scores(cfg, paths, expected, result, aie_dir, stem)
        scores[method] = (manifest, out_path)

    # -- eval: the ONLY part that materializes held-out manifests/caches --
    final_samples, final_path = pipeline._ensure_samples(
        cfg, cfg.samples.final_eval, "final_eval", split, paths
    )
    heldout_manifest, _heldout_path = pipeline._ensure_samples(
        cfg, cfg.samples.heldout_activation, "heldout_activation", split, paths
    )
    z_heldout_fp = cache_fingerprint(cache_identity(cfg, resolved, heldout_manifest))
    final_ref = manifest_ref(final_path, paths, final_samples)

    z_heldout: dict | None = None
    evals: dict[str, tuple[dict, Path]] = {}
    for method in methods:
        scores_manifest, scores_path = scores[method]
        scores_ref = manifest_ref(scores_path, paths, scores_manifest)
        for k in k_values:
            expected = eval_identity(
                cfg, method, k, scores_ref, final_ref, z_heldout_fp
            )
            stem = f"eval-{method}-k{k}-{identity_of(expected)[:8]}"
            out_path = aie_dir / f"{stem}.json"
            existing = reuse_or_refuse(
                out_path,
                expected,
                expect_kind="aie_eval",
                max_schema_version=AIE_EVAL_SCHEMA_VERSION,
            )
            if existing is not None:
                pipeline._verify_outcomes(aie_dir, existing)
                evals[f"{method}-k{k}"] = (existing, out_path)
                continue
            aie_matrix, scored_heads = _load_aie_matrix(aie_dir, scores_manifest)
            top_heads = select_top_k(aie_matrix, scored_heads, k)
            if z_heldout is None:
                z_heldout, fingerprint = ensure_zcache(
                    model_loader, heldout_manifest, cfg, paths, resolved
                )
                if fingerprint != z_heldout_fp:
                    raise ArtifactError(
                        "held-out z-cache fingerprint drifted during the run; "
                        "refusing"
                    )
            manifest = _evaluate_top_k(
                cfg,
                paths,
                expected,
                top_heads,
                z_heldout,
                final_samples,
                aie_dir,
                stem,
                resolved=resolved,
                model_loader=model_loader,
            )
            evals[f"{method}-k{k}"] = (manifest, out_path)

    # -- journal bookkeeping (accumulates across invocations, like nodes.json)
    summary_path = Path(journal_dir) / "aie_nodes.json"
    recorded_scores: dict = {}
    recorded_evals: dict = {}
    if summary_path.exists():
        import json

        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        recorded_scores = previous.get("scores", {})
        recorded_evals = previous.get("evals", {})
    for method, (_manifest, path) in scores.items():
        recorded_scores[method] = paths.relativize(path)
    for name, (manifest, path) in evals.items():
        recorded_evals[name] = {
            "path": paths.relativize(path),
            "metrics": manifest["metrics"],
            "metrics_macro": manifest["metrics_macro"],
            "heads": manifest["heads"][:10],
        }
    summary = make_manifest(
        kind="aie_run",
        schema_version=1,
        paths=paths,
        config={"protocol": cfg.protocol},
        payload={
            "run_id": Path(journal_dir).name,
            "scores": recorded_scores,
            "evals": recorded_evals,
        },
    )
    write_json_atomic(summary_path, summary)
    return summary
