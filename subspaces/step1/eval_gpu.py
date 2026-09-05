"""GPU evaluation primitives: LEGACY numerics driven by manifest prompts.

Design rule (same as the selected substep): the numerical model code is
REUSED from the legacy package (``subspaces.utils.intervene`` /
``subspaces.utils.model``) — zero drift by construction — while everything around
it is new: prompts come exclusively from sample manifests (never internal
sampling), every listed example is evaluated (no ``drop_last`` — the final
partial batch runs at its natural size), and effective batch sizes are
recorded.

Heavy imports are confined to call time; importing this module stays light.
"""

from __future__ import annotations

from subspaces.artifacts import ArtifactError
from subspaces.config import Step1Config

IMPL = {"module": "subspaces.step1.eval_gpu", "algorithm_version": 2}

# Default evaluation batch size when compute.batch_size is unset; part of the
# scan/headset ARTIFACT identity (bf16 batching can flip rare borderline
# generations, so a batch-size change must not silently reuse results).
DEFAULT_BATCH_SIZE = 20


def effective_batch_size(cfg: Step1Config) -> int:
    return cfg.compute.batch_size or DEFAULT_BATCH_SIZE


# Legacy FV injection ADDS the vector to the residual at the injection site
# (every legacy call site passes intervention_mode=0 — the function's mode-1
# REPLACE default is never used); interventions run on the ZERO-SHOT prompt
# while clean accuracy runs on the n-shot prompt.
INTERVENTION_MODE = 0


def gpu_peak_memory() -> dict | None:
    """Process-lifetime peak CUDA memory (``None`` off-GPU).

    Observability only: recorded in GPU-substep artifact payloads OUTSIDE
    the identity keys AND listed in ``VOLATILE_KEYS``, so it participates in
    neither reuse decisions nor semantic fingerprints/lineage (the value is
    device- and allocation-history-dependent, hence non-reproducible). The
    peak is cumulative for the process (never reset), so the value at any
    artifact's creation bounds everything that ran before it — model
    weights, z-cache extraction, and all prior evaluations included.
    """
    import torch  # heavy

    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    return {
        "device_name": torch.cuda.get_device_name(device),
        "max_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
        "max_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 3),
        "total_gib": round(
            torch.cuda.get_device_properties(device).total_memory / 2**30, 3
        ),
    }


def require_resolved_model(resolved: dict) -> None:
    """GPU substeps refuse to run unless the model weights are resolvable."""
    model_validation = resolved["validation"]["model"]
    if model_validation["status"] != "resolved":
        raise ArtifactError(
            "model identity is unresolved: the local HF cache does not "
            "contain the model, so weights cannot load (loading is "
            "offline-only). Set HF_HOME to the model cache — a pinned "
            "model.revision keeps the run identity stable but cannot supply "
            "weights."
        )


def load_model(cfg: Step1Config):
    """Load the model via the legacy loader (offline-only, GPU/CPU per cfg).

    When a revision is pinned, the cache's resolvable revision is re-verified
    at LOAD time (not only at run resolution) so cache drift between the two
    moments cannot silently load different weights.
    """
    import torch  # heavy

    from subspaces.step1.resolve import resolve_model_identity
    from subspaces.utils.model import load_model_no_grad  # heavy

    if cfg.model.revision is not None:
        observed = resolve_model_identity(cfg.model.name)
        if observed["revision"] != cfg.model.revision:
            raise ArtifactError(
                f"the local cache resolves {cfg.model.name} to revision "
                f"{observed['revision']} but the run pins "
                f"{cfg.model.revision}; refusing to load drifted weights."
            )
    dtype = getattr(torch, cfg.model.dtype)
    model, _n_layers, _n_heads = load_model_no_grad(
        cfg.model.name, cfg.model.device, dtype=dtype
    )
    return model


def manifest_prompts(
    samples_manifest: dict, task_id: str, *, zero_shot: bool = False
) -> tuple[list, list]:
    key = "zero_shot_prompt" if zero_shot else "prompt"
    samples = samples_manifest["tasks"][task_id]["samples"]
    return (
        [sample[key] for sample in samples],
        [sample["expected"] for sample in samples],
    )


def eval_prompts(
    model,
    prompts: list[str],
    targets: list[str],
    *,
    layer_name: str | None = None,
    vector=None,
    batch_size: int,
) -> tuple[list[int], list[int]]:
    """Per-example 0/1 correctness over ALL prompts (final partial batch
    included). Returns (correct_list, effective_batch_sizes)."""
    from subspaces.utils.intervene import intervened_generation_with_accuracy  # heavy

    if len(prompts) != len(targets):
        raise ValueError("prompts/targets length mismatch")
    if vector is not None:
        # mode-0 ADD requires the vector on the model device AND dtype (legacy
        # recovery_scan moved z to device; in-place += also rejects dtype
        # promotion into a bf16 activation)
        vector = vector.to(device=model.cfg.device, dtype=model.cfg.dtype)
    correct: list[int] = []
    batch_sizes: list[int] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_targets = targets[start : start + batch_size]
        batch_sizes.append(len(batch_prompts))
        _acc, _generated, correct_flags = intervened_generation_with_accuracy(
            model,
            batch_prompts,
            batch_targets,
            layer_name=layer_name,
            intervention_vector=vector,
            intervention_mode=INTERVENTION_MODE,
            print_correct_list=True,
        )
        correct.extend(int(bool(flag)) for flag in correct_flags)
    if len(correct) != len(prompts):
        raise ArtifactError(
            f"evaluated {len(correct)} of {len(prompts)} prompts — an example "
            "was dropped; refusing."
        )
    return correct, batch_sizes


def compute_z_for_tasks(model, samples_manifest: dict, task_ids: list[str]) -> dict:
    """Mean last-token per-head attention output per task, averaged over the
    MANIFEST prompts (legacy hook mechanics, chunk size 1)."""
    import torch  # heavy

    z_results: dict = {}
    for task_id in task_ids:
        prompts, _targets = manifest_prompts(samples_manifest, task_id)
        chunk_activations = []
        for prompt in prompts:
            tokens = model.to_tokens(prompt, prepend_bos=True, padding_side="left")
            layer_activations: list = []

            def head_activation_hook(activation, hook, _sink=layer_activations):
                _sink.append(torch.mean(activation[:, -1, :, :], dim=0))

            with torch.no_grad():
                model.cfg.use_attn_result = True
                model.run_with_hooks(
                    tokens,
                    return_type=None,
                    fwd_hooks=[
                        (
                            lambda name: name.endswith("attn.hook_result"),
                            head_activation_hook,
                        )
                    ],
                    attention_mask=(tokens != model.tokenizer.pad_token_id),
                )
                model.cfg.use_attn_result = False
            chunk_activations.append(torch.stack(layer_activations, dim=0))
        z_results[task_id] = torch.stack(chunk_activations, dim=0).mean(dim=0).cpu()
    return z_results


def mean_z_over_tasks(z_results: dict):
    import torch  # heavy

    return torch.stack(list(z_results.values()), dim=0).mean(dim=0)
