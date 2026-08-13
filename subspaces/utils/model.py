"""HookedTransformer loading and CUDA-memory diagnostics."""

from __future__ import annotations

import os

import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


def print_cuda_memory(timepoint=""):
    print(timepoint)
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached:    {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    else:
        print("CUDA is not available")


def load_model_no_grad(
    model_name,
    device,
    cache_dir=None,
    token=None,
    dtype=torch.bfloat16,
    local_files_only=True,
):
    """Load an HF model + HookedTransformer wrapper, eval-mode, no grads.

    When ``cache_dir`` is None it is resolved at call time from ``$HF_HOME``;
    if that is also unset, HuggingFace's default cache location is used. The
    HF token falls back to HF_TOKEN / HUGGING_FACE_HUB_TOKEN; fully-offline
    loads need none.
    """
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME") or None
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    common_kwargs = dict(
        cache_dir=cache_dir,
        token=token,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        local_files_only=True,
        trust_remote_code=True,
        force_download=False,
    )

    hf_model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)

    # Load the tokenizer from its resolved LOCAL snapshot dir (not the repo id)
    # when running offline. TransformerLens's set_tokenizer() re-loads the
    # tokenizer via get_tokenizer_with_bos(), which calls AutoTokenizer with the
    # tokenizer's ``name_or_path`` but WITHOUT cache_dir — so a repo-id
    # name_or_path triggers a hub lookup in the default HF_HUB_CACHE
    # (``HF_HOME/hub``) and raises offline when the model was cached under a
    # custom ``cache_dir`` (files at ``<cache_dir>/models--…`` instead). Pointing
    # name_or_path at the local snapshot dir makes that re-load read files
    # directly. Falls back to the repo id if the snapshot can't be resolved.
    tokenizer_src = model_name
    if local_files_only:
        try:
            from huggingface_hub import snapshot_download

            tokenizer_src = snapshot_download(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True,
                allow_patterns=["*.json", "*.model", "*.txt", "tokenizer*"],
            )
        except Exception:
            tokenizer_src = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, **common_kwargs)

    model = HookedTransformer.from_pretrained(
        model_name=model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        cache_dir=cache_dir,
    ).to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, model.cfg.n_layers, model.cfg.n_heads
