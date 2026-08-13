"""cie_replace v2 equivalence: the production delta-at-hook_attn_out patch
(`cie_patched_last_logits`) must match the paper's replace-at-
``attn.hook_result`` formulation (``use_attn_result=True``), which v2 avoids
because that flag materializes a per-layer
``[batch, pos, head, d_head, d_model]`` intermediate (OOM on the real
long-sequence cells). The reference implementation lives INSIDE this test
only. Also pins the algebraic assumption: within the patched forward the
patched layer's own ``hook_z`` equals the baseline."""

from __future__ import annotations

import pytest
import torch

from subspaces.step1.aie import cie_baseline_forward, cie_patched_last_logits


@pytest.fixture(scope="module")
def tiny_model():
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    torch.manual_seed(0)
    config = HookedTransformerConfig(
        n_layers=3,
        d_model=16,
        n_ctx=32,
        d_head=4,
        n_heads=4,
        d_vocab=53,
        act_fn="relu",
        seed=0,
    )
    model = HookedTransformer(config)
    model.eval()
    return model


def _tokens_and_mask(model):
    generator = torch.Generator().manual_seed(1)
    tokens = torch.randint(
        0, model.cfg.d_vocab, (2, 9), generator=generator, dtype=torch.long
    )
    return tokens, torch.ones_like(tokens, dtype=torch.bool)


def _reference_replace_last_logits(model, tokens, mask, layer_idx, head_idx, z_vector):
    """Paper formulation: replace ``act[:, -1, head_idx, :]`` on
    ``attn.hook_result`` under ``use_attn_result=True`` (test-only)."""
    model.set_use_attn_result(True)
    try:

        def replace(activation, hook):
            activation[:, -1, head_idx, :] = z_vector
            return activation

        with torch.no_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(f"blocks.{layer_idx}.attn.hook_result", replace)],
                attention_mask=mask,
                return_type="logits",
            )
    finally:
        model.set_use_attn_result(False)
    return logits[:, -1, :].detach().clone()


def test_v2_delta_matches_reference_replace_across_layers_and_heads(tiny_model):
    model = tiny_model
    tokens, mask = _tokens_and_mask(model)
    _baseline_logits, z_base = cie_baseline_forward(model, tokens, mask)
    contrib_cache: dict = {}
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    cases = [
        (0, n_heads - 1),  # first layer, last head
        (1, 2),  # middle
        (n_layers - 1, 0),  # last layer, head 0
    ]
    z_generator = torch.Generator().manual_seed(2)
    for layer_idx, head_idx in cases:
        z_vector = torch.randn(int(model.cfg.d_model), generator=z_generator)
        new_logits = cie_patched_last_logits(
            model, tokens, mask, layer_idx, head_idx, z_vector, z_base, contrib_cache
        )
        reference = _reference_replace_last_logits(
            model, tokens, mask, layer_idx, head_idx, z_vector
        )
        torch.testing.assert_close(
            new_logits.to(torch.float32),
            reference.to(torch.float32),
            rtol=1e-4,
            atol=1e-4,
        )


def test_patched_forward_own_layer_z_equals_baseline(tiny_model):
    """The delta formulation assumes the patched layer's own z is unchanged
    (the patch applies at hook_attn_out, AFTER z is computed)."""
    model = tiny_model
    tokens, mask = _tokens_and_mask(model)
    _baseline_logits, z_base = cie_baseline_forward(model, tokens, mask)
    layer_idx, head_idx = 1, 3
    z_vector = torch.randn(
        int(model.cfg.d_model), generator=torch.Generator().manual_seed(3)
    )
    sink: dict = {}

    def capture(activation, hook):
        sink["z"] = activation[:, -1, :, :].detach().clone()
        return activation

    model.add_hook(f"blocks.{layer_idx}.attn.hook_z", capture)
    try:
        cie_patched_last_logits(
            model, tokens, mask, layer_idx, head_idx, z_vector, z_base, {}
        )
    finally:
        model.reset_hooks()
    torch.testing.assert_close(sink["z"], z_base[layer_idx], rtol=0.0, atol=0.0)


def test_baseline_forward_logits_match_plain_forward(tiny_model):
    """The z-capture hooks are read-only: the baseline forward's logits must
    equal a plain forward's."""
    from subspaces.step1.aie import _forward_last_logits

    model = tiny_model
    tokens, mask = _tokens_and_mask(model)
    baseline_logits, z_base = cie_baseline_forward(model, tokens, mask)
    plain_logits = _forward_last_logits(model, tokens, mask)
    torch.testing.assert_close(baseline_logits, plain_logits, rtol=0.0, atol=0.0)
    assert sorted(z_base) == list(range(int(model.cfg.n_layers)))
    for z_slice in z_base.values():
        assert z_slice.shape == (
            tokens.shape[0],
            int(model.cfg.n_heads),
            int(model.cfg.d_head),
        )
