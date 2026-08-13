"""CPU tests for subspaces.step1.icl_ablation (hook slicing + subset sampling)."""

from __future__ import annotations

import pytest
import torch

from subspaces.step1.icl_ablation import (
    ablated_generation_correctness,
    head_ablation_hooks,
    result_hook_name,
    sample_head_subsets,
)


def _apply(hooks, name, activation):
    for hook_name, fn in hooks:
        if hook_name == name:
            activation = fn(activation, None)
    return activation


def test_hooks_replace_only_listed_heads_at_position():
    torch.manual_seed(0)
    activation = torch.randn(2, 7, 4, 3)  # batch, seq, heads, d_model
    original = activation.clone()
    values = {(15, 1): torch.ones(3), (15, 3): torch.full((3,), 2.0)}
    hooks = head_ablation_hooks(values, position=-1)
    assert [name for name, _ in hooks] == ["blocks.15.attn.hook_result"]

    out = _apply(hooks, "blocks.15.attn.hook_result", activation)
    assert torch.equal(out[:, -1, 1, :], torch.ones(2, 3))
    assert torch.equal(out[:, -1, 3, :], torch.full((2, 3), 2.0))
    # untouched heads and positions
    assert torch.equal(out[:, -1, 0, :], original[:, -1, 0, :])
    assert torch.equal(out[:, -1, 2, :], original[:, -1, 2, :])
    assert torch.equal(out[:, :-1, :, :], original[:, :-1, :, :])


def test_hooks_group_by_layer_and_track_position():
    # distinct value per (layer, head): a late-binding closure bug (every hook
    # capturing the LAST layer's entries) or head/value mis-assignment fails
    # these per-layer isolation assertions (review finding, 2026-08-09)
    values = {
        (13, 6): torch.full((3,), 5.0),
        (15, 2): torch.full((3,), 2.0),
        (15, 1): torch.full((3,), 1.0),
    }
    hooks = head_ablation_hooks(values, position=-2)
    assert [name for name, _ in hooks] == [
        "blocks.13.attn.hook_result",
        "blocks.15.attn.hook_result",
    ]

    act13 = torch.randn(1, 5, 8, 3)
    orig13 = act13.clone()
    out13 = _apply(hooks, "blocks.13.attn.hook_result", act13)
    assert torch.equal(out13[:, -2, 6, :], torch.full((1, 3), 5.0))
    # layer 13 must NOT receive layer 15's heads
    assert torch.equal(out13[:, -2, 1, :], orig13[:, -2, 1, :])
    assert torch.equal(out13[:, -2, 2, :], orig13[:, -2, 2, :])
    assert torch.equal(out13[:, -1, :, :], orig13[:, -1, :, :])  # final pos untouched

    act15 = torch.randn(1, 5, 8, 3)
    orig15 = act15.clone()
    out15 = _apply(hooks, "blocks.15.attn.hook_result", act15)
    assert torch.equal(out15[:, -2, 1, :], torch.full((1, 3), 1.0))
    assert torch.equal(out15[:, -2, 2, :], torch.full((1, 3), 2.0))
    # layer 15 must NOT receive layer 13's head
    assert torch.equal(out15[:, -2, 6, :], orig15[:, -2, 6, :])
    assert torch.equal(out15[:, -1, :, :], orig15[:, -1, :, :])


def test_result_hook_name():
    assert result_hook_name(15) == "blocks.15.attn.hook_result"


class _StubModel:
    """Minimal HookedTransformer stand-in for the greedy decode loop.

    Tokenizes one character per token (BOS=1, pad=0), records for every
    forward the sequence length, the absolute positions the hooks wrote, and
    the ``use_attn_result`` flag, and emits scripted greedy tokens per batch
    row and decode step.
    """

    BOS = 1
    N_HEADS = 4
    D_MODEL = 3
    VOCAB = 1024

    def __init__(self, scripted_rows, raise_on_forward=None):
        from types import SimpleNamespace

        self.cfg = SimpleNamespace(
            device="cpu", dtype=torch.float32, use_attn_result=False
        )
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.scripted_rows = scripted_rows  # global row -> [token per step]
        self.raise_on_forward = raise_on_forward
        self.calls = []
        self._row_offset = 0
        self._batch_rows = 0
        self._step = 0
        self._last_seq = None

    def to_tokens(self, texts, prepend_bos=False, padding_side="right"):
        seqs = [
            ([self.BOS] if prepend_bos else []) + [ord(c) for c in text]
            for text in texts
        ]
        width = max(len(s) for s in seqs)
        rows = []
        for s in seqs:
            pad = [0] * (width - len(s))
            rows.append(pad + s if padding_side == "left" else s + pad)
        return torch.tensor(rows, dtype=torch.long)

    def run_with_hooks(self, tokens, fwd_hooks=None, attention_mask=None, **_kwargs):
        if (
            self.raise_on_forward is not None
            and len(self.calls) == self.raise_on_forward
        ):
            raise RuntimeError("scripted forward failure")
        batch, seq = tokens.shape
        # a new batch starts whenever the sequence did not grow by exactly 1
        if self._last_seq is None or seq != self._last_seq + 1:
            self._row_offset += self._batch_rows
            self._batch_rows = batch
            self._step = 0
        self._last_seq = seq

        activation = torch.zeros(batch, seq, self.N_HEADS, self.D_MODEL)
        for _name, fn in fwd_hooks or []:
            fn(activation, None)
        touched = sorted(set(torch.nonzero(activation)[:, 1].tolist()))
        self.calls.append(
            {
                "seq": seq,
                "touched_positions": touched,
                "attn_result_flag": self.cfg.use_attn_result,
            }
        )

        logits = torch.zeros(batch, seq, self.VOCAB)
        for i in range(batch):
            scripted = self.scripted_rows[self._row_offset + i]
            token = scripted[min(self._step, len(scripted) - 1)]
            logits[i, -1, token] = 1.0
        self._step += 1
        return logits


def test_decode_loop_positions_scoring_batching_and_flag_restore():
    # rows: "ab" scripted exactly, "cd" wrong on the 2nd token, "e" (shorter
    # target, pad-masked) correct -> expected correctness [1, 0, 1]
    prompts = ["12->", "345->", "6->"]
    targets = ["ab", "cd", "e"]
    scripted = {
        0: [ord("a"), ord("b")],
        1: [ord("c"), ord("z")],
        2: [ord("e")],
    }
    model = _StubModel(scripted)
    values = {(0, 1): torch.full((_StubModel.D_MODEL,), 7.0)}

    correct = ablated_generation_correctness(
        model, prompts, targets, values, batch_size=2
    )
    assert correct == [1, 0, 1]  # partial final batch evaluated, pad-masked score

    # batch 1 (rows 0,1): width 6 with BOS, final prompt token at absolute
    # index 5 for BOTH decode steps (seq grows 6 -> 7); batch 2 (row 2):
    # width 4 with BOS, one step, index 3
    assert [c["seq"] for c in model.calls] == [6, 7, 4]
    assert [c["touched_positions"] for c in model.calls] == [[5], [5], [3]]
    # use_attn_result forced on during every forward, restored afterwards
    assert all(c["attn_result_flag"] for c in model.calls)
    assert model.cfg.use_attn_result is False


def test_decode_loop_restores_flag_on_exception():
    model = _StubModel({0: [ord("a")]}, raise_on_forward=0)
    with pytest.raises(RuntimeError, match="scripted forward failure"):
        ablated_generation_correctness(model, ["12->"], ["a"], {}, batch_size=1)
    assert model.cfg.use_attn_result is False


def test_generation_refuses_length_mismatch():
    with pytest.raises(ValueError):
        ablated_generation_correctness(object(), ["a->"], [], {}, batch_size=2)


def test_subsets_deterministic_distinct_and_within_pool():
    pool = [(layer, head) for layer in (10, 11, 12) for head in range(5)]
    first = sample_head_subsets(pool, n_sets=20, set_size=3, seed=0)
    second = sample_head_subsets(pool, n_sets=20, set_size=3, seed=0)
    assert first == second  # deterministic in seed
    assert len({tuple(map(tuple, s)) for s in first}) == 20  # distinct sets
    for subset in first:
        assert len(subset) == 3
        assert len(set(map(tuple, subset))) == 3  # no repeats within a set
        assert all(tuple(h) in set(pool) for h in map(tuple, subset))
    assert first != sample_head_subsets(pool, n_sets=20, set_size=3, seed=1)


def test_subsets_refuse_impossible_requests():
    with pytest.raises(ValueError):
        sample_head_subsets([(1, 1), (1, 2)], n_sets=1, set_size=3, seed=0)
    with pytest.raises(ValueError):
        sample_head_subsets([(1, 1), (1, 2), (1, 3)], n_sets=2, set_size=3, seed=0)
    with pytest.raises(ValueError):
        sample_head_subsets([(1, 1), (1, 1)], n_sets=1, set_size=1, seed=0)
