"""AIE top-k selection (signed ranking, tie-breaks, head_limit) and the pure
first-token-probability CIE math."""

from __future__ import annotations

import pytest

from subspaces.artifacts import ArtifactError
from subspaces.config import ConfigError
from subspaces.step1.aie import (
    first_token_prob_deltas,
    first_token_probs,
    head_grid,
    select_top_k,
)


def test_top_k_ranks_by_signed_score_never_abs():
    # (0,1) has the largest |AIE| but a NEGATIVE sign: it must NOT be pulled
    # in (a later abs() re-ranking in the legacy code was a deviation).
    aie = [[0.5, -0.9], [0.1, 0.0]]
    heads = head_grid(2, 2)
    assert select_top_k(aie, heads, 2) == [(0, 0, 0.5), (1, 0, 0.1)]
    assert select_top_k(aie, heads, 4)[-1] == (0, 1, -0.9)


def test_top_k_tie_break_is_layer_then_head_ascending():
    aie = [[0.5, 0.5], [0.5, 0.1]]
    assert select_top_k(aie, head_grid(2, 2), 3) == [
        (0, 0, 0.5),
        (0, 1, 0.5),
        (1, 0, 0.5),
    ]


def test_top_k_bounds():
    aie = [[0.5, 0.1]]
    with pytest.raises(ConfigError, match="exceeds"):
        select_top_k(aie, head_grid(1, 2), 3)
    with pytest.raises(ConfigError, match=">= 1"):
        select_top_k(aie, head_grid(1, 2), 0)


def test_top_k_respects_head_limit():
    # best score sits at (1,0), OUTSIDE the flat row-major cap of 2
    aie = [[0.1, 0.2], [0.9, 0.0]]
    limited = head_grid(2, 2, head_limit=2)
    assert limited == [(0, 0), (0, 1)]
    assert select_top_k(aie, limited, 2) == [(0, 1, 0.2), (0, 0, 0.1)]


def test_top_k_refuses_nan_scores():
    aie = [[float("nan"), 0.2]]
    with pytest.raises(ArtifactError, match="NaN"):
        select_top_k(aie, head_grid(1, 2), 1)


def test_head_grid_row_major_and_cap():
    assert head_grid(2, 3) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert head_grid(2, 3, head_limit=4) == [(0, 0), (0, 1), (0, 2), (1, 0)]
    with pytest.raises(ArtifactError, match="degenerate"):
        head_grid(0, 3)


# -- pure CIE math -------------------------------------------------------------


def test_first_token_probs_float32_softmax_and_gather():
    import torch

    logits = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    targets = torch.tensor([0, 1])
    probs = first_token_probs(logits, targets)
    assert probs.dtype == torch.float32
    expected = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs, torch.tensor([expected[0, 0], expected[1, 1]]))


def test_first_token_deltas_match_manual_computation_and_mean():
    import torch

    baseline = torch.tensor([[1.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    patched = torch.tensor([[2.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    targets = torch.tensor([0, 1])
    deltas = first_token_prob_deltas(baseline, patched, targets)
    manual = (
        torch.softmax(patched.to(torch.float32), dim=-1)
        - torch.softmax(baseline.to(torch.float32), dim=-1)
    )[torch.arange(2), targets]
    assert torch.allclose(deltas, manual)
    # CIE(head | task) = MEAN of the per-prompt deltas
    assert torch.isclose(deltas.mean(), manual.mean())


def test_first_token_deltas_upcast_low_precision_logits():
    import torch

    baseline = torch.tensor([[8.0, -8.0], [0.5, 0.25]], dtype=torch.bfloat16)
    patched = torch.tensor([[-8.0, 8.0], [0.25, 0.5]], dtype=torch.bfloat16)
    targets = torch.tensor([1, 0])
    deltas = first_token_prob_deltas(baseline, patched, targets)
    assert deltas.dtype == torch.float32
    manual = (
        torch.softmax(patched.to(torch.float32), dim=-1)
        - torch.softmax(baseline.to(torch.float32), dim=-1)
    )[torch.arange(2), targets]
    assert torch.allclose(deltas, manual)


def test_first_token_deltas_shape_guards():
    import torch

    good = torch.zeros(2, 4)
    with pytest.raises(ArtifactError, match="shape mismatch"):
        first_token_prob_deltas(good, torch.zeros(3, 4), torch.tensor([0, 1]))
    with pytest.raises(ArtifactError, match=r"\[batch, vocab\]"):
        first_token_probs(torch.zeros(2, 3, 4), torch.tensor([0, 1]))
    with pytest.raises(ArtifactError, match="aligned"):
        first_token_probs(good, torch.tensor([0, 1, 2]))
