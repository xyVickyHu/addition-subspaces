"""GPU scoring-contract tests (marked ``gpu``): the exact_token_match v1
semantics on variable-length and multi-token answers, on the REAL model.

MUST pass before any non-addition family launches (sweep-readiness item 4):
the non-add families have 14-68% multi-token answers, so the mask semantics
carry the science there. These tests also DOCUMENT the contract's known
leniencies (prefix over-credit; empty-target auto-pass) as assertions, so a
scorer change that silently alters them fails loudly.

    .venv/bin/pytest -m gpu tests/test_gpu_scoring_contract.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    from subspaces.config import load_step1_config
    from subspaces.step1.eval_gpu import load_model

    cfg = load_step1_config(REPO / "configs" / "step1_number_add_llama3.yaml")
    return load_model(cfg)


def _score(model, prompts, targets, batch_size):
    from subspaces.step1.eval_gpu import eval_prompts

    correct, _sizes = eval_prompts(model, prompts, targets, batch_size=batch_size)
    return correct


# 5-shot arrow-format subtraction demos: negative answers are 2-token on
# Llama-3 (['-', digits]) — the multi-token case the sweep families need.
_SUB_DEMOS = "10-12->-2#5-19->-14#8-11->-3#20-27->-7#3-9->-6#"


def test_variable_length_batch_composition_independence(model):
    """Verdicts must not depend on which other examples share the batch
    (the mask restricts comparison to each example's own target length)."""
    cases = [
        (f"{_SUB_DEMOS}6-13->", "-7"),  # 2-token answer
        (f"{_SUB_DEMOS}9-14->", "-5"),  # 2-token answer
        ("1->8#2->9#3->10#4->11#5->", "12"),  # 1-token answer
        ("1->8#2->9#3->10#4->11#6->", "13"),  # 1-token answer
        (f"{_SUB_DEMOS}7-10->", "-3"),  # 2-token answer
        ("2+3->5#4+4->8#1+2->3#3+3->6#2+2->", "4"),  # 1-token answer
    ]
    prompts = [p for p, _ in cases]
    targets = [t for _, t in cases]
    alone = [_score(model, [p], [t], batch_size=1)[0] for p, t in cases]
    together = _score(model, prompts, targets, batch_size=len(cases))
    assert together == alone


def test_multi_token_answer_requires_every_token(model):
    """A 2-token target is correct only when BOTH tokens match: pattern-copy
    demos force the emission '-7' (tokens ['-','7']); targets differing only
    in the SECOND token must disagree (guards against first-token-only
    scoring). Pattern copying, not arithmetic — no model-capability premise."""
    prompt = "a->-7#b->-7#c->-7#d->-7#e->"
    right = _score(model, [prompt], ["-7"], batch_size=1)[0]
    wrong = _score(model, [prompt], ["-8"], batch_size=1)[0]
    assert (right, wrong) == (1, 0)


def test_prefix_over_credit_is_the_documented_behavior(model):
    """exact_token_match v1 credits a target whose tokenization is a prefix
    of what the model emits (documented leniency — subspaces/step1/scoring.py).
    Llama-3 chunks digits in threes: '3650' tokenizes ['365','0'], so demos
    forcing '3650' make the first greedy token '365' — and the 1-token
    target '365' scores correct although the model is answering 3650.
    (NOT '36' vs '365': those are DIFFERENT single tokens, not a prefix.)"""
    prompt = "a->3650#b->3650#c->3650#d->3650#e->"
    scored = _score(model, [prompt], ["365"], batch_size=1)[0]
    assert scored == 1


def test_empty_target_auto_passes_in_mixed_batches(model):
    """An empty target row padded to the batch-max target length has an
    all-False mask and trivially passes — the hazard affecting one
    abstractive datapoint (present-past.json). Only reachable in MIXED
    batches: a lone empty target crashes on max_new_tokens=0 (shape (1,0)),
    which the corrected protocol's per-task batching could hit. Documented
    as current v1 behavior; a corrected scorer must change it."""
    prompts = ["a->7#b->7#c->", "1->8#2->"]
    targets = ["7", ""]  # batch-max target length 1; empty row all-masked
    scored = _score(model, prompts, targets, batch_size=2)
    assert scored[1] == 1  # auto-pass regardless of what the model emits
