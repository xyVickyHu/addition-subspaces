"""Versioned generation/scoring contract (sweep-readiness item 4).

Every scan/headset artifact records WHICH scorer produced its accuracies
(``config.scoring``), and the scorer joins run identity through requested
semantics — a scorer change can never silently reuse numbers produced by a
different rule.

Registered scorers
------------------

``exact_token_match`` v1 — the legacy contract, preserved verbatim as the
reproducible mode (implementation: ``subspaces.utils.intervene.
intervened_generation_with_accuracy``, reused by ``subspaces.step1.eval_gpu``):

- inputs tokenized ``prepend_bos=True``, LEFT padding; targets tokenized
  STANDALONE (``prepend_bos=False``), RIGHT padding;
- greedy decode of exactly the padded batch-max target length for every
  example;
- correct iff exact token-ID match on the example's own non-pad target
  positions (masked prefix match). No string normalization, no terminator
  check.

Documented consequences (measured 2026-07-22):
batch-composition-independent verdicts (mask); OVER-credits digit/word
continuations (target ``365`` matches a model emitting ``3650`` — Llama-3
chunks digits in threes, so ``3650`` tokenizes ['365','0']); fails
alternative tokenizations of the right string; fails any space-prefixed
emission; structurally unable to match on SentencePiece models
(Mistral-7B-v0.1 ``▁`` prefix); an EMPTY target auto-passes in mixed
batches (all-False mask) and CRASHES when scored alone (max_new_tokens=0)
— one datapoint in abstractive/present-past.json. Corrected scorers for the non-addition
families are PROPOSED and NOT implemented until the
user picks — new (name, version) entries here, never silent changes to v1.
"""

from __future__ import annotations

import re

SCORING_SCHEMA_VERSION = 1

# (name, version) -> frozenset of required parameter names (exact match).
SCORERS: dict[tuple[str, int], frozenset] = {
    ("exact_token_match", 1): frozenset(),
    # Corrected scorers (Q6 resolution, 2026-07-26;
    # "Resolved decisions"). Both score DECODED TEXT from a greedy window of
    # padded_target_len + WINDOW_SLACK tokens; greedy decoding is causal, so
    # exact_token_match flags computed on the same window's prefix are
    # bit-identical to the legacy decode. GPU wiring is a gated scorer
    # pilot; the text rules below are the frozen v1 contract.
    ("numeric_parse", 1): frozenset(),
    ("normalized_exact", 1): frozenset({"out_sep"}),
}

DEFAULT_SCORING = {"name": "exact_token_match", "version": 1}

# Generation-window slack (tokens beyond the padded target length) for the
# text scorers — part of the v1 contract, not tunable.
WINDOW_SLACK = 2


class ScoringError(ValueError):
    pass


_NUMERIC_RE = re.compile(r"^\s*([+-]?\d+)")


def score_numeric_parse(generated_text: str, target_text: str) -> bool:
    """``numeric_parse`` v1: leading-integer parse of the decoded window,
    integer comparison against the target.

    Tokenizer-invariant (fixes Qwen digit-split and SentencePiece space
    prefixes) and STRICTLY harder on over-credit than exact_token_match:
    the generation is truncated at its first non-digit character, so target
    365 does NOT match an emission of 3650 (int 3650 != 365), while the
    legacy token-prefix rule credits it. Leading whitespace and a sign are
    accepted; a target that does not itself parse as a bare integer refuses
    — this scorer is for arithmetic families only.
    """
    target_match = _NUMERIC_RE.match(target_text)
    if not target_match or target_match.group(1) != target_text.strip():
        raise ScoringError(
            f"numeric_parse requires an integer target; got {target_text!r}"
        )
    generated_match = _NUMERIC_RE.match(generated_text)
    if not generated_match:
        return False
    return int(generated_match.group(1)) == int(target_match.group(1))


def _normalize(text: str) -> str:
    return " ".join(text.split())


def score_normalized_exact(
    generated_text: str, target_text: str, *, out_sep: str
) -> bool:
    """``normalized_exact`` v1: truncate the decoded window at the format's
    output separator, collapse whitespace, exact string comparison.

    NO casefolding (capitalize_*/lowercase_* abstractive tasks must remain
    scoreable) and no accent stripping (translation tasks). An empty
    NORMALIZED target refuses instead of auto-passing — the guard for the
    known empty-target datapoint (present-past.json index 110;
    Q7): under the legacy scorer it silently auto-passes
    in mixed batches.
    """
    if not out_sep:
        raise ScoringError("normalized_exact requires a non-empty out_sep")
    normalized_target = _normalize(target_text)
    if not normalized_target:
        raise ScoringError(
            "normalized_exact refuses an empty target (validity overlay: "
            "score this datapoint as invalid, do not auto-pass it)"
        )
    generated = generated_text.split(out_sep, 1)[0]
    return _normalize(generated) == normalized_target


def validate_scoring(name: str, version: int, params: dict | None) -> dict:
    """Exact-registry validation; returns the canonical scoring dict."""
    key = (name, version)
    if key not in SCORERS:
        known = ", ".join(f"{n}@v{v}" for n, v in sorted(SCORERS))
        raise ScoringError(f"unknown scorer {name}@v{version}; registered: {known}")
    required = SCORERS[key]
    given = frozenset(params or {})
    if given != required:
        raise ScoringError(
            f"scorer {name}@v{version} requires exactly params "
            f"{sorted(required)}; got {sorted(given)}"
        )
    out = {"name": name, "version": version}
    if params:
        out["params"] = dict(params)
    return out
