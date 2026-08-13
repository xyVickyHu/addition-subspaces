"""Text rules of the corrected scorers (Q6 resolution, 2026-07-26):
``numeric_parse`` v1 and ``normalized_exact`` v1. Pure CPU contracts —
the GPU windowed-generation wiring is a separately gated scorer pilot."""

from __future__ import annotations

import pytest

from subspaces.step1.scoring import (
    ScoringError,
    score_normalized_exact,
    score_numeric_parse,
    validate_scoring,
)

# -- registry ------------------------------------------------------------------


def test_new_scorers_registered_with_exact_params():
    assert validate_scoring("numeric_parse", 1, None) == {
        "name": "numeric_parse",
        "version": 1,
    }
    assert validate_scoring("normalized_exact", 1, {"out_sep": "#"}) == {
        "name": "normalized_exact",
        "version": 1,
        "params": {"out_sep": "#"},
    }
    with pytest.raises(ScoringError, match="requires exactly params"):
        validate_scoring("numeric_parse", 1, {"window": 3})
    with pytest.raises(ScoringError, match="requires exactly params"):
        validate_scoring("normalized_exact", 1, None)


# -- numeric_parse v1 ----------------------------------------------------------


def test_numeric_parse_rejects_the_etm_overcredit_case():
    """Target 365 vs emission 3650: exact_token_match CREDITS it (token
    prefix ['365','0']); numeric_parse must not."""
    assert score_numeric_parse("3650", "365") is False
    assert score_numeric_parse("365", "365") is True


def test_numeric_parse_is_tokenization_invariant():
    # space-prefixed emissions and separator-terminated continuations match
    assert score_numeric_parse(" 365", "365") is True
    assert score_numeric_parse("365#12->", "365") is True
    assert score_numeric_parse("\n365", "365") is True
    # signs and equivalent integer spellings
    assert score_numeric_parse("-42", "-42") is True
    assert score_numeric_parse("+42", "42") is True
    assert score_numeric_parse("007", "7") is True


def test_numeric_parse_negative_cases():
    assert score_numeric_parse("", "365") is False
    assert score_numeric_parse("answer: 365", "365") is False  # leading text
    assert score_numeric_parse("36", "365") is False
    assert score_numeric_parse("-365", "365") is False


def test_numeric_parse_refuses_non_integer_target():
    for target in ("", "hinted", "3.5", "3 65"):
        with pytest.raises(ScoringError, match="integer target"):
            score_numeric_parse("3", target)


# -- normalized_exact v1 -------------------------------------------------------


def test_normalized_exact_truncates_at_out_sep_and_collapses_whitespace():
    assert score_normalized_exact("hinted#7->8", "hinted", out_sep="#") is True
    assert score_normalized_exact("  hinted \n", "hinted", out_sep="#") is True
    assert score_normalized_exact("a  b\tc", "a b c", out_sep="#") is True
    assert score_normalized_exact("hinted extra", "hinted", out_sep="#") is False
    assert score_normalized_exact("hint", "hinted", out_sep="#") is False


def test_normalized_exact_does_not_casefold():
    """capitalize_*/lowercase_* abstractive tasks stay scoreable: case is
    part of the answer."""
    assert score_normalized_exact("Hinted", "hinted", out_sep="#") is False
    assert score_normalized_exact("HINTED", "HINTED", out_sep="#") is True


def test_normalized_exact_refuses_empty_target():
    """The present-past.json[110] empty-target guard (Q7): refuse instead of
    the legacy auto-pass."""
    with pytest.raises(ScoringError, match="empty target"):
        score_normalized_exact("anything", "", out_sep="#")
    with pytest.raises(ScoringError, match="empty target"):
        score_normalized_exact("anything", "   ", out_sep="#")


def test_normalized_exact_requires_out_sep():
    with pytest.raises(ScoringError, match="out_sep"):
        score_normalized_exact("x", "x", out_sep="")
