"""Appendix-F projection tooling tests (CPU): projection-matrix correctness,
arm naming, and CLI parsing for ``subspaces.runners.run_projection_causal``."""

from __future__ import annotations

import numpy as np
import pytest

from subspaces.runners.run_projection_causal import (
    SUBSPACE_COLS,
    arm_name,
    build_fv_vectors,
    build_projection_basis,
    parse_arms,
    parse_heads,
)

# --------------------------------------------------------------------------- #
# projection matrices                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("subspace", ["parity", "unit", "mag"])
def test_projection_bases_orthonormal_and_complementary(subspace):
    rng = np.random.default_rng(0)
    mod_vectors = rng.normal(size=(64, 6))
    cols = SUBSPACE_COLS[subspace]

    onto = build_projection_basis(mod_vectors, cols, "onto")
    out = build_projection_basis(mod_vectors, cols, "out")

    # shapes: onto spans |cols|, out spans the rest of the 6-D span
    assert onto.shape == (64, len(cols))
    assert out.shape == (64, 6 - len(cols))
    # orthonormal columns
    assert np.allclose(onto.T @ onto, np.eye(onto.shape[1]), atol=1e-10)
    assert np.allclose(out.T @ out, np.eye(out.shape[1]), atol=1e-10)
    # mutually orthogonal: the out basis is the complement WITHIN the span
    assert np.abs(onto.T @ out).max() < 1e-10

    # S Sᵀ + C Cᵀ equals the projector onto the full 6-D span on random input
    full_basis, _ = np.linalg.qr(mod_vectors)
    span_projector = full_basis @ full_basis.T
    split_projector = onto @ onto.T + out @ out.T
    x = rng.normal(size=(16, 64))
    assert np.allclose(x @ split_projector, x @ span_projector, atol=1e-9)


def test_onto_basis_spans_chosen_columns():
    rng = np.random.default_rng(1)
    mod_vectors = rng.normal(size=(32, 6))
    for subspace, cols in SUBSPACE_COLS.items():
        onto = build_projection_basis(mod_vectors, cols, "onto")
        # each chosen column is reproduced exactly by the onto projector
        chosen = mod_vectors[:, list(cols)]
        assert np.allclose(onto @ onto.T @ chosen, chosen, atol=1e-9), subspace
        # out-of projector annihilates the chosen columns
        out = build_projection_basis(mod_vectors, cols, "out")
        assert np.abs(out.T @ chosen).max() < 1e-9, subspace


def test_build_fv_vectors_formula():
    rng = np.random.default_rng(2)
    d_model = 24
    mod_vectors = rng.normal(size=(d_model, 6))
    basis = build_projection_basis(mod_vectors, SUBSPACE_COLS["unit"], "onto")
    z_head = {"number-add1": rng.normal(size=d_model)}
    mean_z = rng.normal(size=d_model)
    others = rng.normal(size=d_model)
    coefficient = 6

    fvs = build_fv_vectors(z_head, mean_z, others, basis, coefficient)
    centered = z_head["number-add1"] - mean_z
    expected = coefficient * (centered @ basis @ basis.T + mean_z) + others
    assert np.allclose(fvs["number-add1"], expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# arm naming / CLI parsing                                                     #
# --------------------------------------------------------------------------- #


def test_arm_name_matches_legacy_format():
    assert arm_name((15, 2), 6, "unit", "onto") == "(15, 2)*6=1-2-3-4"
    assert arm_name((15, 2), 6, "unit", "out") == "(15, 2)*6!=1-2-3-4"
    assert arm_name((13, 6), 5, "parity", "onto") == "(13, 6)*5=1"
    assert arm_name((21, 0), 3, "mag", "out") == "(21, 0)*3!=5-0"


def test_parse_heads_and_arms():
    assert parse_heads("15:2,15:1,13:6") == [(15, 2), (15, 1), (13, 6)]
    assert parse_arms("onto_unit") == [("onto", "unit")]
    assert len(parse_arms("all")) == 6
    with pytest.raises(ValueError):
        parse_arms("onto_diagonal")
    with pytest.raises(ValueError):
        parse_heads("15-2")
