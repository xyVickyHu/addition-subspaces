"""Lane-A selector equivalence: the ports must reproduce the captured legacy
outputs EXACTLY (fixtures committed with source hashes by
the development repo), plus determinism/edge-case behavior."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from subspaces.step1.selectors import (
    ABOVE_ABLATION_V1_PARAMS,
    OUTCOME_SELECTORS,
    PAIRED_BH_V1_PARAMS,
    PAIRED_BY_V1_PARAMS,
    PAIRED_GLOBAL_BONFERRONI_V1_PARAMS,
    PAIRED_PER_HEAD_V1_PARAMS,
    SELECTOR_PARAM_SCHEMAS,
    SELECTORS,
    UNIFIED_V1_PARAMS,
    WEAK_RELEASE_PARAMS,
    WEAK_SHADOW_DEFAULTS,
    SelectorError,
    _binomial_quantile,
    _mcnemar_tail,
    _step_up_adjusted,
    above_ablation_v1,
    above_ablation_zero_v1,
    compare_zero_v1,
    get_selector,
    largest_gap_v1,
    paired_bh_v1,
    paired_by_v1,
    paired_global_bonferroni_v1,
    paired_per_head_v1,
    pin_v1,
    recovery_weak_v1,
    unified_v1,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def scan_fixture() -> dict:
    raw = json.loads(
        (FIXTURES / "selector_scan_repro_v1.json").read_text(encoding="utf-8")
    )
    return {
        "curves": raw["curves"],
        "baselines": {
            "clean_acc": raw["clean_acc"],
            "full_significant_acc": raw["full_significant_fv_acc"],
        },
        "n_eval_examples_per_head_per_c": raw["n_eval"],
        "stored_main_heads_weak_release": raw["stored_main_heads_weak_release"],
    }


@pytest.fixture(scope="module")
def expected() -> dict:
    return json.loads(
        (FIXTURES / "selector_expected_repro_v1.json").read_text(encoding="utf-8")
    )


def _assert_decisions_equal(actual: dict, exp: dict) -> None:
    assert set(actual) == set(exp)
    for key, exp_decision in exp.items():
        act_decision = actual[key]
        assert set(act_decision) == set(exp_decision), key
        for field_name, exp_value in exp_decision.items():
            act_value = act_decision[field_name]
            if isinstance(exp_value, float):
                assert act_value == pytest.approx(exp_value, abs=0.0), (
                    key,
                    field_name,
                )
            else:
                assert act_value == exp_value, (key, field_name)


def test_unified_v1_matches_legacy_capture(scan_fixture, expected):
    exp = expected["results"]["unified@v1"]
    result = unified_v1(scan_fixture, UNIFIED_V1_PARAMS)
    assert [list(h) for h in result.main_heads] == exp["main_heads"]
    assert [list(h) for h in result.minor_heads] == exp["minor_heads"]
    _assert_decisions_equal(result.decisions, exp["decisions"])
    assert result.verdict == exp["verdict"]


def test_unified_v1_selects_paper_trio_with_borderline_as_minor(scan_fixture):
    result = unified_v1(scan_fixture, UNIFIED_V1_PARAMS)
    assert result.main_heads == [(15, 2), (15, 1), (13, 6)]
    assert (15, 28) in result.minor_heads  # borderline head: minor, not main
    assert result.verdict["verdict"] == "localized"


def test_recovery_weak_v1_matches_legacy_capture_both_params(scan_fixture, expected):
    for label, params in (
        ("recovery_weak@release", WEAK_RELEASE_PARAMS),
        ("recovery_weak@shadow_defaults", WEAK_SHADOW_DEFAULTS),
    ):
        exp = expected["results"][label]
        assert exp["params"] == params
        result = recovery_weak_v1(scan_fixture, params)
        assert [list(h) for h in result.main_heads] == exp["main_heads"]
        _assert_decisions_equal(result.decisions, exp["decisions"])


def test_weak_release_reproduces_stored_selection(scan_fixture):
    """The repro run's stored main_heads came from these exact parameters."""
    result = recovery_weak_v1(scan_fixture, WEAK_RELEASE_PARAMS)
    assert [list(h) for h in result.main_heads] == scan_fixture[
        "stored_main_heads_weak_release"
    ]


def test_fixture_sources_are_pinned(expected):
    """The recorded legacy-source hashes must match the files that exist NOW —
    a drifted legacy source invalidates the captured expectations."""
    import hashlib

    for source in expected["sources"].values():
        assert len(source["sha256"]) == 64

    release_heads = (
        Path(__file__).resolve().parents[1] / "subspaces" / "utils" / "heads.py"
    )
    recorded = expected["sources"]["release_heads_py"]["sha256"]
    assert hashlib.sha256(release_heads.read_bytes()).hexdigest() == recorded

    cleaner_source = Path(expected["sources"]["fv_cleaner_heads_py"]["file"])
    if cleaner_source.is_file():  # external workspace; may be absent elsewhere
        recorded = expected["sources"]["fv_cleaner_heads_py"]["sha256"]
        assert hashlib.sha256(cleaner_source.read_bytes()).hexdigest() == recorded


def test_selectors_are_deterministic(scan_fixture):
    for selector, params in (
        (unified_v1, UNIFIED_V1_PARAMS),
        (recovery_weak_v1, WEAK_RELEASE_PARAMS),
        (largest_gap_v1, {}),
        (above_ablation_v1, ABOVE_ABLATION_V1_PARAMS),
        (above_ablation_v1, {"alpha": 0.05, "scanwise": 1.0}),
        (above_ablation_zero_v1, {}),
        (compare_zero_v1, {}),
    ):
        first = selector(scan_fixture, params)
        second = selector(scan_fixture, params)
        assert first == second


def test_exact_tie_is_broken_deterministically():
    curve = {"0": 0.10, "1": 0.50, "2": 0.40}
    scan = {
        "curves": {"5:1": dict(curve), "3:2": dict(curve)},  # identical curves
        "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
        "n_eval_examples_per_head_per_c": 300,
    }
    result = unified_v1(scan, UNIFIED_V1_PARAMS)
    # equal gain and peak -> ordered by (layer, head)
    assert result.main_heads == [(3, 2), (5, 1)]
    weak = recovery_weak_v1(scan, WEAK_RELEASE_PARAMS)
    assert weak.main_heads == [(3, 2), (5, 1)]


def test_sparse_and_nan_curve_points_are_deterministic():
    scan = {
        "curves": {
            "1:1": {"0": 0.1, "5": 0.6},  # sparse grid
            "2:2": {"0": 0.1, "1": float("nan"), "2": 0.55},  # NaN point
        },
        "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
        "n_eval_examples_per_head_per_c": 300,
    }
    first = unified_v1(scan, UNIFIED_V1_PARAMS)
    second = unified_v1(scan, UNIFIED_V1_PARAMS)
    assert first == second
    assert first.main_heads == second.main_heads
    stats = first.decisions["1:1"]
    assert stats["c_star"] == 5 and stats["gain"] == pytest.approx(0.5)
    # NaN never becomes the peak (comparisons are False), result is finite
    assert not math.isnan(first.decisions["2:2"]["acc_at_c_star"])


def _peak_scan(peaks: dict[str, float]) -> dict:
    """Synthetic scan where each head's curve rises from 0.0 to its peak."""
    return {
        "curves": {key: {"0": 0.0, "1": value} for key, value in peaks.items()},
        "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
        "n_eval_examples_per_head_per_c": 300,
    }


def test_largest_gap_v1_selects_trio_on_repro_fixture(scan_fixture):
    """On the shipped repro scan the largest peak-accuracy drop is the one
    below the paper trio: 0.7133… (13,6) -> 0.19 (15,28). Expected numbers
    were derived independently of the implementation (plain-python recompute
    over the fixture's curves)."""
    result = largest_gap_v1(scan_fixture, {})
    assert result.main_heads == [(15, 2), (15, 1), (13, 6)]
    assert result.minor_heads == []
    assert result.verdict["verdict"] == "localized"
    # every scanned head gets a decision entry, selected or not
    assert set(result.decisions) == set(scan_fixture["curves"])
    assert sum(d["is_main"] for d in result.decisions.values()) == 3
    gap = result.verdict["gap"]
    assert gap["rank_of_upper"] == 3
    assert gap["n_selected"] == 3
    assert gap["upper"] == pytest.approx(0.7133333333333334, abs=0.0)
    assert gap["lower"] == pytest.approx(0.19, abs=0.0)
    assert gap["drop"] == pytest.approx(0.5233333333333334, abs=0.0)
    runner_up = gap["runner_up"]
    assert runner_up["rank_of_upper"] == 2
    assert runner_up["drop"] == pytest.approx(0.13, abs=0.0)
    # the winning cut is nowhere near the runner-up on this scan
    assert gap["drop"] - runner_up["drop"] > 0.35
    decisions = result.decisions["15:2"]
    assert decisions["is_main"] and decisions["main_reason"] == "above_largest_gap"
    assert result.decisions["15:28"]["flags"] == ["below_largest_gap"]


def test_largest_gap_v1_upper_edge_ties_stay_in():
    scan = _peak_scan({"1:1": 0.8, "2:2": 0.8, "3:3": 0.2, "4:4": 0.1})
    result = largest_gap_v1(scan, {})
    # drops: [0, 0.6, 0.1] -> cut below BOTH tied 0.8 heads
    assert result.main_heads == [(1, 1), (2, 2)]
    assert result.verdict["gap"]["n_selected"] == 2
    assert result.verdict["threshold"] == pytest.approx(0.2, abs=0.0)


def test_largest_gap_v1_lower_edge_ties_stay_out():
    scan = _peak_scan({"1:1": 0.9, "2:2": 0.5, "3:3": 0.5, "4:4": 0.5})
    result = largest_gap_v1(scan, {})
    assert result.main_heads == [(1, 1)]
    assert all(
        result.decisions[key]["flags"] == ["below_largest_gap"]
        for key in ("2:2", "3:3", "4:4")
    )


def test_largest_gap_v1_drop_tie_takes_fewest_heads():
    # power-of-two values so the two drops tie EXACTLY in float arithmetic
    scan = _peak_scan({"1:1": 0.75, "2:2": 0.5, "3:3": 0.25})
    result = largest_gap_v1(scan, {})
    # drops [0.25, 0.25] tie exactly -> first (highest) cut wins
    assert result.main_heads == [(1, 1)]
    gap = result.verdict["gap"]
    assert gap["rank_of_upper"] == 1
    assert gap["runner_up"]["rank_of_upper"] == 2
    assert gap["runner_up"]["drop"] == pytest.approx(gap["drop"], abs=0.0)


def test_largest_gap_v1_refusals():
    with pytest.raises(SelectorError, match="fewer than two"):
        largest_gap_v1(_peak_scan({"1:1": 0.5}), {})
    with pytest.raises(SelectorError, match="no drop"):
        largest_gap_v1(_peak_scan({"1:1": 0.5, "2:2": 0.5}), {})
    nan_scan = {
        "curves": {
            "1:1": {"0": float("nan"), "1": 0.2},  # NaN at c=0 peaks (legacy)
            "2:2": {"0": 0.1, "1": 0.5},
        },
        "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
        "n_eval_examples_per_head_per_c": 300,
    }
    with pytest.raises(SelectorError, match="NaN peak"):
        largest_gap_v1(nan_scan, {})


def test_largest_gap_v1_tolerates_nan_that_never_peaks():
    scan = {
        "curves": {
            "1:1": {"0": 0.1, "1": float("nan"), "2": 0.9},
            "2:2": {"0": 0.1, "1": 0.2, "2": 0.3},
        },
        "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
        "n_eval_examples_per_head_per_c": 300,
    }
    result = largest_gap_v1(scan, {})
    assert result.main_heads == [(1, 1)]


def test_pin_v1_requires_scanned_heads(scan_fixture):
    result = pin_v1(scan_fixture, {"heads": [[15, 2], [13, 6]]})
    assert result.main_heads == [(15, 2), (13, 6)]
    with pytest.raises(SelectorError, match="not in the scan"):
        pin_v1(scan_fixture, {"heads": [[0, 0]]})
    with pytest.raises(SelectorError, match="requires params"):
        pin_v1(scan_fixture, {})


def test_binomial_quantile_exact_values():
    # Binomial(10, 0.5): CDF(4)=0.3769..., CDF(5)=0.6230... -> q50 is k=5
    assert _binomial_quantile(10, 0.5, 0.5) == 5
    # CDF(3)=0.171875, CDF(4)=0.376953125 -> any q between them returns 4
    assert _binomial_quantile(10, 0.5, 0.3769) == 4
    assert _binomial_quantile(300, 0.0, 0.99) == 0
    assert _binomial_quantile(300, 1.0, 0.5) == 300
    # the repro-fixture thresholds (n=300, pooled c0 = 0.0281818...):
    pooled = 0.028181818181818176
    assert _binomial_quantile(300, pooled, (1 - 0.05) ** (1 / 12)) == 17
    assert _binomial_quantile(300, pooled, (1 - 0.05) ** (1 / 396)) == 21


def test_above_ablation_v1_on_repro_fixture(scan_fixture):
    """Expected numbers derived independently of the implementation
    (plain-python recompute over the fixture's curves): pooled c=0 accuracy
    0.0281818..., per-head null threshold 17/300, scan-wise 21/300."""
    result = above_ablation_v1(scan_fixture, ABOVE_ABLATION_V1_PARAMS)
    assert result.main_heads == [
        (15, 2),
        (15, 1),
        (13, 6),
        (15, 28),
        (15, 17),
        (13, 27),
        (13, 21),
        (13, 22),
        (16, 8),
        (15, 30),
    ]
    assert result.minor_heads == []
    assert result.verdict["verdict"] == "distributed"
    assert result.verdict["n_selected"] == 10
    noise = result.verdict["noise"]
    assert noise["pooled_c0"] == pytest.approx(0.028181818181818176, abs=0.0)
    assert noise["threshold"] == pytest.approx(17 / 300, abs=0.0)
    assert noise["scanwise"] is False
    assert noise["grid_points_total"] == 33 * 12
    assert noise["expected_false_selected_upper"] == pytest.approx(0.05 * 33)
    # every scanned head gets a decision entry, selected or not
    assert set(result.decisions) == set(scan_fixture["curves"])
    top = result.decisions["15:2"]
    assert top["is_main"] and top["main_reason"] == "above_noise_band"
    assert top["excess"] == pytest.approx(top["acc_at_c_star"] - 17 / 300)

    scanwise = above_ablation_v1(scan_fixture, {"alpha": 0.05, "scanwise": 1.0})
    assert scanwise.main_heads == result.main_heads[:9]  # (15,30) drops out
    assert scanwise.verdict["noise"]["threshold"] == pytest.approx(0.07, abs=0.0)
    assert scanwise.verdict["noise"]["expected_false_selected_upper"] == 0.05
    assert scanwise.params == {"alpha": 0.05, "scanwise": 1.0}


def test_above_ablation_zero_v1_on_repro_fixture(scan_fixture):
    result = above_ablation_zero_v1(scan_fixture, {})
    # strict gain > 0 == peak at a non-zero coefficient (first-index argmax)
    expected = {
        key
        for key, curve in scan_fixture["curves"].items()
        if max(curve.values()) > curve["0"]
    }
    assert {f"{h[0]}:{h[1]}" for h in result.main_heads} == expected
    assert result.verdict["n_selected"] == 26
    assert result.main_heads[:3] == [(15, 2), (15, 1), (13, 6)]
    assert result.verdict["noise"] == {"band": "zero"}
    for key in set(scan_fixture["curves"]) - expected:
        assert result.decisions[key]["flags"] == ["peak_at_c0"]


def test_above_ablation_band_nesting_on_repro_fixture(scan_fixture):
    """Stricter noise regions select subsets: scanwise ⊆ per-head ⊆ zero."""
    zero = set(above_ablation_zero_v1(scan_fixture, {}).main_heads)
    per_head = set(
        above_ablation_v1(scan_fixture, {"alpha": 0.05, "scanwise": 0.0}).main_heads
    )
    scanwise = set(
        above_ablation_v1(scan_fixture, {"alpha": 0.05, "scanwise": 1.0}).main_heads
    )
    assert scanwise <= per_head <= zero
    stricter = set(
        above_ablation_v1(scan_fixture, {"alpha": 0.01, "scanwise": 0.0}).main_heads
    )
    assert stricter <= per_head


def test_above_ablation_v1_orders_ties_deterministically():
    curve = {"0": 0.10, "1": 0.50, "2": 0.40}
    scan = {
        "curves": {"5:1": dict(curve), "3:2": dict(curve)},
        "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
        "n_eval_examples_per_head_per_c": 300,
    }
    result = above_ablation_v1(scan, ABOVE_ABLATION_V1_PARAMS)
    assert result.main_heads == [(3, 2), (5, 1)]
    zero = above_ablation_zero_v1(scan, {})
    assert zero.main_heads == [(3, 2), (5, 1)]


def test_above_ablation_v1_param_validation(scan_fixture):
    with pytest.raises(SelectorError, match="exactly params"):
        above_ablation_v1(scan_fixture, {"alpha": 0.05})
    with pytest.raises(SelectorError, match="exactly params"):
        above_ablation_v1(scan_fixture, {"alpha": 0.05, "scanwise": 0.0, "bonus": 1.0})
    for bad_alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(SelectorError, match="alpha must be in"):
            above_ablation_v1(scan_fixture, {"alpha": bad_alpha, "scanwise": 0.0})
    with pytest.raises(SelectorError, match="scanwise must be"):
        above_ablation_v1(scan_fixture, {"alpha": 0.05, "scanwise": 0.5})
    with pytest.raises(SelectorError, match="no parameters"):
        above_ablation_zero_v1(scan_fixture, {"alpha": 0.05})


def test_above_ablation_v1_refusals():
    no_n_eval = {
        "curves": {"1:1": {"0": 0.1, "1": 0.5}},
        "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
    }
    with pytest.raises(SelectorError, match="n_eval_examples_per_head_per_c"):
        above_ablation_v1(no_n_eval, ABOVE_ABLATION_V1_PARAMS)
    single_point = {
        "curves": {"1:1": {"0": 0.1}},
        "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
        "n_eval_examples_per_head_per_c": 300,
    }
    with pytest.raises(SelectorError, match="non-zero"):
        above_ablation_v1(single_point, ABOVE_ABLATION_V1_PARAMS)
    with pytest.raises(SelectorError, match="non-zero"):
        above_ablation_zero_v1(single_point, {})
    # a grid without the c=0 point (or starting below 0) has no ablated
    # baseline: acc_at_c0 would silently be the smallest-c accuracy
    for grid in ({"1": 0.1, "2": 0.5}, {"-1": 0.1, "0": 0.2, "1": 0.5}):
        no_c0 = {
            "curves": {"1:1": dict(grid)},
            "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
            "n_eval_examples_per_head_per_c": 300,
        }
        with pytest.raises(SelectorError, match="needs c=0"):
            above_ablation_v1(no_c0, ABOVE_ABLATION_V1_PARAMS)
        with pytest.raises(SelectorError, match="needs c=0"):
            above_ablation_zero_v1(no_c0, {})
    nan_scan = {
        "curves": {
            "1:1": {"0": float("nan"), "1": 0.2},  # NaN at c=0 peaks (legacy)
            "2:2": {"0": 0.1, "1": 0.5},
        },
        "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
        "n_eval_examples_per_head_per_c": 300,
    }
    with pytest.raises(SelectorError, match="NaN peak"):
        above_ablation_v1(nan_scan, ABOVE_ABLATION_V1_PARAMS)
    with pytest.raises(SelectorError, match="NaN peak"):
        above_ablation_zero_v1(nan_scan, {})


def test_above_ablation_v1_selects_none_when_all_peaks_at_c0():
    scan = {
        "curves": {
            "1:1": {"0": 0.5, "1": 0.4},
            "2:2": {"0": 0.5, "1": 0.3},
        },
        "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
        "n_eval_examples_per_head_per_c": 300,
    }
    result = above_ablation_v1(scan, ABOVE_ABLATION_V1_PARAMS)
    assert result.main_heads == []
    assert result.verdict["verdict"] == "none"
    assert all(d["flags"] == ["peak_at_c0"] for d in result.decisions.values())
    zero = above_ablation_zero_v1(scan, {})
    assert zero.main_heads == [] and zero.verdict["verdict"] == "none"


def test_compare_zero_v1_on_repro_fixture(scan_fixture):
    """Expected set derived independently of the implementation: the bar is
    the largest c=0 accuracy over the fixture's curves; main iff the curve's
    max strictly exceeds it."""
    max_c0 = max(curve["0"] for curve in scan_fixture["curves"].values())
    expected = {
        key
        for key, curve in scan_fixture["curves"].items()
        if max(curve.values()) > max_c0
    }
    result = compare_zero_v1(scan_fixture, {})
    assert {f"{h[0]}:{h[1]}" for h in result.main_heads} == expected
    # peak-accuracy order puts the paper trio first (as for largest_gap)
    assert result.main_heads[:3] == [(15, 2), (15, 1), (13, 6)]
    noise = result.verdict["noise"]
    assert noise["band"] == "max_c0"
    assert noise["threshold"] == pytest.approx(max_c0, abs=0.0)
    assert noise["n_c0_pooled"] == len(scan_fixture["curves"])
    for key in noise["max_c0_heads"]:
        head_key = f"{key[0]}:{key[1]}"
        assert scan_fixture["curves"][head_key]["0"] == max_c0
    # every scanned head gets a decision entry, selected or not
    assert set(result.decisions) == set(scan_fixture["curves"])
    top = result.decisions["15:2"]
    assert top["is_main"] and top["main_reason"] == "above_max_c0"
    assert top["excess"] == pytest.approx(top["acc_at_c_star"] - max_c0)
    # the bar >= every head's own c0 => always a subset of the zero band
    zero = set(above_ablation_zero_v1(scan_fixture, {}).main_heads)
    assert set(result.main_heads) <= zero


def test_compare_zero_v1_orders_ties_deterministically():
    curve = {"0": 0.10, "1": 0.50, "2": 0.40}
    scan = {
        "curves": {"5:1": dict(curve), "3:2": dict(curve)},
        "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
    }
    result = compare_zero_v1(scan, {})
    assert result.main_heads == [(3, 2), (5, 1)]
    assert result.verdict["noise"]["max_c0_heads"] == [[3, 2], [5, 1]]


def test_compare_zero_v1_param_validation(scan_fixture):
    with pytest.raises(SelectorError, match="no parameters"):
        compare_zero_v1(scan_fixture, {"alpha": 0.05})


def test_compare_zero_v1_refusals():
    single_point = {
        "curves": {"1:1": {"0": 0.1}},
        "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
    }
    with pytest.raises(SelectorError, match="non-zero"):
        compare_zero_v1(single_point, {})
    for grid in ({"1": 0.1, "2": 0.5}, {"-1": 0.1, "0": 0.2, "1": 0.5}):
        no_c0 = {
            "curves": {"1:1": dict(grid)},
            "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
        }
        with pytest.raises(SelectorError, match="needs c=0"):
            compare_zero_v1(no_c0, {})
    # a NaN at c=0 surfaces as a NaN peak (first-index argmax), so the
    # NaN-peak refusal also protects the max-c0 bar from NaN poisoning
    nan_scan = {
        "curves": {
            "1:1": {"0": float("nan"), "1": 0.2},
            "2:2": {"0": 0.1, "1": 0.5},
        },
        "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
    }
    with pytest.raises(SelectorError, match="NaN peak"):
        compare_zero_v1(nan_scan, {})


def test_compare_zero_v1_selects_none_when_no_peak_beats_max_c0():
    # head 2:2 beats its OWN c0 (the zero band selects it) but not the
    # scan's largest c0 — the distinguishing case vs above_ablation_zero
    scan = {
        "curves": {
            "1:1": {"0": 0.5, "1": 0.4},
            "2:2": {"0": 0.1, "1": 0.45},
        },
        "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
        "n_eval_examples_per_head_per_c": 300,
    }
    result = compare_zero_v1(scan, {})
    assert result.main_heads == []
    assert result.verdict["verdict"] == "none"
    assert result.verdict["noise"]["threshold"] == pytest.approx(0.5)
    assert result.verdict["noise"]["max_c0_heads"] == [[1, 1]]
    assert result.decisions["1:1"]["flags"] == ["peak_at_c0"]
    assert result.decisions["2:2"]["flags"] == ["below_max_c0"]
    zero = above_ablation_zero_v1(scan, {})
    assert zero.main_heads == [(2, 2)]


def test_registry_and_schemas_consistent():
    for key in SELECTOR_PARAM_SCHEMAS:
        assert callable(get_selector(*key))
    with pytest.raises(SelectorError, match="unknown selector"):
        get_selector("magic", 1)
    # every outcome-consuming selector is registered with a schema
    assert set(SELECTOR_PARAM_SCHEMAS) >= OUTCOME_SELECTORS
    assert set(SELECTORS) >= OUTCOME_SELECTORS


# --- paired McNemar family --------------------------------------------------


def _paired_scan(vectors: dict[str, dict[int, list[int]]]) -> tuple[dict, dict]:
    """Scan manifest + outcomes mapping from explicit per-head 0/1 vectors.

    Curve accuracies are derived from the vectors exactly as the recovery
    scan derives them (mean of the outcome vector), and outcome values are
    handed over as ``bytes`` — the same form ``apply_selector`` produces
    from the npz arrays."""
    curves: dict[str, dict[str, float]] = {}
    outcomes: dict[str, bytes] = {}
    n_eval = None
    for head_key, by_c in vectors.items():
        curve: dict[str, float] = {}
        for c, y in by_c.items():
            n_eval = len(y)
            curve[str(c)] = sum(y) / len(y)
            outcomes[f"{head_key}:{c}"] = bytes(y)
        curves[head_key] = curve
    scan = {
        "curves": curves,
        "baselines": {"clean_acc": 0.9, "full_significant_acc": 0.8},
        "n_eval_examples_per_head_per_c": n_eval,
        "outcomes": {"file": "scan_outcomes.npz", "content_sha256": "f" * 64},
    }
    return scan, outcomes


def _vec(n_eval: int, fixed: int, broken: int, base_correct: int) -> dict[int, list]:
    """One head's {0: y0, 1: y1} pair with exact discordant counts.

    Layout: [base_correct correct-at-both][fixed 0->1][broken 1->0][rest 0/0].
    """
    assert base_correct + fixed + broken <= n_eval
    y0 = (
        [1] * base_correct
        + [0] * fixed
        + [1] * broken
        + [0] * (n_eval - base_correct - fixed - broken)
    )
    y1 = (
        [1] * base_correct
        + [1] * fixed
        + [0] * broken
        + [0] * (n_eval - base_correct - fixed - broken)
    )
    return {0: y0, 1: y1}


def test_mcnemar_tail_exact_values():
    # no discordant examples -> p = 1 by convention
    assert _mcnemar_tail(0, 0) == 1.0
    # b = 0 -> the whole distribution is in the tail
    assert _mcnemar_tail(6, 0) == 1.0
    # P[Binomial(4, 1/2) >= 3] = (4 + 1) / 16
    assert _mcnemar_tail(4, 3) == 5 / 16
    # odd-n symmetry: P[Binomial(2k+1, 1/2) >= k+1] = 1/2 exactly
    assert _mcnemar_tail(1, 1) == 0.5
    assert _mcnemar_tail(3, 2) == 0.5
    assert _mcnemar_tail(5, 3) == 0.5
    # all-fixed extremes
    assert _mcnemar_tail(2, 2) == 0.25
    assert _mcnemar_tail(10, 10) == 2.0**-10
    # defensive: b beyond n has zero tail
    assert _mcnemar_tail(3, 4) == 0.0


def test_step_up_adjusted_bh_and_by_hand_example():
    p_values = [0.01, 0.04, 0.03, 0.005]
    # sorted: .005 .01 .03 .04 -> raw N*p/rank: .02 .02 .04 .04 (monotone)
    bh = _step_up_adjusted(p_values, harmonic=False)
    assert bh == pytest.approx([0.02, 0.04, 0.04, 0.02], abs=1e-15)
    # BY = BH scaled by c(4) = 1 + 1/2 + 1/3 + 1/4 = 25/12 before the cummin
    c_4 = 25 / 12
    by = _step_up_adjusted(p_values, harmonic=True)
    assert by == pytest.approx([v * c_4 for v in bh], abs=1e-15)
    # the step-up minimum propagates: a large late p is capped by a smaller
    # N*p/rank further down the sorted order
    capped = _step_up_adjusted([0.001, 0.5, 0.011], harmonic=False)
    # sorted .001 .011 .5 -> raw .003 .0165 .5 -> cummin right: unchanged
    assert capped == pytest.approx([0.003, 0.5, 0.0165], abs=1e-15)
    non_monotone = _step_up_adjusted([0.02, 0.021], harmonic=False)
    # raw: .04, .021 -> rank-1 value is capped by the rank-2 one
    assert non_monotone == pytest.approx([0.021, 0.021], abs=1e-15)
    # ties collapse to one adjusted value regardless of input order
    tied = _step_up_adjusted([0.02, 0.02], harmonic=False)
    assert tied[0] == tied[1] == pytest.approx(0.02, abs=1e-15)
    # clipping at 1
    assert _step_up_adjusted([0.9, 1.0], harmonic=True) == [1.0, 1.0]


def test_paired_bh_selects_planted_head():
    scan, outcomes = _paired_scan(
        {
            "15:2": _vec(20, fixed=10, broken=0, base_correct=2),  # p = 2^-10
            "9:9": _vec(20, fixed=0, broken=0, base_correct=2),  # no discordant
            "7:7": _vec(20, fixed=0, broken=5, base_correct=2),  # only harm
        }
    )
    result = paired_bh_v1(scan, PAIRED_BH_V1_PARAMS, outcomes)
    assert result.main_heads == [(15, 2)]
    assert result.verdict["verdict"] == "localized"
    assert result.verdict["n_sig"] == 3
    assert set(result.decisions) == {"15:2", "9:9", "7:7"}
    top = result.decisions["15:2"]
    assert top["c_star"] == 1
    assert top["b"] == 10 and top["d"] == 0
    assert top["p_raw_min"] == 2.0**-10
    assert top["n_pos_coefs"] == 1
    assert top["p_head"] == 2.0**-10  # m = 1: Bonferroni is a no-op
    assert top["p_adjusted"] == pytest.approx(3 * 2.0**-10, abs=1e-18)  # rank 1 of 3
    assert top["gain"] == pytest.approx(0.5)
    assert top["is_main"] and top["main_reason"] == "significant_bh"
    assert top["per_coef"]["1"] == {
        "b": 10,
        "d": 0,
        "n_discordant": 10,
        "gain": 0.5,
        "acc": 0.6,
        "p_raw": 2.0**-10,
    }
    null = result.decisions["9:9"]
    assert null["p_raw_min"] == 1.0 and null["p_head"] == 1.0
    assert null["per_coef"]["1"]["n_discordant"] == 0
    assert null["flags"] == ["p_above_level", "gain_not_positive"]
    harmed = result.decisions["7:7"]
    assert harmed["b"] == 0 and harmed["d"] == 5
    assert harmed["p_raw_min"] == 1.0  # P[Bin(5, 1/2) >= 0]
    assert harmed["gain"] == pytest.approx(-0.25)
    assert not harmed["is_main"]
    # alignment record + fingerprint quote
    checks = result.verdict["outcomes_check"]
    assert checks["n_eval"] == 20
    assert checks["n_vectors_checked"] == 6  # 3 heads x (c=0 + one positive c)
    assert checks["content_sha256"] == "f" * 64
    assert result.verdict["test"]["across_heads"] == "bh"
    assert result.verdict["test"]["q"] == 0.05


def test_paired_family_hand_computed_selection_sets():
    """Six heads with one positive coefficient each and exact power-of-two
    p-values; every variant's selection set is hand-derived:

    p: A=2^-12, B=2^-8, C=2^-6, D=2^-5, E=2^-4, F=1 (no discordant).
    per_head a=.05: p <= .05           -> A B C D
    BH q=.05: adj = [6p/1, 6p/2, ...]  -> .00146 .0117 .0313 .0469 .075 1
                                       -> A B C D
    BY q=.05: BH * c(6)=2.45           -> .0036 .0287 .0766 .115 ...
                                       -> A B
    global Bonferroni a=.05: 6p        -> .0015 .0234 .0938 ...
                                       -> A B
    BH q=.01                           -> A
    """
    heads = {
        "1:1": _vec(4000, fixed=12, broken=0, base_correct=100),
        "2:2": _vec(4000, fixed=8, broken=0, base_correct=100),
        "3:3": _vec(4000, fixed=6, broken=0, base_correct=100),
        "4:4": _vec(4000, fixed=5, broken=0, base_correct=100),
        "5:5": _vec(4000, fixed=4, broken=0, base_correct=100),
        "6:6": _vec(4000, fixed=0, broken=0, base_correct=100),
    }
    scan, outcomes = _paired_scan(heads)
    per_head = paired_per_head_v1(scan, {"alpha": 0.05}, outcomes)
    assert per_head.main_heads == [(1, 1), (2, 2), (3, 3), (4, 4)]
    bh = paired_bh_v1(scan, {"q": 0.05}, outcomes)
    assert bh.main_heads == [(1, 1), (2, 2), (3, 3), (4, 4)]
    by = paired_by_v1(scan, {"q": 0.05}, outcomes)
    assert by.main_heads == [(1, 1), (2, 2)]
    bonferroni = paired_global_bonferroni_v1(scan, {"alpha": 0.05}, outcomes)
    assert bonferroni.main_heads == [(1, 1), (2, 2)]
    strict_bh = paired_bh_v1(scan, {"q": 0.01}, outcomes)
    assert strict_bh.main_heads == [(1, 1)]
    # hand-checked adjusted values (exact dyadic rationals scaled by ints)
    assert bh.decisions["3:3"]["p_adjusted"] == pytest.approx(
        6 * 2.0**-6 / 3, abs=1e-15
    )
    assert by.verdict["test"]["harmonic_c_n"] == pytest.approx(2.45, abs=1e-12)
    assert bonferroni.decisions["2:2"]["p_adjusted"] == pytest.approx(
        6 * 2.0**-8, abs=1e-15
    )
    # provable nesting at the same level: by ⊆ bh ⊆ per_head, bonferroni ⊆ bh
    # (bonferroni vs by are NOT nested in general — see the dedicated test)
    sets = {
        "bonferroni": set(bonferroni.main_heads),
        "by": set(by.main_heads),
        "bh": set(bh.main_heads),
        "per_head": set(per_head.main_heads),
    }
    assert sets["by"] <= sets["bh"] <= sets["per_head"]
    assert sets["bonferroni"] <= sets["bh"]
    assert set(strict_bh.main_heads) <= sets["bh"]


def test_paired_bonferroni_and_by_are_not_nested():
    """Counterexample pinning the docstring's non-nesting caveat: a rank-1
    head with p_head in (q/(N*c(N)), q/N] is Bonferroni-selected but
    BY-rejected (BY's rank-1 factor N*c(N) > Bonferroni's N). Here
    p = P[Bin(12,1/2) >= 10] = 79/4096 = 0.0193; with N=2, c(2)=1.5:
    Bonferroni 2p = 0.0386 <= 0.05 but BY 3p = 0.0579 > 0.05."""
    scan, outcomes = _paired_scan(
        {
            "1:1": _vec(200, fixed=10, broken=2, base_correct=50),
            "2:2": _vec(200, fixed=0, broken=0, base_correct=50),
        }
    )
    bonferroni = paired_global_bonferroni_v1(scan, {"alpha": 0.05}, outcomes)
    by = paired_by_v1(scan, {"q": 0.05}, outcomes)
    assert bonferroni.main_heads == [(1, 1)]
    assert by.main_heads == []
    assert bonferroni.decisions["1:1"]["p_head"] == 79 / 4096
    assert bonferroni.decisions["1:1"]["p_adjusted"] == pytest.approx(
        2 * 79 / 4096, abs=1e-15
    )
    assert by.decisions["1:1"]["p_adjusted"] == pytest.approx(3 * 79 / 4096, abs=1e-15)


def test_paired_across_head_step_consumes_bonferroni_head_p():
    """The across-head corrections must consume p_head = min(1, m * min_c p),
    NOT the uncorrected raw minimum (spec steps 4 -> 5). Head (1,1) has m=2
    positive coefficients with min raw p = 2^-6, so p_head = 2^-5 = 0.03125:
    at per-head alpha = 0.02 it must NOT be selected (the raw min 0.015625
    would pass), and through BH with a null companion (N=2, adjusted =
    2 * 0.03125 = 0.0625) it must NOT be selected at q = 0.04 (the raw min
    would give 0.03125 <= 0.04)."""
    strong = _vec(100, fixed=6, broken=0, base_correct=20)
    null_pair = _vec(100, fixed=0, broken=0, base_correct=20)
    scan, outcomes = _paired_scan(
        {
            "1:1": {0: strong[0], 1: strong[1], 2: strong[0]},
            "2:2": {0: null_pair[0], 1: null_pair[1], 2: null_pair[0]},
        }
    )
    tight = paired_per_head_v1(scan, {"alpha": 0.02}, outcomes)
    assert tight.decisions["1:1"]["p_raw_min"] == 2.0**-6
    assert tight.decisions["1:1"]["p_head"] == 2.0**-5
    assert tight.main_heads == []
    loose = paired_per_head_v1(scan, {"alpha": 0.05}, outcomes)
    assert loose.main_heads == [(1, 1)]
    bh = paired_bh_v1(scan, {"q": 0.04}, outcomes)
    assert bh.decisions["1:1"]["p_adjusted"] == pytest.approx(2.0**-4, abs=1e-15)
    assert bh.main_heads == []


def test_paired_p_caps_bind_at_one():
    """min(1, .) caps: a no-discordant head with m=3 would record
    p_head = 3.0 uncapped; the global-Bonferroni adjustment of p_head = 1
    would record 2.0 uncapped."""
    null_pair = _vec(30, fixed=0, broken=0, base_correct=10)
    strong = _vec(30, fixed=8, broken=0, base_correct=10)
    scan, outcomes = _paired_scan(
        {
            "1:1": {
                0: null_pair[0],
                1: null_pair[1],
                2: null_pair[0],
                3: null_pair[0],
            },
            "2:2": {0: strong[0], 1: strong[1], 2: strong[0], 3: strong[0]},
        }
    )
    result = paired_global_bonferroni_v1(scan, {"alpha": 0.05}, outcomes)
    null_decision = result.decisions["1:1"]
    assert null_decision["n_pos_coefs"] == 3
    assert null_decision["p_raw_min"] == 1.0
    assert null_decision["p_head"] == 1.0  # capped, not 3.0
    assert null_decision["p_adjusted"] == 1.0  # capped, not 2.0
    bh = paired_bh_v1(scan, {"q": 0.05}, outcomes)
    assert bh.decisions["1:1"]["p_adjusted"] == 1.0


def test_paired_selection_boundary_is_inclusive():
    """Spec: select iff adjusted p <= level. p = P[Bin(2,1/2) >= 2] = 0.25
    exactly; at level exactly 0.25 the head must be selected."""
    scan, outcomes = _paired_scan({"1:1": _vec(16, fixed=2, broken=0, base_correct=4)})
    result = paired_per_head_v1(scan, {"alpha": 0.25}, outcomes)
    assert result.decisions["1:1"]["p_adjusted"] == 0.25
    assert result.main_heads == [(1, 1)]
    bh = paired_bh_v1(scan, {"q": 0.25}, outcomes)
    assert bh.main_heads == [(1, 1)]


def test_paired_c_star_tie_breaks():
    y0 = [0] * 10 + [1] * 10 + [0] * 10
    # c=1: b=0, d=0 (p=1, net 0); c=2: b=0, d=5 (p=1, net -5)
    # -> equal p, LARGER net gain wins: c_star = 1
    scan, outcomes = _paired_scan(
        {
            "1:1": {
                0: y0,
                1: list(y0),
                2: [0] * 10 + [1] * 5 + [0] * 5 + [0] * 10,
            }
        }
    )
    result = paired_per_head_v1(scan, {"alpha": 0.05}, outcomes)
    assert result.decisions["1:1"]["c_star"] == 1
    assert result.decisions["1:1"]["per_coef"]["2"]["d"] == 5

    # same two conditions with the coefficients SWAPPED: the larger net gain
    # now sits at the LARGER coefficient and must override the smaller-c
    # preference (gain is the first tie-break, smaller c only second)
    scan_swapped, outcomes_swapped = _paired_scan(
        {
            "1:1": {
                0: y0,
                1: [0] * 10 + [1] * 5 + [0] * 5 + [0] * 10,
                2: list(y0),
            }
        }
    )
    swapped = paired_per_head_v1(scan_swapped, {"alpha": 0.05}, outcomes_swapped)
    assert swapped.decisions["1:1"]["c_star"] == 2

    # equal p AND equal net gain (identical vectors) -> smaller c wins
    y_fix = [1] * 2 + [1] * 10 + [0] * 8 + [0] * 10
    scan2, outcomes2 = _paired_scan({"1:1": {0: y0, 1: y_fix, 2: list(y_fix)}})
    result2 = paired_per_head_v1(scan2, {"alpha": 0.05}, outcomes2)
    decision = result2.decisions["1:1"]
    assert decision["per_coef"]["1"] == decision["per_coef"]["2"]
    assert decision["c_star"] == 1


def test_paired_gain_gate_is_enforced():
    # b=1, d=2: p = P[Bin(3,1/2) >= 1] = 7/8 <= 0.9, but the net gain is
    # negative -> the positive-gain gate must block the selection. (For any
    # level < 0.5 the gate is provably redundant — this guards the contract
    # at permissive levels.)
    scan, outcomes = _paired_scan(
        {
            "1:1": {
                0: [1, 1, 0, 0, 0, 0, 0, 0],
                1: [0, 0, 1, 0, 0, 0, 0, 0],
            }
        }
    )
    result = paired_per_head_v1(scan, {"alpha": 0.9}, outcomes)
    decision = result.decisions["1:1"]
    assert decision["b"] == 1 and decision["d"] == 2
    assert decision["p_head"] == 7 / 8
    assert decision["p_adjusted"] <= 0.9
    assert not decision["is_main"]
    assert decision["flags"] == ["gain_not_positive"]
    assert result.main_heads == []


def test_paired_no_discordant_head_is_never_selected():
    scan, outcomes = _paired_scan({"1:1": _vec(50, fixed=0, broken=0, base_correct=25)})
    for selector, params in (
        (paired_bh_v1, {"q": 0.05}),
        (paired_per_head_v1, {"alpha": 0.05}),
        (paired_by_v1, {"q": 0.05}),
        (paired_global_bonferroni_v1, {"alpha": 0.05}),
    ):
        result = selector(scan, params, outcomes)
        assert result.main_heads == []
        assert result.verdict["verdict"] == "none"
        decision = result.decisions["1:1"]
        assert decision["p_raw_min"] == 1.0
        assert decision["p_head"] == 1.0
        assert decision["gain"] == 0.0


def test_paired_non_contiguous_grid():
    pair = _vec(40, fixed=9, broken=0, base_correct=5)
    # positive grid {2, 5} (non-contiguous, no c=1): m = 2
    scan, outcomes = _paired_scan(
        {"1:1": {0: pair[0], 2: pair[1], 5: pair[0]}}  # c=5 has no discordant
    )
    result = paired_per_head_v1(scan, {"alpha": 0.05}, outcomes)
    decision = result.decisions["1:1"]
    assert decision["n_pos_coefs"] == 2
    assert sorted(decision["per_coef"]) == ["2", "5"]
    assert decision["c_star"] == 2
    assert decision["p_raw_min"] == 2.0**-9
    assert decision["p_head"] == 2 * 2.0**-9  # Bonferroni over m=2
    assert result.main_heads == [(1, 1)]


def test_paired_refusals():
    good = _vec(20, fixed=5, broken=0, base_correct=5)
    # missing c=0
    scan, outcomes = _paired_scan({"1:1": {1: good[1], 2: good[0]}})
    with pytest.raises(SelectorError, match="needs c=0"):
        paired_bh_v1(scan, PAIRED_BH_V1_PARAMS, outcomes)
    # no positive coefficient
    scan, outcomes = _paired_scan({"1:1": {0: good[0]}})
    with pytest.raises(SelectorError, match="at least one positive"):
        paired_bh_v1(scan, PAIRED_BH_V1_PARAMS, outcomes)
    # negative coefficients are out of the family's domain
    scan, outcomes = _paired_scan({"1:1": {-1: good[0], 0: good[0], 1: good[1]}})
    with pytest.raises(SelectorError, match="c >= 0 only"):
        paired_bh_v1(scan, PAIRED_BH_V1_PARAMS, outcomes)
    # outcomes not supplied at all
    scan, outcomes = _paired_scan({"1:1": good})
    with pytest.raises(SelectorError, match="per-example outcomes"):
        paired_bh_v1(scan, PAIRED_BH_V1_PARAMS)
    # a vector is missing
    broken = dict(outcomes)
    del broken["1:1:1"]
    with pytest.raises(SelectorError, match="missing the vector '1:1:1'"):
        paired_bh_v1(scan, PAIRED_BH_V1_PARAMS, broken)
    # a vector is short
    truncated = dict(outcomes)
    truncated["1:1:1"] = outcomes["1:1:1"][:-1]
    with pytest.raises(SelectorError, match="entries, expected"):
        paired_bh_v1(scan, PAIRED_BH_V1_PARAMS, truncated)
    # a vector is non-binary
    corrupt = dict(outcomes)
    corrupt["1:1:1"] = bytes([2]) + outcomes["1:1:1"][1:]
    with pytest.raises(SelectorError, match="non-binary"):
        paired_bh_v1(scan, PAIRED_BH_V1_PARAMS, corrupt)
    # the curve accuracy does not match the vector (misalignment)
    tampered, outcomes_ok = _paired_scan({"1:1": good})
    tampered["curves"]["1:1"]["1"] += 0.05
    with pytest.raises(SelectorError, match="does not reproduce"):
        paired_bh_v1(tampered, PAIRED_BH_V1_PARAMS, outcomes_ok)
    # a NaN curve point is also a mismatch (NaN != anything)
    nan_scan, outcomes_nan = _paired_scan({"1:1": good})
    nan_scan["curves"]["1:1"]["1"] = float("nan")
    with pytest.raises(SelectorError, match="does not reproduce"):
        paired_bh_v1(nan_scan, PAIRED_BH_V1_PARAMS, outcomes_nan)
    # n_eval missing from the scan
    no_n, outcomes_n = _paired_scan({"1:1": good})
    del no_n["n_eval_examples_per_head_per_c"]
    with pytest.raises(SelectorError, match="n_eval_examples_per_head_per_c"):
        paired_bh_v1(no_n, PAIRED_BH_V1_PARAMS, outcomes_n)


def test_paired_param_validation():
    scan, outcomes = _paired_scan({"1:1": _vec(20, 5, 0, 5)})
    with pytest.raises(SelectorError, match="exactly params"):
        paired_bh_v1(scan, {"alpha": 0.05}, outcomes)  # q, not alpha
    with pytest.raises(SelectorError, match="exactly params"):
        paired_per_head_v1(scan, {"q": 0.05}, outcomes)  # alpha, not q
    with pytest.raises(SelectorError, match="exactly params"):
        paired_bh_v1(scan, {"q": 0.05, "bonus": 1.0}, outcomes)
    with pytest.raises(SelectorError, match="exactly params"):
        paired_bh_v1(scan, {}, outcomes)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(SelectorError, match="must be in"):
            paired_bh_v1(scan, {"q": bad}, outcomes)
        with pytest.raises(SelectorError, match="must be in"):
            paired_global_bonferroni_v1(scan, {"alpha": bad}, outcomes)


def test_paired_ordering_and_determinism():
    pair = _vec(30, fixed=6, broken=0, base_correct=4)
    scan, outcomes = _paired_scan({"5:1": pair, "3:2": pair})
    for selector, params in (
        (paired_bh_v1, PAIRED_BH_V1_PARAMS),
        (paired_per_head_v1, PAIRED_PER_HEAD_V1_PARAMS),
        (paired_by_v1, PAIRED_BY_V1_PARAMS),
        (paired_global_bonferroni_v1, PAIRED_GLOBAL_BONFERRONI_V1_PARAMS),
    ):
        first = selector(scan, params, outcomes)
        second = selector(scan, params, outcomes)
        assert first == second
        # identical statistics -> deterministic (layer, head) order
        assert first.main_heads == [(3, 2), (5, 1)]
