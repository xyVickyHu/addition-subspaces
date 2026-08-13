"""Substep 5 — main-head selectors: pure CPU algorithms over a saved scan.

Selectors are named + versioned; a fixed scan must produce ONE exact selector
result. ``select-main`` never loads the model, never re-runs the scan, and
never touches the matrix — changing the selector reuses all preceding
artifacts.

The implementations below are FAITHFUL PORTS of the legacy code, verified
against committed fixtures captured from the development repo
(legacy sources pinned by sha256 in ``tests/fixtures/
selector_expected_repro_v1.json``):

- ``unified`` v1 — the adopted standard (2026-07-05, β recalibrated to 1/4 on
  2026-07-13; derivation in the internal main-head-standard study).
  Port of ``select_main_heads_unified``. Any corrected minor-head statistic
  derived from paired per-example outcomes will be a NEW selector
  ("unified-v2-paired"), never a silent change to v1.
- ``recovery_weak`` v1 — the released weak rule (port of
  ``select_main_heads_by_recovery_weak``; the two historically shadowed
  definitions share one body — parameters are always passed explicitly here).
- ``pin`` v1 — explicit head list (paper-exact pinning).

``largest_gap`` v1 is NOT a port (added 2026-07-24): the parameter-free
peak-accuracy analogue of the significant-side ``largest_gap`` method
(``subspaces.step1.significant.largest_gap_threshold``), introduced as a comparison
method against ``unified`` v1.

``above_ablation`` v1 / ``above_ablation_zero`` v1 are NOT ports (added
2026-07-28): a head is main iff its peak recovery accuracy exceeds the
mean-ablated (c=0, "corrupted") baseline beyond a noise region. The ``_zero``
variant sets the noise region to zero (strict ``gain > 0``); the banded
variant derives the region from the corrupted accuracy itself — the
(1-alpha) quantile of the max over the non-zero coefficient grid of iid
Binomial(n_eval, pooled_c0) draws, where pooled_c0 is the mean of the
per-head c=0 accuracies. NOTE the c=0 semantics (subspaces.step1.recovery): each
head's c=0 point is a per-head LEAVE-ONE-OUT condition (that head zeroed,
all other significant heads at their means), evaluated deterministically on
shared examples — the per-head c0 values differ systematically (by
-mean_z[l,h]), not stochastically, and on the canonical addition scan their
cross-head spread is SMALLER than the binomial SE (the spread is
scan-dependent: multi-SE on e.g. the abstractive and qa scans, where which
head attains the extremes is NOT noise-level). Pooling is a stability
choice: it anchors
one scan-level corrupted level (~= baselines.mean_fv_acc) instead of
letting a head lower its own bar via its own low c0 value. Comparison
methods, not defaults; like ``largest_gap`` they have NO existence gate.

``compare_zero`` v1 is NOT a port (added 2026-08-03): a head is main iff its
peak recovery accuracy (max over the scanned coefficient grid, c=0 included)
STRICTLY exceeds the largest per-head c=0 accuracy across all scanned heads
— the empirical envelope of the leave-one-out corrupted levels, with no
distributional null and no ``n_eval``. Always a subset of the
``above_ablation_zero`` selection (max c0 >= own c0). NOTE the threshold is
grid-size-independent but the SELECTION is not: the peak side is an
uncorrected max over the grid, so on scans whose c0 spread is small
relative to the binomial SE the envelope adds little beyond a head's own
c0 draw and the variant over-selects like ``above_ablation_zero``; it
concentrates only where the c0 spread is large (verified 2026-08-03,
adversarial stats review of the 33-scan backfill). Comparison method, not
a default; like the rest of the family it has NO existence gate.

The ``paired_*`` family (v1, added 2026-08-03) is NOT a port: the first
selectors to consume the scan's PER-EXAMPLE outcomes (``scan_outcomes.npz``)
instead of only the aggregate accuracy curves. For each head and each
positive coefficient c, the c and c=0 outcome vectors are paired on the
shared evaluation examples (``subspaces.step1.recovery`` evaluates every (head, c)
point on the same examples in the same order, deterministically); the test
statistic is the discordant split — b_c examples fixed (wrong at c=0, right
at c) vs d_c broken — and the p-value is the exact one-sided McNemar tail
P[Binomial(b_c+d_c, 1/2) >= b_c] (p=1 when b_c+d_c=0). Pairing removes the
shared example-difficulty noise that the curve-level selectors must model
distributionally. Within a head, the coefficient grid is Bonferroni-corrected
(p_head = min(1, m * min_c p_c), m = positive coefficients present — valid
under arbitrary dependence, which matters because all y_c share y_0). Across
heads the four variants differ only in the multiplicity step over the
head-level p-values:

- ``paired_bh`` — Benjamini-Hochberg step-up at FDR q (the primary method;
  exactly valid under PRDS positive dependence, which is plausible but not
  guaranteed for heads sharing evaluation examples — hence the BY variant);
- ``paired_by`` — Benjamini-Yekutieli (BH with the harmonic factor c(N));
  valid under ARBITRARY dependence; the conservative sensitivity check;
- ``paired_per_head`` — no across-head correction (head-level level alpha);
- ``paired_global_bonferroni`` — family-wise Bonferroni across heads.

Selection additionally requires the observed gain at c_star to be positive
(b > d at c_star; the one-sided tail already points that way, so the gate
only bites in degenerate cases). c_star is the coefficient with the smallest
raw p; ties break by larger net gain b_c - d_c, then smaller c. Provable
nesting at the same level: ``paired_by`` ⊆ ``paired_bh`` ⊆
``paired_per_head``, and ``paired_global_bonferroni`` ⊆ ``paired_bh``.
``paired_global_bonferroni`` and ``paired_by`` are NOT nested in general:
BY's rank-r step-up factor N*c(N)/r exceeds Bonferroni's N at ranks below
c(N) ~ ln N, so a low-rank head with p_head in (q/(N*c(N)), q/N] can be
Bonferroni-selected yet BY-rejected (adversarial review 2026-08-04 —
observed nesting of the two on a given scan is empirical, not guaranteed).
Comparison methods, not defaults;
like the rest of the family they have NO existence gate. These selectors
REQUIRE the outcomes npz (``OUTCOME_SELECTORS``): the pipeline loads it
digest-verified and the selector re-derives every curve accuracy from the
vectors, refusing on any misalignment.

Head-scan manifest fields consumed: ``curves`` ({"L:H": {c: acc}} — key order
preserves the scan's head order, which carries the legacy tie-break
semantics), ``baselines.clean_acc``, ``baselines.full_significant_acc``,
``n_eval_examples_per_head_per_c``; the paired family additionally consumes
``outcomes.content_sha256`` (quoted into its verdict) and the outcome
vectors themselves.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

MAIN_HEADS_SCHEMA_VERSION = 1

# Historical weak-selector parameterizations (recorded explicitly; no defaults).
WEAK_RELEASE_PARAMS = {"rel_floor": 0.2, "abs_floor": 0.01, "eps": 0.01}
WEAK_SHADOW_DEFAULTS = {"rel_floor": 0.2, "abs_floor": 0.05, "eps": 0.03}

# The adopted unified standard constants (fixed; do not tune per task).
# floor 0.12 -> 0.0 on 2026-08-04 (owner decision after the floor ablation,
# the step-1 methods notes, Substep 5 "unified floor ablation"): the existence
# gate was inert on every saved registry scan (min leader gain 0.164) and
# only ever fired on no-material-head scans now relegated to legacy;
# clean_min stays the sole scan-level gate. Historical floor=0.12 nodes
# remain valid — params enter node identity, so the two defaults coexist.
UNIFIED_V1_PARAMS = {"beta": 0.25, "floor": 0.0, "clean_min": 0.25, "minor_z": 3.5}

# above_ablation defaults: the conventional per-head test level. At alpha=0.05
# the per-head form (scanwise=0.0) is markedly more replicate-stable than the
# family-wise form on the n-shot near-replicate scans (mean pairwise Jaccard
# 0.943 vs 0.895 — the family-wise threshold lands mid-cluster in the weak
# tail); scanwise=1.0 with alpha=0.01 is the strict variant (0.952, smaller
# sets). Pass floats — params enter the node identity.
ABOVE_ABLATION_V1_PARAMS = {"alpha": 0.05, "scanwise": 0.0}

# paired family defaults: q = target FDR (BH/BY step-up), alpha = test level
# (per-head / family-wise Bonferroni). Pass floats — params enter the node
# identity. paired_bh q=0.05 is the family's primary method.
PAIRED_BH_V1_PARAMS = {"q": 0.05}
PAIRED_PER_HEAD_V1_PARAMS = {"alpha": 0.05}
PAIRED_BY_V1_PARAMS = {"q": 0.05}
PAIRED_GLOBAL_BONFERRONI_V1_PARAMS = {"alpha": 0.05}

# Exact parameter schemas per (name, version) — configuration validation uses
# these (exact-set match) so composed runs are checked at load, not only CLIs.
SELECTOR_PARAM_SCHEMAS: dict[tuple[str, int], frozenset[str]] = {
    ("unified", 1): frozenset(UNIFIED_V1_PARAMS),
    ("recovery_weak", 1): frozenset(WEAK_RELEASE_PARAMS),
    ("pin", 1): frozenset(),
    ("largest_gap", 1): frozenset(),
    ("above_ablation", 1): frozenset(ABOVE_ABLATION_V1_PARAMS),
    ("above_ablation_zero", 1): frozenset(),
    ("compare_zero", 1): frozenset(),
    ("paired_bh", 1): frozenset(PAIRED_BH_V1_PARAMS),
    ("paired_per_head", 1): frozenset(PAIRED_PER_HEAD_V1_PARAMS),
    ("paired_by", 1): frozenset(PAIRED_BY_V1_PARAMS),
    ("paired_global_bonferroni", 1): frozenset(PAIRED_GLOBAL_BONFERRONI_V1_PARAMS),
}

# Selectors that consume the scan's per-example outcome vectors: the caller
# (``pipeline.apply_selector``) must load ``scan_outcomes.npz``, verify its
# content digest against the scan manifest, and pass the vectors as the third
# selector argument. Curve-only selectors are never handed outcomes.
OUTCOME_SELECTORS: frozenset[tuple[str, int]] = frozenset(
    {
        ("paired_bh", 1),
        ("paired_per_head", 1),
        ("paired_by", 1),
        ("paired_global_bonferroni", 1),
    }
)

Head = tuple[int, int]


@dataclass
class SelectorResult:
    selector_name: str
    selector_version: int
    params: dict
    main_heads: list[Head]
    minor_heads: list[Head]
    decisions: dict = field(default_factory=dict)  # keyed "L:H"
    verdict: dict = field(default_factory=dict)


class SelectorError(ValueError):
    pass


def _parse_curves(scan: dict) -> dict[Head, dict]:
    curves_raw = scan.get("curves")
    if not curves_raw:
        raise SelectorError("head_scan manifest has no 'curves'")
    curves: dict[Head, dict] = {}
    for key, curve in curves_raw.items():
        layer_str, head_str = str(key).split(":")
        curves[(int(layer_str), int(head_str))] = curve
    return curves


def _keyed(decisions: dict[Head, dict]) -> dict[str, dict]:
    return {
        f"{layer_idx}:{head_idx}": value
        for (layer_idx, head_idx), value in decisions.items()
    }


def _curve_stats(curve: dict) -> dict:
    """Peak statistics of one dose-response curve (legacy-identical)."""
    cs = sorted(curve.keys(), key=lambda c: int(c))
    accs = [float(curve[c]) for c in cs]
    i_star = int(max(range(len(accs)), key=lambda i: accs[i]))
    return {
        "cs": cs,
        "accs": accs,
        "i_star": i_star,
        "c_star": int(cs[i_star]),
        "acc_at_c_star": accs[i_star],
        "acc_at_c0": accs[0],
        "gain": accs[i_star] - accs[0],
    }


def unified_v1(scan: dict, params: dict) -> SelectorResult:
    """Faithful port of the legacy ``select_main_heads_unified`` (heads.py).

    A scan has main heads iff the model performs the task (clean_acc >=
    clean_min) and some single head materially recovers it (best peak gain >=
    floor); then a significant head is MAIN iff its peak recovery gain is at
    least beta of the scan's best gain, ordered by gain descending. Heads real
    beyond noise (z >= minor_z) but below the main bar are MINOR.
    """
    beta = float(params["beta"])
    floor = float(params["floor"])
    clean_min = float(params["clean_min"])
    minor_z = float(params["minor_z"])
    clean_acc = float(scan["baselines"]["clean_acc"])
    n_eval = scan.get("n_eval_examples_per_head_per_c")

    curves = _parse_curves(scan)
    stats: dict[Head, dict] = {}
    for pos, curve in curves.items():
        s = _curve_stats(curve)
        stats[pos] = {
            "c_star": s["c_star"],
            "acc_at_c_star": s["acc_at_c_star"],
            "acc_at_c0": s["acc_at_c0"],
            "gain": s["gain"],
        }

    leader = (
        max(stats, key=lambda p: (stats[p]["gain"], stats[p]["acc_at_c_star"]))
        if stats
        else None
    )
    g_best = stats[leader]["gain"] if leader else 0.0
    applicable = clean_acc >= clean_min
    material = g_best >= floor
    bar = beta * g_best if (applicable and material) else float("inf")

    def _z(s: dict) -> float | None:
        if not n_eval:
            return None
        a0, a1 = s["acc_at_c0"], s["acc_at_c_star"]
        se = (max(a0 * (1 - a0) + a1 * (1 - a1), 0.0) / n_eval) ** 0.5
        return float(s["gain"] / max(se, 1.0 / n_eval))

    decisions: dict[Head, dict] = {}
    main_heads: list[Head] = []
    minor_heads: list[Head] = []
    for pos, s in stats.items():
        is_main = bool(applicable and material and s["gain"] >= bar)
        z = _z(s)
        is_minor = bool(
            applicable
            and not is_main
            and z is not None
            and z >= minor_z
            and s["gain"] > 0
        )
        flags: list[str] = []
        if not applicable:
            flags.append("task_below_floor")
        elif not material:
            flags.append("no_material_head")
        elif not is_main:
            flags.append("gain_below_bar")
        decisions[pos] = {
            **s,
            "gain": float(s["gain"]),
            "rel_recovery": float(s["gain"] / g_best) if g_best > 0 else None,
            "bar": float(bar) if bar != float("inf") else None,
            "z": z,
            "steady": None,  # no steadiness gate in the unified standard
            "is_main": is_main,
            "is_minor": is_minor,
            "main_reason": "unified_leader_relative" if is_main else None,
            "flags": flags,
        }
        if is_main:
            main_heads.append(pos)
        elif is_minor:
            minor_heads.append(pos)

    def order_key(p: Head) -> tuple:
        return (-decisions[p]["gain"], -decisions[p]["acc_at_c_star"], p[0], p[1])

    main_heads.sort(key=order_key)
    minor_heads.sort(key=order_key)

    if not applicable:
        verdict_label, none_reason = "none", "task_below_floor"
    elif not material:
        verdict_label, none_reason = "none", "no_material_head"
    else:
        verdict_label = "localized" if len(main_heads) <= 5 else "distributed"
        none_reason = None
    verdict = {
        "verdict": verdict_label,
        "none_reason": none_reason,
        "n_selected": len(main_heads),
        "n_sig": len(stats),
        "leader": list(leader) if leader else None,
        "leader_gain": float(g_best),
        "leader_near_floor": bool(g_best < 1.5 * floor),
        "minor_heads": [[list(p), round(stats[p]["gain"], 4)] for p in minor_heads],
        "params": {
            "beta": beta,
            "floor": floor,
            "clean_min": clean_min,
            "minor_z": minor_z,
            "n_eval": n_eval,
            "clean_acc_ref": float(clean_acc),
        },
    }
    return SelectorResult(
        selector_name="unified",
        selector_version=1,
        params=dict(params),
        main_heads=main_heads,
        minor_heads=minor_heads,
        decisions=_keyed(decisions),
        verdict=verdict,
    )


def recovery_weak_v1(scan: dict, params: dict) -> SelectorResult:
    """Faithful port of ``select_main_heads_by_recovery_weak`` (released rule).

    Main iff: (1) the peak comes from scaling up (c* > 0); (2) no drop greater
    than eps on the way to the peak; (3) gain >= max(abs_floor, rel_floor *
    headroom) where headroom = A_sig - acc0. Ordered by peak accuracy, then
    gain, then layer/head. All parameters are explicit (the legacy code had
    two shadowed definitions differing only in defaults).
    """
    rel_floor = float(params["rel_floor"])
    abs_floor = float(params["abs_floor"])
    eps = float(params["eps"])
    a_sig = float(scan["baselines"]["full_significant_acc"])

    curves = _parse_curves(scan)
    main_heads: list[Head] = []
    decisions: dict[Head, dict] = {}
    for pos, curve in curves.items():
        s = _curve_stats(curve)
        seg = s["accs"][: s["i_star"] + 1]
        max_drop = max((seg[i] - seg[i + 1] for i in range(len(seg) - 1)), default=0.0)
        steady = max_drop <= eps
        headroom = a_sig - s["acc_at_c0"]
        bar = max(abs_floor, rel_floor * headroom) if headroom > 0 else abs_floor
        rel = (s["gain"] / headroom) if abs(headroom) > 1e-9 else float("nan")
        is_main = bool(s["c_star"] != 0 and steady and s["gain"] >= bar)

        flags: list[str] = []
        if s["c_star"] == 0:
            flags.append("peak_at_c=0_leave_one_out")
        if not steady:
            flags.append("not_steady_to_peak")
        if s["gain"] < bar:
            flags.append("gain_below_bar")

        decisions[pos] = {
            "c_star": s["c_star"],
            "acc_at_c_star": s["acc_at_c_star"],
            "acc_at_c0": s["acc_at_c0"],
            "gain": float(s["gain"]),
            "rel_recovery": float(rel) if rel == rel else None,  # NaN -> None
            "headroom_vs_sumSig": float(headroom),
            "bar": float(bar),
            "max_drop_to_peak": float(max_drop),
            "steady": bool(steady),
            "is_main": is_main,
            "main_reason": "steady_scaling_recovery" if is_main else None,
            "flags": flags,
        }
        if is_main:
            main_heads.append(pos)

    main_heads.sort(
        key=lambda p: (
            -decisions[p]["acc_at_c_star"],
            -decisions[p]["gain"],
            p[0],
            p[1],
        )
    )
    return SelectorResult(
        selector_name="recovery_weak",
        selector_version=1,
        params=dict(params),
        main_heads=main_heads,
        minor_heads=[],
        decisions=_keyed(decisions),
        verdict={"verdict": "weak_rule", "n_selected": len(main_heads)},
    )


def pin_v1(scan: dict, params: dict) -> SelectorResult:
    """Explicit head list (paper-exact pinning); heads must be in the scan."""
    heads_raw = params.get("heads")
    if not heads_raw:
        raise SelectorError("pin selector requires params['heads']")
    pinned = [(int(layer_idx), int(head_idx)) for layer_idx, head_idx in heads_raw]
    scanned = set(_parse_curves(scan))
    missing = [p for p in pinned if p not in scanned]
    if missing:
        raise SelectorError(
            f"pinned head(s) not in the scan's significant set: {missing}"
        )
    decisions = {
        f"{layer_idx}:{head_idx}": {"is_main": True, "main_reason": "pinned"}
        for layer_idx, head_idx in pinned
    }
    return SelectorResult(
        selector_name="pin",
        selector_version=1,
        params={"heads": [list(p) for p in pinned]},
        main_heads=pinned,
        minor_heads=[],
        decisions=decisions,
        verdict={"verdict": "pinned", "n_selected": len(pinned)},
    )


def largest_gap_v1(scan: dict, params: dict) -> SelectorResult:
    """Cut the descending peak-accuracy curve at its largest consecutive drop.

    Parameter-free analogue of the significant-side ``largest_gap`` method
    (``subspaces.step1.significant.largest_gap_threshold``), applied to each
    significant head's peak recovery accuracy ``acc_at_c_star`` (max over the
    scanned coefficient grid, c=0 included) instead of the |coefficient|
    curve. Peaks are sorted descending; the cut is the largest drop between
    consecutive values; on an exact tie between drops the highest cut (fewest
    heads) wins deterministically. The realized threshold is the value at the
    drop's LOWER edge and membership is strictly above it — ties at the upper
    edge stay in, ties at the lower edge stay out. Refuses scans with fewer
    than two heads, NaN peaks (reachable only when acc(c=0) is NaN — a NaN
    elsewhere never displaces a finite peak), and all-equal peaks (no drop).

    Unlike ``unified`` v1 there is NO existence gate (no clean_min/floor), so
    at least one head is always selected — even on scans where the task is
    dead; scan-level applicability is the caller's judgement. The ``verdict``
    records the winning drop and the runner-up drop, because a near-tie
    between candidate cuts is scientifically material.
    """
    curves = _parse_curves(scan)
    if len(curves) < 2:
        raise SelectorError(
            "largest_gap is undefined on a scan with fewer than two heads "
            f"(got {len(curves)})"
        )
    stats: dict[Head, dict] = {}
    for pos, curve in curves.items():
        s = _curve_stats(curve)
        stats[pos] = {
            "c_star": s["c_star"],
            "acc_at_c_star": s["acc_at_c_star"],
            "acc_at_c0": s["acc_at_c0"],
            "gain": s["gain"],
        }
    nan_peaks = sorted(
        pos for pos, s in stats.items() if s["acc_at_c_star"] != s["acc_at_c_star"]
    )
    if nan_peaks:
        raise SelectorError(
            f"largest_gap refuses NaN peak accuracies (heads: {nan_peaks})"
        )
    ordered = sorted(
        stats,
        key=lambda p: (-stats[p]["acc_at_c_star"], -stats[p]["gain"], p[0], p[1]),
    )
    values = [stats[p]["acc_at_c_star"] for p in ordered]
    drops = [values[i] - values[i + 1] for i in range(len(values) - 1)]
    # max() keeps the first of tied drops = the highest cut (fewest heads).
    best = max(range(len(drops)), key=drops.__getitem__)
    if drops[best] <= 0.0:
        raise SelectorError(
            "largest_gap is undefined: all peak accuracies are equal "
            f"({values[0]!r}); the sorted curve has no drop"
        )
    threshold = values[best + 1]
    runner_up = None
    if len(drops) > 1:
        remaining = list(drops)
        remaining[best] = float("-inf")
        second = max(range(len(remaining)), key=remaining.__getitem__)
        runner_up = {
            "upper": float(values[second]),
            "lower": float(values[second + 1]),
            "drop": float(drops[second]),
            "rank_of_upper": second + 1,
        }
    gap = {
        "upper": float(values[best]),
        "lower": float(values[best + 1]),
        "drop": float(drops[best]),
        "rank_of_upper": best + 1,
        "n_selected": best + 1,
        "runner_up": runner_up,
    }

    decisions: dict[Head, dict] = {}
    main_heads: list[Head] = []
    for pos in ordered:
        s = stats[pos]
        is_main = s["acc_at_c_star"] > threshold
        decisions[pos] = {
            "c_star": s["c_star"],
            "acc_at_c_star": float(s["acc_at_c_star"]),
            "acc_at_c0": float(s["acc_at_c0"]),
            "gain": float(s["gain"]),
            "threshold": float(threshold),
            "is_main": bool(is_main),
            "main_reason": "above_largest_gap" if is_main else None,
            "flags": [] if is_main else ["below_largest_gap"],
        }
        if is_main:
            main_heads.append(pos)

    verdict = {
        "verdict": "localized" if len(main_heads) <= 5 else "distributed",
        "n_selected": len(main_heads),
        "n_sig": len(stats),
        "threshold": float(threshold),
        "gap": gap,
        "order": "acc_at_c_star desc",
    }
    return SelectorResult(
        selector_name="largest_gap",
        selector_version=1,
        params=dict(params),
        main_heads=main_heads,
        minor_heads=[],
        decisions=_keyed(decisions),
        verdict=verdict,
    )


def _binomial_quantile(n_trials: int, p_success: float, quantile: float) -> int:
    """Smallest k with Binomial(n, p) CDF(k) >= quantile (exact, log-space pmf).

    Terms whose log-pmf underflows float range contribute 0 to the running
    CDF, which only matters far in the negligible lower tail; accumulation
    error is ~1e-11 over n ~ 10^4, far below any quantile decision margin
    used here.
    """
    if p_success <= 0.0:
        return 0
    if p_success >= 1.0:
        return n_trials
    log_p = math.log(p_success)
    log_1mp = math.log1p(-p_success)
    log_nfact = math.lgamma(n_trials + 1)
    cdf = 0.0
    for k in range(n_trials + 1):
        log_pmf = (
            log_nfact
            - math.lgamma(k + 1)
            - math.lgamma(n_trials - k + 1)
            + k * log_p
            + (n_trials - k) * log_1mp
        )
        cdf += math.exp(log_pmf) if log_pmf > -745.0 else 0.0
        if cdf >= quantile:
            return k
    return n_trials


def _peak_vs_c0_stats(scan: dict, label: str) -> dict[Head, dict]:
    """Shared per-head peak stats + refusals for the selectors that compare
    peak recovery against c=0 baselines (above_ablation family,
    compare_zero). A NaN at c=0 always surfaces as a NaN peak (first-index
    argmax; a NaN elsewhere never displaces a finite value), so the NaN-peak
    refusal also guards every consumer of ``acc_at_c0``."""
    curves = _parse_curves(scan)
    stats: dict[Head, dict] = {}
    for pos, curve in curves.items():
        s = _curve_stats(curve)
        if len(s["accs"]) < 2 or int(s["cs"][0]) != 0:
            raise SelectorError(
                f"{label} needs c=0 plus at least one non-zero "
                f"coefficient per head (head {pos} has grid {s['cs']})"
            )
        stats[pos] = {
            "c_star": s["c_star"],
            "acc_at_c_star": s["acc_at_c_star"],
            "acc_at_c0": s["acc_at_c0"],
            "gain": s["gain"],
            "n_nonzero_c": len(s["accs"]) - 1,
        }
    nan_peaks = sorted(
        pos for pos, s in stats.items() if s["acc_at_c_star"] != s["acc_at_c_star"]
    )
    if nan_peaks:
        raise SelectorError(f"{label} refuses NaN peak accuracies (heads: {nan_peaks})")
    return stats


def _above_ablation_order(decisions: dict[Head, dict]) -> Callable[[Head], tuple]:
    def order_key(pos: Head) -> tuple:
        return (
            -decisions[pos]["gain"],
            -decisions[pos]["acc_at_c_star"],
            pos[0],
            pos[1],
        )

    return order_key


def above_ablation_zero_v1(scan: dict, params: dict) -> SelectorResult:
    """Every head whose peak recovery exceeds its own c=0 (mean-ablated)
    accuracy strictly — the noise region set to zero.

    With the legacy first-index argmax, ``gain > 0`` is equivalent to the
    peak lying at a non-zero coefficient. No existence gate; scan-level
    applicability is the caller's judgement. Intended as the permissive
    endpoint of the above_ablation family — on real scans max-over-grid
    sampling noise alone lifts most null heads above their c=0 draw, so this
    variant is expected to over-select (kept as a named comparison point).
    """
    if params:
        raise SelectorError(
            f"above_ablation_zero takes no parameters (got {sorted(params)})"
        )
    stats = _peak_vs_c0_stats(scan, "above_ablation_zero")
    decisions: dict[Head, dict] = {}
    main_heads: list[Head] = []
    for pos, s in stats.items():
        is_main = s["gain"] > 0.0
        decisions[pos] = {
            "c_star": s["c_star"],
            "acc_at_c_star": float(s["acc_at_c_star"]),
            "acc_at_c0": float(s["acc_at_c0"]),
            "gain": float(s["gain"]),
            "threshold": float(s["acc_at_c0"]),
            "is_main": is_main,
            "main_reason": "above_c0" if is_main else None,
            "flags": [] if is_main else ["peak_at_c0"],
        }
        if is_main:
            main_heads.append(pos)
    main_heads.sort(key=_above_ablation_order(decisions))
    if not main_heads:
        verdict_label = "none"
    elif len(main_heads) <= 5:
        verdict_label = "localized"
    else:
        verdict_label = "distributed"
    verdict = {
        "verdict": verdict_label,
        "n_selected": len(main_heads),
        "n_sig": len(stats),
        "noise": {"band": "zero"},
        "order": "gain desc",
    }
    return SelectorResult(
        selector_name="above_ablation_zero",
        selector_version=1,
        params={},
        main_heads=main_heads,
        minor_heads=[],
        decisions=_keyed(decisions),
        verdict=verdict,
    )


def above_ablation_v1(scan: dict, params: dict) -> SelectorResult:
    """Every head whose peak recovery exceeds the mean-ablated (c=0)
    baseline beyond a noise region derived from the corrupted accuracy.

    Null model: a head whose injection does nothing leaves a flat curve at
    the corrupted level p0; the finite-sample (example-sampling) error of
    each scanned point is treated as a Binomial(n_eval, p0)/n_eval draw, so
    the null peak is the max over the non-zero grid of such draws. p0 is
    estimated by pooling every head's c=0 accuracy — a stability choice
    that anchors one scan-level corrupted level rather than letting a head
    lower its own bar via its own low c0 value (c=0 is a per-head
    leave-one-out condition; see the module docstring). A head is MAIN iff
    its peak is a) at a non-zero coefficient and b) strictly above the
    (1-alpha) null quantile of that max: threshold =
    BinomialQuantile(n_eval, pooled_c0, (1-alpha)^(1/m)) / n_eval with m =
    the head's non-zero grid size (``scanwise=0.0``, per-head test at level
    alpha; expected false selections <= alpha * n_sig) or m = the total
    non-zero points across all scanned heads (``scanwise=1.0``, family-wise
    level alpha across the scan). Points share evaluation examples across
    c, so against the flat-curve null the iid/Sidak assumptions
    over-correct — the band is conservative in that direction. CAVEATS: the
    plug-in pooled_c0 carries ~one single-evaluation binomial SE of
    unpropagated uncertainty (the pooled c0 draws are near-perfectly
    correlated — no sqrt(H) shrink); a +/-1 SE shift moves borderline heads
    (~4/33 on the n=2500 canonical scan) — audit ``decisions.excess`` near
    0. No existence gate (cf. ``largest_gap``); requires
    ``n_eval_examples_per_head_per_c``.
    """
    unknown = set(params) - {"alpha", "scanwise"}
    missing = {"alpha", "scanwise"} - set(params)
    if unknown or missing:
        raise SelectorError(
            "above_ablation takes exactly params {'alpha', 'scanwise'} "
            f"(missing: {sorted(missing)}, unknown: {sorted(unknown)})"
        )
    alpha = float(params["alpha"])
    if not 0.0 < alpha < 1.0:
        raise SelectorError(f"above_ablation alpha must be in (0, 1), got {alpha}")
    scanwise_raw = float(params["scanwise"])
    if scanwise_raw not in (0.0, 1.0):
        raise SelectorError(
            f"above_ablation scanwise must be 0.0 or 1.0, got {scanwise_raw}"
        )
    scanwise = scanwise_raw == 1.0
    n_eval = scan.get("n_eval_examples_per_head_per_c")
    if not n_eval or int(n_eval) != n_eval or n_eval <= 0:
        raise SelectorError(
            "above_ablation needs a positive integer "
            f"n_eval_examples_per_head_per_c in the scan (got {n_eval!r})"
        )
    n_eval = int(n_eval)

    stats = _peak_vs_c0_stats(scan, "above_ablation")
    pooled_c0 = sum(s["acc_at_c0"] for s in stats.values()) / len(stats)
    total_nonzero = sum(s["n_nonzero_c"] for s in stats.values())
    thresholds: dict[int, float] = {}

    def threshold_for(m_points: int) -> float:
        if m_points not in thresholds:
            quantile = (1.0 - alpha) ** (1.0 / m_points)
            thresholds[m_points] = (
                _binomial_quantile(n_eval, pooled_c0, quantile) / n_eval
            )
        return thresholds[m_points]

    decisions: dict[Head, dict] = {}
    main_heads: list[Head] = []
    for pos, s in stats.items():
        m_points = total_nonzero if scanwise else s["n_nonzero_c"]
        threshold = threshold_for(m_points)
        is_main = s["c_star"] != 0 and s["acc_at_c_star"] > threshold
        flags: list[str] = []
        if s["c_star"] == 0:
            flags.append("peak_at_c0")
        elif not is_main:
            flags.append("below_noise_band")
        decisions[pos] = {
            "c_star": s["c_star"],
            "acc_at_c_star": float(s["acc_at_c_star"]),
            "acc_at_c0": float(s["acc_at_c0"]),
            "gain": float(s["gain"]),
            "threshold": float(threshold),
            "excess": float(s["acc_at_c_star"] - threshold),
            "is_main": is_main,
            "main_reason": "above_noise_band" if is_main else None,
            "flags": flags,
        }
        if is_main:
            main_heads.append(pos)
    main_heads.sort(key=_above_ablation_order(decisions))
    if not main_heads:
        verdict_label = "none"
    elif len(main_heads) <= 5:
        verdict_label = "localized"
    else:
        verdict_label = "distributed"
    threshold_values = sorted(set(thresholds.values()))
    verdict = {
        "verdict": verdict_label,
        "n_selected": len(main_heads),
        "n_sig": len(stats),
        "noise": {
            "band": "corrupted_binomial_max_q",
            "alpha": alpha,
            "scanwise": scanwise,
            "pooled_c0": float(pooled_c0),
            "n_eval": n_eval,
            "grid_points_total": total_nonzero,
            "threshold": threshold_values[0] if len(threshold_values) == 1 else None,
            "thresholds": threshold_values,
            "expected_false_selected_upper": float(
                alpha if scanwise else alpha * len(stats)
            ),
            "mean_fv_acc_ref": scan.get("baselines", {}).get("mean_fv_acc"),
        },
        "order": "gain desc",
    }
    return SelectorResult(
        selector_name="above_ablation",
        selector_version=1,
        params={"alpha": alpha, "scanwise": scanwise_raw},
        main_heads=main_heads,
        minor_heads=[],
        decisions=_keyed(decisions),
        verdict=verdict,
    )


def compare_zero_v1(scan: dict, params: dict) -> SelectorResult:
    """Every head whose peak recovery accuracy (max over the scanned
    coefficient grid, c=0 included) STRICTLY exceeds the largest c=0
    (mean-ablated leave-one-out) accuracy across all scanned heads.

    One scan-level bar: max_c0 = max over heads of ``acc_at_c0`` — the
    empirical envelope of the leave-one-out corrupted levels. Because
    max_c0 >= every head's own c0, the selection is always a subset of
    ``above_ablation_zero``'s (identical to it on a single-head scan, where
    the envelope degenerates to the head's own c0); peaks at c=0 can never
    pass (their value is itself a c0 value); the bar rises with the single
    highest-c0 head and no head can exceed a bar of 1.0 (a saturated c0
    selects nothing). Needs no ``n_eval`` and no distributional null, and
    the THRESHOLD does not scale with the grid size searched — but the
    SELECTION does (the peak is an uncorrected running max over the grid
    against a fixed bar), and how much the bar adds over the zero band is
    set entirely by the cross-head c0 spread. On low-spread scans
    (canonical add: max_c0 - pooled_c0 ~ 0.4 binomial SE, ~ the
    ``mean_fv_acc`` plateau) a flat-null head clears the bar at some grid
    point almost surely, so this variant is expected to over-select there,
    like ``above_ablation_zero`` (within 2-3 heads of it on the add scans,
    vs ~half the significant set for the banded ``above_ablation``); on
    high-spread scans (extractive ~2.4 SE, abstractive ~15 SE) the envelope
    does real work and the selection approaches the banded one.
    ``verdict.noise.n_c0_pooled`` counts the c0 values enveloped (field
    name kept family-consistent; the bar is a max, not a pooled mean).
    Ordered by peak accuracy descending (equivalently ``excess`` = peak -
    max_c0 descending), then gain, then layer/head. No existence gate
    (cf. ``largest_gap``); scan-level applicability is the caller's
    judgement.
    """
    if params:
        raise SelectorError(f"compare_zero takes no parameters (got {sorted(params)})")
    stats = _peak_vs_c0_stats(scan, "compare_zero")
    max_c0 = max(s["acc_at_c0"] for s in stats.values())
    max_c0_heads = sorted(pos for pos, s in stats.items() if s["acc_at_c0"] == max_c0)
    decisions: dict[Head, dict] = {}
    main_heads: list[Head] = []
    for pos, s in stats.items():
        is_main = s["acc_at_c_star"] > max_c0
        flags: list[str] = []
        if s["c_star"] == 0:
            flags.append("peak_at_c0")
        elif not is_main:
            flags.append("below_max_c0")
        decisions[pos] = {
            "c_star": s["c_star"],
            "acc_at_c_star": float(s["acc_at_c_star"]),
            "acc_at_c0": float(s["acc_at_c0"]),
            "gain": float(s["gain"]),
            "threshold": float(max_c0),
            "excess": float(s["acc_at_c_star"] - max_c0),
            "is_main": is_main,
            "main_reason": "above_max_c0" if is_main else None,
            "flags": flags,
        }
        if is_main:
            main_heads.append(pos)
    main_heads.sort(
        key=lambda p: (
            -decisions[p]["acc_at_c_star"],
            -decisions[p]["gain"],
            p[0],
            p[1],
        )
    )
    if not main_heads:
        verdict_label = "none"
    elif len(main_heads) <= 5:
        verdict_label = "localized"
    else:
        verdict_label = "distributed"
    verdict = {
        "verdict": verdict_label,
        "n_selected": len(main_heads),
        "n_sig": len(stats),
        "noise": {
            "band": "max_c0",
            "threshold": float(max_c0),
            "max_c0_heads": [list(p) for p in max_c0_heads],
            "n_c0_pooled": len(stats),
            "mean_fv_acc_ref": scan.get("baselines", {}).get("mean_fv_acc"),
        },
        "order": "acc_at_c_star desc",
    }
    return SelectorResult(
        selector_name="compare_zero",
        selector_version=1,
        params={},
        main_heads=main_heads,
        minor_heads=[],
        decisions=_keyed(decisions),
        verdict=verdict,
    )


def _mcnemar_tail(n_discordant: int, n_fixed: int) -> float:
    """Exact one-sided McNemar p-value P[Binomial(n, 1/2) >= b].

    Integer-exact: the tail mass is a rational sum(C(n,k), k=b..n) / 2^n and
    Python's big-int true division rounds it correctly to the nearest float.
    b <= 0 gives 1.0 (includes the no-discordant-examples convention n=0).
    The sum walks the binomial-coefficient recurrence from the shorter end
    (one exact small-int multiply/divide per term instead of a fresh
    ``math.comb`` per k), which keeps llama-scale scans (discordant counts
    in the hundreds) fast without giving up exactness."""
    if n_fixed <= 0:
        return 1.0
    if n_fixed > n_discordant:
        return 0.0
    if n_fixed >= n_discordant - n_fixed + 1:
        # upper tail is the shorter side: sum C(n,k) for k = b..n
        term = math.comb(n_discordant, n_fixed)
        tail = term
        for k in range(n_fixed, n_discordant):
            term = term * (n_discordant - k) // (k + 1)
            tail += term
        return tail / (1 << n_discordant)
    # complement is shorter: P >= b = 1 - sum C(n,k) for k = 0..b-1 (exact int)
    term = 1  # C(n, 0)
    head = term
    for k in range(n_fixed - 1):
        term = term * (n_discordant - k) // (k + 1)
        head += term
    return ((1 << n_discordant) - head) / (1 << n_discordant)


def _step_up_adjusted(p_values: list[float], *, harmonic: bool) -> list[float]:
    """BH (harmonic=False) / BY (harmonic=True) adjusted p-values.

    adj_(i) = min(1, min_{j>=i} scale * p_(j) / j) over the ascending order,
    with scale = N (BH) or N * c(N), c(N) = sum_{k<=N} 1/k (BY). Equal raw
    p-values collapse to one adjusted value (the running minimum), so the
    result is independent of how ties are ordered."""
    n_tests = len(p_values)
    order = sorted(range(n_tests), key=lambda i: (p_values[i], i))
    scale = float(n_tests)
    if harmonic:
        scale *= sum(1.0 / k for k in range(1, n_tests + 1))
    adjusted = [1.0] * n_tests
    running = float("inf")
    for rank in range(n_tests, 0, -1):
        index = order[rank - 1]
        running = min(running, p_values[index] * scale / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def _paired_outcome_vector(
    outcomes, key: str, n_eval: int, recorded_acc, label: str
) -> tuple:
    """Fetch + validate one per-example outcome vector; returns (vector, sum).

    Alignment contract: the vector must exist, have exactly ``n_eval``
    entries, be strictly 0/1, and reproduce the recorded curve accuracy
    EXACTLY (both sides are the same integer-count/n_eval float64 division,
    so any difference — including a NaN curve point — is misalignment)."""
    if key not in outcomes:
        raise SelectorError(
            f"{label}: scan outcomes are missing the vector {key!r}; the "
            "outcomes npz does not cover the scan's curves (misaligned or "
            "truncated outcomes)."
        )
    vector = outcomes[key]
    if len(vector) != n_eval:
        raise SelectorError(
            f"{label}: outcome vector {key!r} has {len(vector)} entries, "
            f"expected n_eval_examples_per_head_per_c = {n_eval}; refusing "
            "misaligned outcomes."
        )
    total = 0
    for value in vector:
        if value == 1:
            total += 1
        elif value != 0:
            raise SelectorError(
                f"{label}: outcome vector {key!r} contains a non-binary "
                f"value {value!r}; refusing."
            )
    recomputed = total / n_eval
    if float(recorded_acc) != recomputed:
        raise SelectorError(
            f"{label}: outcome vector {key!r} does not reproduce the "
            f"recorded curve accuracy (recorded {recorded_acc!r}, recomputed "
            f"{recomputed!r}); refusing misaligned outcomes."
        )
    return vector, total


def _paired_head_stats(scan: dict, outcomes, label: str) -> tuple[dict, dict]:
    """Per-head paired McNemar statistics over the scan's outcome vectors.

    Returns (stats keyed by head, alignment-check record). Refuses: absent
    outcomes, missing/short/non-binary/curve-mismatched vectors, grids
    without c=0 or without a positive coefficient, and negative coefficients
    (the one-sided 'scaling up fixes examples' alternative is defined for
    c >= 0 only). Non-contiguous positive grids are fine — m counts the
    positive coefficients actually present in the head's curve."""
    if outcomes is None:
        raise SelectorError(
            f"{label} needs the scan's per-example outcomes "
            "(scan_outcomes.npz); apply it through the pipeline, which loads "
            "the digest-verified npz alongside head_scan.json."
        )
    n_eval = scan.get("n_eval_examples_per_head_per_c")
    if not n_eval or int(n_eval) != n_eval or n_eval <= 0:
        raise SelectorError(
            f"{label} needs a positive integer n_eval_examples_per_head_per_c "
            f"in the scan (got {n_eval!r})"
        )
    n_eval = int(n_eval)
    curves = _parse_curves(scan)
    n_keys_checked = 0
    stats: dict[Head, dict] = {}
    for pos, curve in curves.items():
        try:
            grid = {int(c): c for c in curve}
        except (TypeError, ValueError):
            raise SelectorError(
                f"{label}: head {pos} has a non-integer coefficient key in "
                f"its curve ({sorted(map(str, curve))}); refusing."
            ) from None
        if len(grid) != len(curve):
            raise SelectorError(
                f"{label}: head {pos} has duplicate coefficients in its "
                f"curve ({sorted(curve)}); refusing."
            )
        if any(c < 0 for c in grid):
            raise SelectorError(
                f"{label} is defined for c >= 0 only (head {pos} has grid "
                f"{sorted(grid)})"
            )
        positive = sorted(c for c in grid if c > 0)
        if 0 not in grid or not positive:
            raise SelectorError(
                f"{label} needs c=0 plus at least one positive coefficient "
                f"per head (head {pos} has grid {sorted(grid)})"
            )
        y0, total0 = _paired_outcome_vector(
            outcomes, f"{pos[0]}:{pos[1]}:0", n_eval, curve[grid[0]], label
        )
        n_keys_checked += 1
        per_coef: dict[str, dict] = {}
        best_key = None
        best_c = None
        for c in positive:
            yc, total_c = _paired_outcome_vector(
                outcomes, f"{pos[0]}:{pos[1]}:{c}", n_eval, curve[grid[c]], label
            )
            n_keys_checked += 1
            fixed = broken = 0
            for value_c, value_0 in zip(yc, y0, strict=True):
                if value_c:
                    if not value_0:
                        fixed += 1
                elif value_0:
                    broken += 1
            p_raw = _mcnemar_tail(fixed + broken, fixed)
            per_coef[str(c)] = {
                "b": fixed,
                "d": broken,
                "n_discordant": fixed + broken,
                "gain": (fixed - broken) / n_eval,
                "acc": total_c / n_eval,
                "p_raw": p_raw,
            }
            # smallest p, then larger net gain b-d, then smaller coefficient
            tie_key = (p_raw, -(fixed - broken), c)
            if best_key is None or tie_key < best_key:
                best_key = tie_key
                best_c = c
        chosen = per_coef[str(best_c)]
        m_coefs = len(per_coef)
        stats[pos] = {
            "c_star": best_c,
            "b": chosen["b"],
            "d": chosen["d"],
            "gain": chosen["gain"],
            "acc_at_c_star": chosen["acc"],
            "acc_at_c0": total0 / n_eval,
            "p_raw_min": chosen["p_raw"],
            "n_pos_coefs": m_coefs,
            "p_head": min(1.0, m_coefs * chosen["p_raw"]),
            "per_coef": per_coef,
        }
    checks = {
        "n_eval": n_eval,
        "n_vectors_checked": n_keys_checked,
        "curve_accuracy_match": "exact",
        "content_sha256": (scan.get("outcomes") or {}).get("content_sha256"),
    }
    return stats, checks


_PAIRED_ACROSS = {
    "paired_bh": ("bh", "q"),
    "paired_per_head": ("none", "alpha"),
    "paired_by": ("by", "q"),
    "paired_global_bonferroni": ("bonferroni", "alpha"),
}


def _paired_family(scan: dict, params: dict, outcomes, *, name: str) -> SelectorResult:
    across, level_name = _PAIRED_ACROSS[name]
    unknown = set(params) - {level_name}
    missing = {level_name} - set(params)
    if unknown or missing:
        raise SelectorError(
            f"{name} takes exactly params {{{level_name!r}}} "
            f"(missing: {sorted(missing)}, unknown: {sorted(unknown)})"
        )
    level = float(params[level_name])
    if not 0.0 < level < 1.0:
        raise SelectorError(f"{name} {level_name} must be in (0, 1), got {level}")

    stats, checks = _paired_head_stats(scan, outcomes, name)
    heads = list(stats)
    n_heads = len(heads)
    p_heads = [stats[pos]["p_head"] for pos in heads]
    if across == "none":
        adjusted = list(p_heads)  # p_head is already capped at 1
    elif across == "bonferroni":
        adjusted = [min(1.0, n_heads * p) for p in p_heads]
    else:
        adjusted = _step_up_adjusted(p_heads, harmonic=(across == "by"))

    decisions: dict[Head, dict] = {}
    main_heads: list[Head] = []
    for index, pos in enumerate(heads):
        s = stats[pos]
        gain_positive = s["b"] > s["d"]
        significant = adjusted[index] <= level
        is_main = bool(significant and gain_positive)
        flags: list[str] = []
        if not significant:
            flags.append("p_above_level")
        if not gain_positive:
            flags.append("gain_not_positive")
        decisions[pos] = {
            "c_star": s["c_star"],
            "acc_at_c_star": float(s["acc_at_c_star"]),
            "acc_at_c0": float(s["acc_at_c0"]),
            "gain": float(s["gain"]),
            "b": s["b"],
            "d": s["d"],
            "n_pos_coefs": s["n_pos_coefs"],
            "p_raw_min": float(s["p_raw_min"]),
            "p_head": float(s["p_head"]),
            "p_adjusted": float(adjusted[index]),
            "per_coef": s["per_coef"],
            "is_main": is_main,
            "main_reason": f"significant_{across}" if is_main else None,
            "flags": flags,
        }
        if is_main:
            main_heads.append(pos)
    main_heads.sort(
        key=lambda p: (
            decisions[p]["p_adjusted"],
            decisions[p]["p_head"],
            -decisions[p]["gain"],
            p[0],
            p[1],
        )
    )
    if not main_heads:
        verdict_label = "none"
    elif len(main_heads) <= 5:
        verdict_label = "localized"
    else:
        verdict_label = "distributed"
    test_block = {
        "statistic": "exact_one_sided_mcnemar",
        "within_head": "bonferroni_over_positive_coefficients",
        "across_heads": across,
        level_name: level,
        "n_heads": n_heads,
        "positive_gain_gate": "b > d at c_star",
    }
    if across == "by":
        test_block["harmonic_c_n"] = float(sum(1.0 / k for k in range(1, n_heads + 1)))
    verdict = {
        "verdict": verdict_label,
        "n_selected": len(main_heads),
        "n_sig": n_heads,
        "test": test_block,
        "outcomes_check": checks,
        "order": "p_adjusted asc",
    }
    return SelectorResult(
        selector_name=name,
        selector_version=1,
        params={level_name: level},
        main_heads=main_heads,
        minor_heads=[],
        decisions=_keyed(decisions),
        verdict=verdict,
    )


def paired_bh_v1(scan: dict, params: dict, outcomes=None) -> SelectorResult:
    """Paired McNemar per head, Benjamini-Hochberg across heads at FDR q.

    The family's primary method (q=0.05). BH is exactly FDR-controlling
    under PRDS positive dependence; heads share evaluation examples, so
    positive dependence is plausible but not guaranteed — ``paired_by`` is
    the dependence-robust sensitivity check. See the module docstring for
    the full construction (pairing validity, within-head Bonferroni,
    c_star tie-breaks, positive-gain gate)."""
    return _paired_family(scan, params, outcomes, name="paired_bh")


def paired_per_head_v1(scan: dict, params: dict, outcomes=None) -> SelectorResult:
    """Paired McNemar with within-head Bonferroni only: a head is selected
    iff its head-level p-value (already corrected over its coefficient grid)
    is <= alpha, with NO across-head multiplicity step — the permissive
    endpoint of the paired family (expected false selections <= alpha *
    n_heads under the per-head null)."""
    return _paired_family(scan, params, outcomes, name="paired_per_head")


def paired_by_v1(scan: dict, params: dict, outcomes=None) -> SelectorResult:
    """Paired McNemar per head, Benjamini-Yekutieli across heads at FDR q:
    BH with the harmonic factor c(N) = sum_{k<=N} 1/k, FDR-valid under
    ARBITRARY dependence between the head-level tests. The conservative
    sensitivity check for ``paired_bh``."""
    return _paired_family(scan, params, outcomes, name="paired_by")


def paired_global_bonferroni_v1(
    scan: dict, params: dict, outcomes=None
) -> SelectorResult:
    """Paired McNemar per head, family-wise Bonferroni across heads at level
    alpha (head selected iff n_heads * p_head <= alpha and gain positive).
    The strongest guarantee in the family (FWER — probability of even ONE
    false selection — under arbitrary dependence); its selection is always
    ⊆ ``paired_bh``'s, but NOT in general ⊆ ``paired_by``'s (see the module
    docstring)."""
    return _paired_family(scan, params, outcomes, name="paired_global_bonferroni")


# (name, version) -> pure function(scan, params[, outcomes]) -> SelectorResult
# (members of OUTCOME_SELECTORS take the per-example outcome vectors as a
# third argument; everything else is called with (scan, params) only)
SELECTORS: dict[tuple[str, int], Callable[..., SelectorResult]] = {
    ("unified", 1): unified_v1,
    ("recovery_weak", 1): recovery_weak_v1,
    ("pin", 1): pin_v1,
    ("largest_gap", 1): largest_gap_v1,
    ("above_ablation", 1): above_ablation_v1,
    ("above_ablation_zero", 1): above_ablation_zero_v1,
    ("compare_zero", 1): compare_zero_v1,
    ("paired_bh", 1): paired_bh_v1,
    ("paired_per_head", 1): paired_per_head_v1,
    ("paired_by", 1): paired_by_v1,
    ("paired_global_bonferroni", 1): paired_global_bonferroni_v1,
}


def get_selector(name: str, version: int) -> Callable[..., SelectorResult]:
    try:
        return SELECTORS[(name, version)]
    except KeyError:
        known = ", ".join(f"{n} v{v}" for n, v in sorted(SELECTORS))
        raise SelectorError(
            f"unknown selector {name!r} v{version} (known: {known})"
        ) from None
