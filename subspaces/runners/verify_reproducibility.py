"""CPU-only reproducibility check for head selection in the refactored ``subspaces`` pkg.

Two things this asserts/reports (no GPU, no model forward pass — pure matrix +
precomputed-accuracy loads):

1. **Addition baseline reproduces (asserted).**
   - Significant heads: ``select_heads(canon, 0.2)`` returns exactly the paper's
     33-head set, and it equals the legacy ``head_ordered.jsonl`` coef>0.2 set.
   - Main heads: ``auto_main_heads(rank_by='accuracy', k=3)`` returns the paper's
     ``{(13,6),(15,2),(15,1)}`` from the cached per-head ``head_acc_dict.pth``.
   The script exits non-zero if either assertion fails.

2. **Beyond-addition head selection (reported, not asserted).**
   For each non-addition coefficient matrix that exists on disk
   (``abstractive``/``extractive``/``number_mul``/``number``) it tabulates the
   significant-head set (elbow + |coef|>0.2), the top heads, and whether the
   per-head accuracy dict needed for *main*-head selection is present. These
   matrices are *unbounded* (not clipped to [0,1]), so the 0.2 threshold is not
   pre-calibrated for them — the scale-free elbow is the meaningful selector.
   Main-head selection for these requires a GPU pass (``run_head_eval`` to make
   ``head_acc_dict.pth``, or the recovery scan in ``run_pipeline``); that is
   reported as a TODO, not run here.

Usage:
    python -m subspaces.runners.verify_reproducibility            # canon + any non-add matrices found
    python -m subspaces.runners.verify_reproducibility --json out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from subspaces.utils.heads import (  # noqa: E402
    _load_per_head_accuracy_dict,
    auto_main_heads,
    auto_threshold,
    load_latest_matrix,
    select_heads,
)

CANON_DIR = "artifacts/matrix_add_0204_clip_lambda0.05"
PAPER_SIG_COUNT = 33
PAPER_MAIN_HEADS = {(13, 6), (15, 2), (15, 1)}
NONADD_TASKS = ("abstractive", "extractive", "number_mul")


def _pick_matrix_dir(task: str) -> str | None:
    """Newest dir under ``log/<task>/`` that has a ``checkpoints/`` (prefer lambda0.05)."""
    cands = [
        os.path.dirname(c)
        for c in sorted(
            glob.glob(str(PROJECT_ROOT / "log" / task / "*" / "checkpoints"))
        )
    ]
    for c in cands:
        if "lambda0.05" in c:
            return c
    return cands[-1] if cands else None


def verify_addition() -> dict:
    """Assert the paper add baseline (33 significant + 3 main). Returns a report dict."""
    d = str(PROJECT_ROOT / CANON_DIR)
    matrix, ckpt = load_latest_matrix(d)

    sig = select_heads(matrix, 0.2)
    sig_set = {(int(l), int(h)) for (l, h) in sig}

    # Cross-check against the legacy head_ordered.jsonl coef>0.2 set.
    legacy_path = os.path.join(d, "head_ordered.jsonl")
    legacy_set = None
    if os.path.exists(legacy_path):
        legacy_set = set()
        with open(legacy_path) as f:
            for line in f:
                row = json.loads(line)
                if row["coefficient"] > 0.2:
                    legacy_set.add(tuple(int(x) for x in row["head position"]))

    main = auto_main_heads(matrix, k=3, rank_by="accuracy", matrix_dir=d)
    main_set = {(int(l), int(h)) for (l, h) in main}

    sig_ok = len(sig) == PAPER_SIG_COUNT
    legacy_ok = (legacy_set is None) or (sig_set == legacy_set)
    main_ok = main_set == PAPER_MAIN_HEADS

    report = {
        "matrix_dir": CANON_DIR,
        "checkpoint": os.path.basename(ckpt),
        "n_significant_at_0.2": len(sig),
        "significant_ok": sig_ok,
        "matches_legacy_jsonl": legacy_ok,
        "main_heads_accuracy_mode": sorted(main_set),
        "main_heads_ok": main_ok,
        "elbow_threshold": round(auto_threshold(matrix, method="elbow"), 4),
        "passed": bool(sig_ok and legacy_ok and main_ok),
    }
    return report


def report_beyond_addition() -> list[dict]:
    """Report CPU-available head selection for each non-add matrix on disk."""
    rows = []
    for task in NONADD_TASKS:
        d = _pick_matrix_dir(task)
        if not d:
            rows.append({"task": task, "status": "no_matrix_on_disk"})
            continue
        matrix, ckpt = load_latest_matrix(d)
        elbow = auto_threshold(matrix, method="elbow")
        sig_elbow = select_heads(matrix, elbow)
        sig_02 = select_heads(matrix, 0.2)
        top = auto_main_heads(matrix, k=8, rank_by="coef")
        has_acc = bool(_load_per_head_accuracy_dict(d, n_shot=5))
        rows.append(
            {
                "task": task,
                "matrix_dir": os.path.relpath(d, PROJECT_ROOT),
                "checkpoint": os.path.basename(ckpt),
                "clipped_0_1": bool(matrix.min().item() >= 0),
                "coef_min": round(matrix.min().item(), 3),
                "coef_max": round(matrix.max().item(), 3),
                "n_sig_elbow": len(sig_elbow),
                "elbow_threshold": round(elbow, 4),
                "n_sig_at_0.2": len(sig_02),
                "top8_by_coef": [list(map(int, t)) for t in top],
                "main_heads_cpu_available": has_acc,
                "main_heads_note": (
                    "accuracy-mode available (head_acc_dict.pth present)"
                    if has_acc
                    else "MAIN HEADS NEED GPU: no head_acc_dict.pth — run subspaces.runners.run_head_eval, "
                    "or use run_pipeline --main-rank-by recovery (forward passes)"
                ),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--json", default=None, help="Optional path to dump the full report as JSON."
    )
    args = ap.parse_args()

    add = verify_addition()
    beyond = report_beyond_addition()

    print("=" * 72)
    print("ADDITION BASELINE REPRODUCTION (asserted)")
    print("=" * 72)
    print(f"  matrix:           {add['matrix_dir']} [{add['checkpoint']}]")
    print(
        f"  significant @0.2: {add['n_significant_at_0.2']}  (expect {PAPER_SIG_COUNT})  "
        f"-> {'OK' if add['significant_ok'] else 'FAIL'}"
    )
    print(f"  == legacy jsonl:  {add['matches_legacy_jsonl']}")
    print(
        f"  main (accuracy):  {add['main_heads_accuracy_mode']}  "
        f"-> {'OK' if add['main_heads_ok'] else 'FAIL'}"
    )
    print(f"  VERDICT:          {'PASS' if add['passed'] else 'FAIL'}")

    print()
    print("=" * 72)
    print("BEYOND ADDITION — head selection on existing non-add matrices (CPU)")
    print("=" * 72)
    for r in beyond:
        if r.get("status") == "no_matrix_on_disk":
            print(
                f"  {r['task']:<14} (no trained matrix under log/<task>/; the"
                " shipped artifacts/ matrices are consumed by the config-driven"
                " step-1 pipeline, not this report)"
            )
            continue
        print(f"  {r['task']:<14} {r['matrix_dir']}")
        print(
            f"  {'':<14} signed={not r['clipped_0_1']} coef[{r['coef_min']},{r['coef_max']}] | "
            f"elbow={r['elbow_threshold']} -> {r['n_sig_elbow']} sig | |coef|>0.2 -> {r['n_sig_at_0.2']} sig"
        )
        print(f"  {'':<14} top8|coef|: {[tuple(t) for t in r['top8_by_coef']]}")
        print(f"  {'':<14} main heads: {r['main_heads_note']}")

    out = {"addition": add, "beyond_addition": beyond}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\n[wrote] {args.json}")

    return 0 if add["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
