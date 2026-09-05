"""Paper phase 1 (post-training) — extract selected heads from a trained matrix.

Loads the latest ``matrix_epoch*.pth`` under ``<log_dir>/checkpoints/`` and
writes a JSONL of ``{"head position": [layer, head], "coefficient": coef}``
rows for every entry above the chosen threshold.

The threshold can be set:
  - explicitly via ``--threshold`` (matches the legacy CLI),
  - or automatically via ``--auto-threshold elbow|fraction`` (uses
    ``subspaces.heads.auto_threshold`` to pick a knee in the sorted coefficient curve).

Default invocation reproduces the legacy head-select script exactly
(threshold=0.0 → every entry sorted by coefficient descending).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from subspaces.utils.heads import (
    auto_main_heads,
    auto_threshold,
    extract_heads_above_threshold,
    load_latest_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract large coefficients from the latest matrix checkpoint produced by FV_matrix.py "
            "and save them to a JSONL file."
        )
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Path to the log directory produced by FV_matrix.py (required).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0,
        help="Minimum coefficient value to include (default: 0). Ignored when --auto-threshold is set.",
    )
    parser.add_argument(
        "--auto-threshold",
        choices=("none", "elbow", "fraction"),
        default="none",
        help="If set, derive the threshold automatically from the coefficient distribution.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.95,
        help="Fraction of coefficient mass to retain when --auto-threshold=fraction.",
    )
    parser.add_argument(
        "--report-main-heads",
        type=int,
        default=0,
        help="If >0, also print the top-K main heads (ranked by per-head accuracy if "
        "<log_dir>/head_acc_dict.pth exists, else by |coef|).",
    )
    parser.add_argument(
        "--output_name",
        default="head_ordered.jsonl",
        help="Name of the JSONL file created inside --log_dir.",
    )
    return parser.parse_args()


def write_jsonl(entries, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as json_file:
        for entry in entries:
            if isinstance(entry, dict):
                json_line = entry
            else:
                (row, col), coeff = entry
                json_line = {"head position": [row, col], "coefficient": coeff}
            json_file.write(json.dumps(json_line))
            json_file.write("\n")


def main() -> None:
    args = parse_args()
    if args.log_dir is None:
        raise SystemExit(
            "--log_dir is required: pass the matrix log directory produced by "
            "subspaces.runners.train_matrix (it must contain a checkpoints/ subdirectory), "
            "e.g. --log_dir log/<task>/<run_name>."
        )
    matrix, checkpoint_path = load_latest_matrix(args.log_dir)

    if args.auto_threshold == "elbow":
        threshold = auto_threshold(matrix, method="elbow")
        print(f"[auto-threshold] elbow → {threshold:.4f}")
    elif args.auto_threshold == "fraction":
        threshold = auto_threshold(matrix, method="fraction", fraction=args.fraction)
        print(f"[auto-threshold] fraction={args.fraction} → {threshold:.4f}")
    else:
        threshold = args.threshold

    heads = extract_heads_above_threshold(matrix, threshold)
    output_path = os.path.join(args.log_dir, args.output_name)
    write_jsonl(heads, output_path)
    print(
        f"Wrote {len(heads)} entries to {output_path} from checkpoint {checkpoint_path} "
        f"using threshold {threshold}."
    )

    if args.report_main_heads > 0:
        acc_path = os.path.join(args.log_dir, "head_acc_dict.pth")
        rank_by = "accuracy" if os.path.exists(acc_path) else "coef"
        main_heads = auto_main_heads(
            matrix, k=args.report_main_heads, rank_by=rank_by, matrix_dir=args.log_dir
        )
        print(f"Top-{args.report_main_heads} main heads: {main_heads}")


if __name__ == "__main__":
    main()
