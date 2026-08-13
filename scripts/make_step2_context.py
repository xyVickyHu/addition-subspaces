"""Generate a Step-2 analysis-context YAML from a Step-1 head artifact (CPU).

Thin wrapper over :mod:`subspaces.step2.context_gen` (lineage walk, samples lookup,
protocol-filtered held-out cache discovery — see that module for the rules).
Review the YAML, commit it (e.g. under ``configs/contexts/``), then run::

    .venv/bin/python scripts/make_step2_context.py \\
        --heads-artifact log/runs/<node>/sig-*/scan-*/main-largest_gap-v1-<h8>/main_heads.json \\
        --include-heldout --out configs/contexts/<cell>.yaml
    .venv/bin/python -m subspaces.runners.step2 --context configs/contexts/<cell>.yaml
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from subspaces.artifacts import ArtifactError
from subspaces.paths import ProjectPaths
from subspaces.step2.context_gen import build_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument(
        "--heads-artifact",
        required=True,
        help="heads.json or main_heads.json to analyze",
    )
    parser.add_argument(
        "--include-heldout",
        action="store_true",
        help="also reference the cell's held-out z cache (all-tasks PCA)",
    )
    parser.add_argument(
        "--heldout-fingerprint",
        default=None,
        help="explicit held-out z-cache fingerprint (skips discovery)",
    )
    parser.add_argument("--out", required=True, help="output context YAML path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = ProjectPaths.from_root(args.root)
        result = build_context(
            paths,
            args.heads_artifact,
            include_heldout=args.include_heldout,
            heldout_fingerprint=args.heldout_fingerprint,
        )
    except (ArtifactError, OSError, KeyError, json.JSONDecodeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    for warning in result["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    out_path = paths.resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(result["context"], sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
