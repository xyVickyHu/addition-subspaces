"""Label-share distributions from the step-3 token-group artifacts (CPU).

For every cell of ``configs/step23_recpos_cells.tsv`` with a step-3 artifact
in its recovery-positive node: per head, the LABEL share = the sum of the
``demo{i}_output`` group shares (per prompt, from the outcomes npz; the mean
equals the sum of the per-group means by linearity). Reports, per cell and
per head: mean / sd / [min, max] of the per-prompt label-share sum, tagged
``main`` (unified floor-0 set) or ``recpos-only``; plus the across-head
distribution summary (min / median / max of per-head means) over the MAIN set
and over the full RECPOS set.

    PYTHONPATH=$PWD .venv/bin/python scripts/step23_label_share_summary.py \
        [--json out.json] [--markdown out.md]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from subspaces.paths import ProjectPaths

LABEL_GROUP = re.compile(r"^demo\d+_output$")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _head_key(head) -> str:
    layer_idx, head_idx = head
    return f"{int(layer_idx)}:{int(head_idx)}"


def summarize_cell(paths: ProjectPaths, recpos_artifact: str, unified_artifact: str):
    recpos_node = paths.resolve(recpos_artifact).parent
    artifacts = sorted(recpos_node.glob("step3-tokengroups-*.json"))
    if not artifacts:
        return None
    if len(artifacts) > 1:
        raise RuntimeError(
            f"{recpos_node}: {len(artifacts)} step-3 artifacts; expected one "
            "(smoke siblings should not live in wave nodes)"
        )
    manifest = _load_json(artifacts[0])
    group_order = manifest["group_order"]
    label_columns = [
        index for index, name in enumerate(group_order) if LABEL_GROUP.match(name)
    ]
    if not label_columns:
        raise RuntimeError(f"{artifacts[0]}: no demo output groups in {group_order}")
    npz = np.load(recpos_node / manifest["outcomes"]["file"])

    main_set = {
        _head_key(head)
        for head in _load_json(paths.resolve(unified_artifact))["main_heads"]
    }
    recpos_heads = [
        _head_key(head)
        for head in _load_json(paths.resolve(recpos_artifact))["main_heads"]
    ]

    per_head = {}
    for key in recpos_heads:
        layer_idx, head_idx = key.split(":")
        shares = npz[f"L{layer_idx}H{head_idx}_shares"]
        label_sum = shares[:, label_columns].sum(axis=1)
        per_head[key] = {
            "role": "main" if key in main_set else "recpos_only",
            "mean": float(label_sum.mean()),
            "sd": float(label_sum.std(ddof=1)) if label_sum.size > 1 else 0.0,
            "min": float(label_sum.min()),
            "max": float(label_sum.max()),
            "n": int(label_sum.size),
        }

    def distribution(keys):
        means = sorted(per_head[key]["mean"] for key in keys)
        if not means:
            return None
        return {
            "n_heads": len(means),
            "min": means[0],
            "median": float(np.median(means)),
            "max": means[-1],
        }

    return {
        "artifact": str(artifacts[0]),
        "n_prompts": int(manifest["n_prompts_seen"]),
        "n_skipped_grouping": int(manifest["n_skipped_grouping"]),
        "per_head": per_head,
        "distribution_main": distribution([k for k in recpos_heads if k in main_set]),
        "distribution_recpos": distribution(recpos_heads),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--cells", default="configs/step23_recpos_cells.tsv")
    parser.add_argument("--json", default=None, help="write full JSON here")
    parser.add_argument("--markdown", default=None, help="write summary table here")
    args = parser.parse_args()
    paths = ProjectPaths.from_root(args.root)

    results: dict[str, dict] = {}
    for line in paths.resolve(args.cells).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        slug, _config, recpos, unified = line.split("\t")
        cell = summarize_cell(paths, recpos, unified)
        if cell is None:
            print(f"[label-share] {slug}: no step-3 artifact yet, skipped")
            continue
        results[slug] = cell

    rows = [
        "| cell | label-share mean over MAIN heads (min/med/max) | over RECPOS heads (min/med/max) | n_heads (main/recpos) | skipped |",
        "|---|---|---|---|---|",
    ]
    for slug, cell in results.items():
        main_distribution = cell["distribution_main"]
        recpos_distribution = cell["distribution_recpos"]
        rows.append(
            f"| {slug} "
            f"| {main_distribution['min']:.3f} / {main_distribution['median']:.3f} / {main_distribution['max']:.3f} "
            f"| {recpos_distribution['min']:.3f} / {recpos_distribution['median']:.3f} / {recpos_distribution['max']:.3f} "
            f"| {main_distribution['n_heads']}/{recpos_distribution['n_heads']} "
            f"| {cell['n_skipped_grouping']} |"
        )
    table = "\n".join(rows)
    print(table)
    if args.json:
        Path(args.json).write_text(
            json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[label-share] wrote {args.json}")
    if args.markdown:
        Path(args.markdown).write_text(table + "\n", encoding="utf-8")
        print(f"[label-share] wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
