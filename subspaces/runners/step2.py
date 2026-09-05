"""Step-2 CLI. Argparse only — computation lives in ``subspaces.step2``.

    # dry run (writes nothing)
    python -m subspaces.runners.step2 --heads '15:2,15:1,13:6' --dry-run

    # PCA of the main heads of a selector node over its cell's z caches
    python -m subspaces.runners.step2 \\
        --heads-artifact log/runs/<node>__<id12>/selected-largest_gap-v1/scan-<tag>/main-largest_gap-v1-<h8>/main_heads.json \\
        --head-set main --context configs/contexts/<cell>.yaml

Execution requires ``--context``; the artifact (kind ``step2_subspace``) lands
next to the heads artifact by default (``--out-dir`` overrides; explicit
``--heads`` runs must pass ``--out-dir``).
"""

from __future__ import annotations

import argparse
import sys

from subspaces.artifacts import ArtifactError
from subspaces.context import ContextError, load_context
from subspaces.head_sets import HeadSpecError, resolve_heads_arg
from subspaces.paths import ProjectPaths
from subspaces.step2 import api

_CLI_ERRORS = (
    ArtifactError,
    ContextError,
    HeadSpecError,
    NotImplementedError,
    OSError,
    ValueError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subspaces.runners.step2", description=__doc__
    )
    parser.add_argument("--root", default=None)
    parser.add_argument("--heads", default=None, help="explicit 'L:H,L:H,...' list")
    parser.add_argument(
        "--heads-artifact",
        default=None,
        help="Step-1 head artifact: composed heads.json or a selector's "
        "main_heads.json",
    )
    parser.add_argument(
        "--head-set", default="main", choices=("main", "selected", "minor")
    )
    parser.add_argument(
        "--context",
        default=None,
        help="analysis-context YAML (model, task split, prompt protocol, "
        "sample manifests, activation refs); see subspaces/context.py",
    )
    parser.add_argument(
        "--z-cache",
        action="append",
        default=[],
        help="z cache to analyze: a 16-hex fingerprint under log/cache/z/ or "
        "a cache directory path; repeatable (merged caches must share "
        "model/dataset identity and cover disjoint tasks). Overrides "
        "context.activations.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory for the step2_subspace artifact (default: the "
        "heads artifact's node directory; required with explicit --heads)",
    )
    parser.add_argument("--n-pcs", type=int, default=6)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="cumulative explained-variance threshold (default 0.95)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="write per-head cumulative-variance plots next to the artifact",
    )
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        help="optional addition-specific analyses (periodic, causal_projection). "
        "periodic = the §4 mod-vector fit over add-k tasks (subspaces/step2/periodic.py); "
        "causal_projection refuses (use step1 evaluate-headset --subspace)",
    )
    parser.add_argument(
        "--theta-step",
        type=float,
        default=0.01,
        help="periodic plugin: phase-search grid step for opt_shift "
        "(default 0.01 rad)",
    )
    parser.add_argument(
        "--gt-mod-vectors",
        default=None,
        help="periodic plugin: ground-truth mod_vectors_dict.pth to compare "
        "against, e.g. "
        "artifacts/matrix_add_0204_clip_lambda0.05/mod_vectors_dict.pth "
        "(llama add trio + (15,17) + (15,28)); heads absent from the dict "
        "report null GT fields",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.from_root(args.root)
    cfg = api.Step2Config(
        n_pcs=args.n_pcs,
        threshold=args.threshold,
        plugins=list(args.plugin),
        theta_step=args.theta_step,
        gt_mod_vectors=args.gt_mod_vectors,
    )
    try:
        context = load_context(paths.resolve(args.context)) if args.context else None
        # Precedence: explicit --heads > --heads-artifact > context.heads_artifact
        heads_artifact = args.heads_artifact
        if args.heads is None and heads_artifact is None and context is not None:
            heads_artifact = context.heads_artifact_path()
        heads = resolve_heads_arg(
            heads=args.heads,
            heads_artifact=heads_artifact,
            head_set=args.head_set,
            paths=paths,
        )
        if args.dry_run:
            print(api.describe(cfg, heads, paths, context, z_caches=args.z_cache))
            return 0
        api.run(
            cfg,
            heads,
            paths,
            context,
            head_set=args.head_set if args.heads is None else "explicit",
            heads_artifact=heads_artifact if args.heads is None else None,
            z_caches=args.z_cache,
            out_dir=args.out_dir,
            plots=args.plots,
        )
        return 0
    except _CLI_ERRORS as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
