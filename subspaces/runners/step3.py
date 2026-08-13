"""Step-3 CLI. Argparse only — see ``subspaces.step3.api`` / ``subspaces.step3.token_groups``.

    # dry run (no context needed)
    python -m subspaces.runners.step3 --heads '15:2,15:1,13:6' --dry-run

    # token-group FV-share decomposition for the largest_gap MAIN heads
    python -m subspaces.runners.step3 \\
        --heads-artifact log/runs/<node>__<id>/sig-<m>-v1/scan-<tag>/main-largest_gap-v1-<h8>/heads.json \\
        --head-set main --context configs/contexts/<context>.yaml

Execution requires ``--context`` (model, task protocol, sample manifest,
z-cache); explicit ``--heads`` additionally requires ``--out-dir``.
"""

from __future__ import annotations

import argparse
import sys

from subspaces.artifacts import ArtifactError
from subspaces.context import ContextError, load_context
from subspaces.head_sets import HeadSpecError, resolve_heads_arg
from subspaces.paths import ProjectPaths
from subspaces.step3 import api


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subspaces.runners.step3", description=__doc__
    )
    parser.add_argument("--root", default=None)
    parser.add_argument("--heads", default=None, help="explicit 'L:H,L:H,...' list")
    parser.add_argument("--heads-artifact", default=None, help="Step-1 heads.json")
    parser.add_argument(
        "--head-set", default="main", choices=("main", "significant", "minor")
    )
    parser.add_argument(
        "--context",
        default=None,
        help="analysis-context YAML (model, task split, prompt protocol, "
        "sample manifests, activation refs); see subspaces/context.py",
    )
    parser.add_argument("--n-prompts-per-task", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-boot", type=int, default=2000, help="bootstrap resamples for CIs"
    )
    parser.add_argument(
        "--samples-name",
        default="analysis",
        help="which context.samples entry provides the analysis prompts",
    )
    parser.add_argument(
        "--interval",
        default="var",
        choices=api.PLOT_INTERVALS,
        help="bar-chart interval around the mean (presentation only)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="artifact directory (default: the heads artifact's node dir; "
        "required with explicit --heads)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.from_root(args.root)
    cfg = api.Step3Config(
        n_prompts_per_task=args.n_prompts_per_task,
        seed=args.seed,
        n_boot=args.n_boot,
        samples_name=args.samples_name,
        interval=args.interval,
    )
    try:
        context = load_context(paths.resolve(args.context)) if args.context else None
        # Precedence: explicit --heads > --heads-artifact > context.heads_artifact
        # (an empty-string --heads counts as absent, so provenance stays honest)
        heads_arg = args.heads or None
        heads_artifact = args.heads_artifact
        if heads_arg is None and heads_artifact is None and context is not None:
            heads_artifact = context.heads_artifact_path()
        elif (
            heads_arg is None
            and heads_artifact is not None
            and context is not None
            and context.heads_artifact_path()
        ):
            # A CLI artifact silently outranking a context that pins its own
            # heads (fingerprint included) would break lineage honesty: the
            # two sources must agree in CONTENT (path aliasing, e.g. a
            # _legacy mirror, is fine).
            from subspaces.artifacts import semantic_fingerprint
            from subspaces.head_sets import read_heads_manifest

            cli_manifest, _ = read_heads_manifest(paths.resolve(heads_artifact))
            context_manifest, _ = read_heads_manifest(
                paths.resolve(context.heads_artifact_path())
            )
            if semantic_fingerprint(cli_manifest) != semantic_fingerprint(
                context_manifest
            ):
                raise ArtifactError(
                    "--heads-artifact disagrees with the context's "
                    f"heads_artifact ({heads_artifact} vs "
                    f"{context.heads_artifact_path()}); refusing mismatched "
                    "head sources."
                )
        heads = resolve_heads_arg(
            heads=heads_arg,
            heads_artifact=heads_artifact,
            head_set=args.head_set,
            paths=paths,
        )
        if args.dry_run:
            print(api.describe(cfg, heads, paths, context))
            return 0
        api.run(
            cfg,
            heads,
            paths,
            context,
            heads_artifact=heads_artifact if heads_arg is None else None,
            out_dir=args.out_dir,
        )
        return 0
    except (
        ArtifactError,
        ContextError,
        HeadSpecError,
        NotImplementedError,
        ValueError,
    ) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
