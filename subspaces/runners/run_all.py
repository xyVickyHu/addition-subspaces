"""End-to-end orchestrator. Runs Step 1, then passes Step-1 main heads on.

    python -m subspaces.runners.run_all --config configs/step1_number_add_llama3.yaml \\
        --through step1

``--through step2``/``step3`` will chain the later steps once the chaining is
wired (deferred); today they fail clearly and never create
fake successful artifacts. Step 2 itself is implemented STANDALONE — run
``python -m subspaces.runners.step2`` on the composed run's heads artifact.
"""

from __future__ import annotations

import argparse
import sys

from subspaces.artifacts import ArtifactError
from subspaces.config import ConfigError, load_step1_config
from subspaces.paths import ProjectPaths
from subspaces.step1 import pipeline
from subspaces.step1.samples import HoldoutViolation
from subspaces.step1.selectors import SelectorError
from subspaces.step1.split import TaskSplitError

THROUGH = ("step1", "step2", "step3")

_CLI_ERRORS = (
    ArtifactError,
    ConfigError,
    HoldoutViolation,
    NotImplementedError,
    SelectorError,
    TaskSplitError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subspaces.runners.run_all", description=__doc__
    )
    parser.add_argument("--root", default=None)
    parser.add_argument("--config", required=True, help="Step-1 variant YAML")
    parser.add_argument("--through", default="step1", choices=THROUGH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.from_root(args.root)
    cfg = load_step1_config(paths.resolve(args.config))
    try:
        pipeline.run(cfg, paths)
        if args.through in ("step2", "step3"):
            raise NotImplementedError(
                "run_all --through step2/step3 chaining is not wired yet "
                "(deferred). Step 2 is implemented "
                "standalone: python -m subspaces.runners.step2 --heads-artifact "
                "<main-node>/heads.json --context <ctx.yaml>."
            )
        return 0
    except _CLI_ERRORS as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
